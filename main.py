import glob
import matplotlib.pyplot as plt
import tensorflow as tf

#print(f'Tensorflow version {tf.__version__}')
#print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF" }')

SAMPLE_SIZE = (256, 256)
OUTPUT_SIZE = (1080, 1920)
EPOCHS = 20



def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    masks.append(tf.where(tf.equal(mask, float(0)), 1.0, 0.0))

    masks = tf.stack(masks, axis=2)  # обьеденяет тензоры
    masks = tf.reshape(masks, OUTPUT_SIZE + (1,))

    return image, masks


def augmentate_images(image, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)

    return image, masks



images = sorted(glob.glob('./data/peoples/*.jpg'))
masks = sorted(glob.glob('./data/masks/*.jpg'))

images_dataset = tf.data.Dataset.from_tensor_slices(images)

masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))


dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.repeat(60)

dataset = dataset.map(augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

images_and_masks = list(dataset.take(5))

t = len(dataset)
print(t)
train_dataset = dataset.take(t - 100).cache()
test_dataset = dataset.skip(t - 100).take(100).cache()

print(len(train_dataset), t)

train_dataset = train_dataset.batch(16)  # поменять на dataset
test_dataset = test_dataset.batch(16)




def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))  # 256x256x3


def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())
    return result


def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(1, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')


inp_layer = input_layer()

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

out_layer = output_layer(4)

# skip connection realization
x = inp_layer

downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])  # connecting layers

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)


#tf.keras.utils.plot_model(unet_like, show_shapes=True, dpi=72)

def dice_mc_metric(a, b):
    a = tf.unstack(a, axis=3)
    b = tf.unstack(b, axis=3)

    dice_summ = 0

    for i, (aa, bb) in enumerate(zip(a, b)):
        numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
        denomerator = tf.math.reduce_sum(aa + bb) + 1
        dice_summ += numenator / denomerator

    avg_dice = dice_summ / 1

    return avg_dice


def dice_mc_loss(a, b):
    return 1 - dice_mc_metric(a, b)


def dice_bce_mc_loss(a, b):
    return 0.3 * dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./model/",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


unet_like.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])
history_dice = unet_like.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=0, callbacks=[model_checkpoint_callback])

unet_like.save_weights('./model/')
