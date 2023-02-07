from PIL import Image


n = int(input())
size = [256, 256]

for i in range(n):
    im = Image.open(f'data/peoples/{i + 1}.jpg')
    im = im.resize((size[0], size[1]), Image.Resampling.LANCZOS)
    im.save(f'data/peoples/{i + 1}.jpg')