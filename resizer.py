from PIL import Image
import os


my_path = os.getcwd()
size = [256, 256]

for i in os.listdir(f'{my_path}\data\peoples'):
    im = Image.open(f'{my_path}\data\peoples\{i}')
    im = im.resize((size[0], size[1]), Image.Resampling.LANCZOS)
    im.save(f'{my_path}\data\peoples\{i}')
