import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os


my_path = os.getcwd()

for i in os.listdir(f'{my_path}/data/peoples'):
    name = i[:-4]
    tree = ET.parse(f'{my_path}/data/xmls/{name}.xml')
    root = tree.getroot()

    maska = Image.new('RGB', (256, 256), 'black')

    idraw = ImageDraw.Draw(maska)

    data = [[j.find('bndbox')[t].text for t in range(len(j.find('bndbox')))] for j in root.findall('object')]

    for t in range(len(data)):
        idraw.rectangle((int(data[t][0]), int(data[t][1]), int(data[t][2]), int(data[t][3])), fill='white')

    maska.save(f'{my_path}/data/masks/{name}.jpg')
