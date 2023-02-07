import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


n = int(input())

for i in range(n):
    tree = ET.parse(f'data/xmls/{i + 1}.xml')
    root = tree.getroot()

    maska = Image.new('RGB', (256, 256), 'white')

    idraw = ImageDraw.Draw(maska)

    data = [[j.find('bndbox')[t].text for t in range(len(j.find('bndbox')))] for j in root.findall('object')]

    for t in range(len(data)):
        idraw.rectangle((int(data[t][0]), int(data[t][1]), int(data[t][2]), int(data[t][3])), fill='black')

    maska.save(f'data/masks/{i + 1}.jpg')