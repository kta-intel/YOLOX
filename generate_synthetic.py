import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pascal_voc_writer import Writer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-height", help="input height")
parser.add_argument("-width", help="input width")
parser.add_argument("-num", help="input number")

def normalize(im):
    min, max = im.min(), im.max()
    return (im.astype(float)-min)/(max-min)

def gen_image(height, width, num):
    for i in range(num):
        # Create image
        im = np.random.rand(width, height, 3)
        im = normalize(im)
        im = Image.fromarray(np.uint8(im * 255))
        im.save('./datasets/VOCdevkit/VOC2007/JPEGImages/' + str(i).zfill(2) + '.jpg')
        
        # Create annotations
        xmins = np.random.randint(0, width, 9).astype(int)
        xmaxs = []
        for xmin in xmins:
            xmaxs = np.append(xmaxs, np.random.randint(xmin, width)).astype(int)

        ymins = np.random.randint(0, height, 9).astype(int)
        ymaxs = []
        for ymin in ymins:
            ymaxs = np.append(ymaxs, np.random.randint(ymin, height)).astype(int)

        # create pascal voc writer (image_path, width, height)
        writer = Writer('./datasets/VOCdevkit/VOC2007/JPEGImages/' + str(i).zfill(2) + '.jpg', width, height, depth=3, database='generic')

        # add objects (class, xmin, ymin, xmax, ymax)
        for loc in range(len(xmins)):
            writer.addObject('peak', xmins[loc], xmaxs[loc], ymins[loc], ymaxs[loc])

        # write to file
        writer.save('./datasets/VOCdevkit/VOC2007/Annotations/' + str(i).zfill(2) + '.xml')

if __name__ == '__main__':
    args = parser.parse_args()
    gen_image(int(args.height), int(args.width), int(args.num))

    with open('./datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w') as f:
        for i in range(round(int(args.num)/2)):
            f.write(str(i).zfill(2) + "\n")

    with open('./datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w') as f:
        for i in range(round(int(args.num)/2),int(args.num)):
            f.write(str(i).zfill(2) + "\n")

    print("Complete")

