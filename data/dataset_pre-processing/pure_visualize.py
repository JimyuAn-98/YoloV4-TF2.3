"""
Include function:
    1. Read xml file
    2. Visualize the bbox based on the xml
    3. Save the visualized image
"""

from absl import app, flags
import cv2 as cv
import numpy as np
import os
from absl.flags import FLAGS
from lxml import etree

flags.DEFINE_string('xml', './data/dataset/tiny_train.txt', 'path to the former annotation txt file')
flags.DEFINE_string('img_input', './data/dataset/tiny_train.txt', 'path to the former annotation txt file')
flags.DEFINE_string('img_output', './data/dataset/tiny_cut_train.txt', 'path to save processed annotation file')

def load_xml(xml, xml_path):
    path = os.path.join(xml_path, xml)
    root = etree.parse(path).getroot()
    bboxes = root.xpath('//object/bndbox')
    names = root.xpath('//object/name')
    return  bboxes, names


def pure_visualize():
    output_img_path = FLAGS.img_output
    xml_path = FLAGS.xml
    img_path = FLAGS.img_input
    xml_list = os.listdir(xml_path)

    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)

    count = 1

    for xml in xml_list:
        bboxes, names = load_xml(xml, xml_path)
        for b, n in zip(bboxes, names):
            name = n.text
            img_name = xml.split(".")[0] + '.JPG'
            im_p = os.path.join(img_path, img_name)
            img = cv.imread(im_p)
            visualized_img = draw_box(img, b, name)
            output_path = output_img_path + '/' + img_name
            cv.imwrite(output_path, visualized_img)
            print("finish No.%d origin image" % count)
            count += 1


def draw_box(img, b, name):
    category = name
    bx_min = int(b.find('xmin').text)
    by_min = int(b.find('ymin').text)
    bx_max = int(b.find('xmax').text)
    by_max = int(b.find('ymax').text)
    font = cv.FONT_ITALIC
    if category == 'MQ':
        text = 'MQ'
        cv.rectangle(img, (bx_min, by_min), (bx_max, by_max), (0, 255, 0), 1)
        cv.putText(img, text, (bx_min, by_min-5), font, 0.5, (0, 255, 0), 1)
    elif category == 'HQ':
        text = 'HQ'
        cv.rectangle(img, (bx_min, by_min), (bx_max, by_max), (0, 0, 255), 1)
        cv.putText(img, text, (bx_min, by_min - 5), font, 0.5, (0, 0, 255), 1)
    else:
        text = 'GQ'
        cv.rectangle(img, (bx_min, by_min), (bx_max, by_max), (255, 0, 0), 1)
        cv.putText(img, text, (bx_min, by_min - 5), font, 0.5, (255, 0, 0), 1)

    return img


def main(_argv):
    pure_visualize()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
