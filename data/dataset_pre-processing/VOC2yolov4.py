"""
Include function:
    1. Read xml file
    2. Cut image into 4 blocks, one contain the bbox, others not
    3. Re-calculate the coordinate of the bbox
    4. Save into xml and jpg
"""

import os
import cv2 as cv
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from lxml import etree

flags.DEFINE_string('anno_dir', '../../data/VOCtraintest_11-May-2012/VOCdevkit/VOC2012/Annotations', 'path to anno dir')
flags.DEFINE_string('classes', '../../data/classes/voc2012.names', 'path to a list of class names')
flags.DEFINE_string('train_output', '../../data/dataset/voc2012_train.txt', 'path to a file for train')
flags.DEFINE_string('test_output', '../../data/dataset/voc2012_test.txt', 'path to a file for test')


def convert_annotation(train_output, test_output, anno_dir, class_names):
    anno_list = os.listdir(anno_dir)

    train_list = np.random.randint(0, len(anno_list), (int(0.8 * len(anno_list))))

    with open(train_output, 'w') as train, open(test_output, 'w') as test:
        for i in range(len(anno_list)):
            xml = anno_list[i]

            an_p = os.path.join(anno_dir, xml)

            # Get annotation.
            root = etree.parse(an_p).getroot()
            path = root.xpath('//path')
            bboxes = root.xpath('//object/bndbox')
            names = root.xpath('//object/name')
            im_p = path[0].text

            annotation = im_p
            if not bboxes:
                if (i in train_list) is True:
                    annotation = annotation + '\n'
                    train.write(annotation)
                else:
                    annotation = annotation + '\n'
                    test.write(annotation)
            else:
                for box, n in zip(bboxes, names):
                    category = n.text
                    class_idx = str(class_names.index(category))

                    bxmin = box.find('xmin').text
                    bymin = box.find('ymin').text
                    bxmax = box.find('xmax').text
                    bymax = box.find('ymax').text

                    box_anno = ','.join([bxmin, bymin, bxmax, bymax, class_idx])

                    annotation = ':'.join([annotation, box_anno])

                if (i in train_list) is True:
                    annotation = annotation + '\n'
                    train.write(annotation)
                else:
                    annotation = annotation + '\n'
                    test.write(annotation)


def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    convert_annotation(FLAGS.train_output, FLAGS.test_output, FLAGS.anno_dir, class_names)
    print("Complete convert voc data!")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
