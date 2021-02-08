"""
Include function:
    1. Read txt file
    2. Transform txt file into yolov4 txt file
    3. Save the images to a new direction
"""

from absl import app, flags
import shutil
import numpy as np
import os
from absl.flags import FLAGS

flags.DEFINE_string('i', './data/dataset/', 'path to the former annotation txt file')
flags.DEFINE_string('oa', './data/dataset/tiny_cut_train.txt', 'path to save processed annotation file')
flags.DEFINE_string('oi', './data/dataset/tiny_cut_train.txt', 'path to save processed annotation file')


def load_file():
    anno_path = FLAGS.i  # get the path of the former annotation txt file
    output_path = FLAGS.oa
    output_img_path = FLAGS.oi
    file_list = os.listdir(anno_path)

    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)

    if os.path.exists(output_path):    # if there is a annotation file which has the same name, than erase it
        os.remove(output_path)

    with open(output_path, 'w') as op:
        count = 1
        for files in file_list:
            file_name = files.split(".")
            image_name = file_name[0] + '.jpg'
            image_path = '/'.join([anno_path, image_name])
            new_image_path = '/'.join([output_img_path, image_name])
            if file_name[-1] == 'jpg':
                shutil.copy(image_path, new_image_path)
                # pass
            elif file_name[-1] == 'txt':
                txt_path = '/'.join([anno_path, files])
                with open(txt_path) as txt:
                    full_bbox = []
                    for lines in txt:
                        elements = lines.split()
                        attribute = elements[0]
                        bboxes = np.array([float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])]) * 416
                        bbox_xmin = int(bboxes[0] - bboxes[2]/2)
                        bbox_ymin = int(bboxes[1] - bboxes[3]/2)
                        bbox_xmax = int(bboxes[0] + bboxes[2]/2)
                        bbox_ymax = int(bboxes[1] + bboxes[3]/2)

                        '''img = cv.imread(image_path)
                        cv.rectangle(img, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), (0, 255, 0), 1)'''

                        bbox = ','.join([str(bbox_xmin), str(bbox_ymin), str(bbox_xmax), str(bbox_ymax), attribute])
                        full_bbox.append(bbox)
                    anno_bbox = ':'.join([i for i in full_bbox])
                    annotation = ':'.join([new_image_path, anno_bbox]) + '\n'
                    op.write(annotation)
                    print('finish %d file'%count)
                    count += 1


def main(_argv):
    load_file()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
