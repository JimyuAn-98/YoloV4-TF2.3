"""
Include function:
    1. Read VOC xml file
    2. Segment the high resolution image into low resolution (608*608)
        2.1 A quarter of the images contain the bndbox
        2.2 The bndboxes' coordinates are re-calculated
        2.3 Three quarters of the images are pure background
    3. Save the new data into VOC format
"""

import os
import cv2 as cv
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from lxml import etree
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import csv
import math
import random

flags.DEFINE_string('xml_path', '../../data/VOCtraintest_11-May-2012/VOCdevkit/VOC2012/Annotations', 'path to anno dir')
flags.DEFINE_string('img_path', '../../data/classes/voc2012.names', 'path to a list of class names')
flags.DEFINE_string('img_output', '../../data/dataset/voc2012_train.txt', 'path to a file for train')
flags.DEFINE_string('img_v_output', '../../data/dataset/voc2012_train.txt', 'path to a file for train')
flags.DEFINE_string('csv_output', '../../data/dataset/voc2012_train.txt', 'path to a file for train')
flags.DEFINE_string('xml_output', '../../data/dataset/voc2012_train.txt', 'path to a file for train')


def data_visualize(param):
    xml_path, im_p_l, im_p_h, img_name, con_1, con_2 = param
    img_path = '/'.join([FLAGS.img_output, img_name])
    # print(FLAGS.img_v_output)
    img_v_path = '/'.join([FLAGS.img_v_output, img_name + '_v'])
    xml_output = '/'.join([FLAGS.xml_output, img_name])

    if os.path.exists(im_p_l):
        img = cv.imread(im_p_l)
        root = etree.parse(xml_path).getroot()
        boxes = root.xpath('//object/bndbox')
        # size = root.xpath('//size')
        name = root.xpath('//object/name')
        con_2.append(len(boxes))
        group_box(img, boxes, con_1, xml_output, img_name, img_path, name, img_v_path)
        print('Finish {}'.format(img_name))
    elif os.path.exists(im_p_h):
        img = cv.imread(im_p_h)
        root = etree.parse(xml_path).getroot()
        boxes = root.xpath('//object/bndbox')
        name = root.xpath('//object/name')
        con_2.append(len(boxes))
        group_box(img, boxes, con_1, xml_output, img_name, img_path, name, img_v_path)
        print('Finish {}'.format(img_name))
    else:
        pass


def group_box(img, boxes, con_1, xml_output, img_name, img_path, name, img_v_path):
    full_box = []
    for box in boxes:
        bxmin = int(box.find('xmin').text)
        bymin = int(box.find('ymin').text)
        bxmax = int(box.find('xmax').text)
        bymax = int(box.find('ymax').text)
        full_box.append([bxmin, bymin, bxmax, bymax])
    i = 1
    full_box = np.array(full_box)
    all_box = full_box.copy()
    group_box_core(full_box, all_box, img, con_1, xml_output, img_name, img_path, name, img_v_path, i)


def group_box_core(full_box, all_box, img, con, xml_output, img_name, img_path, name, img_v_path, i):
    xml_path = xml_output + '_%d' % i
    new_img_path = img_path + '_%d.jpg' % i
    new_img_v_path = img_v_path + '_%d.jpg' % i

    # full_box = np.array(full_box)
    # mini = full_box.min(axis=0)
    mini_index = full_box.argmin(axis=0)
    min_box = full_box[mini_index[0]]
    min_box = np.insert(min_box, 4, 0)
    min_x_cen = (min_box[0] + min_box[2]) / 2
    min_y_cen = (min_box[1] + min_box[3]) / 2
    group = np.zeros(shape=(1, 5))
    # group[0] = min_box
    another_group = []

    # background sampling
    cutimg_bg_1 = cut_img_bg(img.copy(), full_box)
    if cutimg_bg_1 is not None:
        cutimg_bg_1_name = img_name + '_%d_bg_1' % i
        cutimg_bg_1_dir = xml_output + '_%d_bg_1' % i
        cutimg_bg_1_path = img_path + '_%d_bg_1.jpg' % i
        cv.imwrite(cutimg_bg_1_path, cutimg_bg_1)
        gen_xml(cutimg_bg_1_dir, cutimg_bg_1_name, cutimg_bg_1_path, cutimg_bg_1.shape, None)

    cutimg_bg_2 = cut_img_bg(img.copy(), full_box)
    if cutimg_bg_2 is not None:
        cutimg_bg_2_name = img_name + '_%d_bg_2' % i
        cutimg_bg_2_dir = xml_output + '_%d_bg_2' % i
        cutimg_bg_2_path = img_path + '_%d_bg_2.jpg' % i
        cv.imwrite(cutimg_bg_2_path, cutimg_bg_2)
        gen_xml(cutimg_bg_2_dir, cutimg_bg_2_name, cutimg_bg_2_path, cutimg_bg_2.shape, None)

    cutimg_bg_3 = cut_img_bg(img.copy(), full_box)
    if cutimg_bg_3 is not None:
        cutimg_bg_3_name = img_name + '_%d_bg_3' % i
        cutimg_bg_3_dir = xml_output + '_%d_bg_3' % i
        cutimg_bg_3_path = img_path + '_%d_bg_3.jpg' % i
        cv.imwrite(cutimg_bg_3_path, cutimg_bg_3)
        gen_xml(cutimg_bg_3_dir, cutimg_bg_3_name, cutimg_bg_3_path, cutimg_bg_3.shape, None)

    for box in full_box:
        box_xcen = (box[0] + box[2]) / 2
        box_ycen = (box[1] + box[3]) / 2
        dis = math.sqrt((box_xcen - min_x_cen) ** 2 + (box_ycen - min_y_cen) ** 2)
        bbox = np.array([np.insert(box, 4, dis)])
        if dis <= 200:
            min_x_cen = (min_x_cen + box_xcen) / 2
            min_y_cen = (min_y_cen + box_ycen) / 2
            group = np.append(group, bbox, axis=0)
        else:
            another_group.append([i for i in bbox[0]][:-1])

    cut_image, cut_img_v, bboxes, annotation = cut_img(img, min_x_cen, min_y_cen, all_box, name)
    cv.imwrite(new_img_path, cut_image)
    cv.imwrite(new_img_v_path, cut_img_v)
    con.append(bboxes)

    gen_xml(xml_path, img_name, new_img_path, cut_image.shape, annotation)

    i += 1
    another_group = np.array(another_group)
    # another_group = np.reshape(ne)
    if not len(another_group) == 0:
        group_box_core(another_group, all_box, img, con, xml_output, img_name, img_path, name, img_v_path, i)


def gen_xml(xml_output, img_name, new_img_path, shape, annotation):
    xml_path = xml_output + '.xml'
    newline = '\n'
    indent = '\t'
    form = newline + indent

    root = ET.Element("annotation")
    root.text = form

    folder = ET.Element('folder')
    folder.text = 'TEST'
    folder.tail = form
    root.append(folder)

    filename = ET.Element('filename')
    filename.text = img_name
    filename.tail = form
    root.append(filename)

    path = ET.Element('path')
    path.text = new_img_path
    path.tail = form
    root.append(path)

    source = ET.Element('source')
    source.text = form + indent
    database = ET.Element('database')
    database.text = 'Unknown'
    database.tail = form
    source.tail = form
    source.append(database)
    root.append(source)

    size = ET.Element('size')
    size.text = form + indent
    width = ET.Element('width')
    width.text = '%d' % shape[1]
    width.tail = form + indent
    size.append(width)
    height = ET.Element('height')
    height.text = '%d' % shape[0]
    height.tail = form + indent
    size.append(height)
    depth = ET.Element('depth')
    depth.text = '%d' % shape[2]
    depth.tail = form
    size.append(depth)
    size.tail = form
    root.append(size)

    if annotation is None:
        segmented = ET.Element('segmented')
        segmented.text = '0'
        segmented.tail = newline
        root.append(segmented)
    else:
        segmented = ET.Element('segmented')
        segmented.text = '0'
        segmented.tail = form
        root.append(segmented)
        i = 0
        for objects in annotation:
            if i < len(annotation):
                objects.tail = form
                root.append(objects)
                i += 1
            else:
                objects.tail = newline
                root.append(objects)

    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding='utf-8', short_empty_elements=False)


def cut_img_bg(img, full_box):
    x_min = random.randint(0, img.shape[1] - 608)
    y_min = random.randint(0, img.shape[0] - 608)
    x_max = x_min + 608
    y_max = y_min + 608
    cutimg = None
    for i in range(100):
        flag_box_in = False
        for box in full_box:
            if ((y_max > box[1] and x_max > box[0]) and (x_min < box[0] and y_min < box[1])) \
                    or ((box[3] > y_min > box[1]) and (box[2] > x_min > box[0])):
                # cut_img_bg(img, full_box)
                flag_box_in = True
                break
            else:
                pass
        if not flag_box_in:
            cutimg = img[y_min:y_max, x_min:x_max]
            break
        else:
            continue
    return cutimg


def cut_img(img, xcen, ycen, full_box, name):
    """group = group[1:, :]
    group_index = np.lexsort(group.T[:4, :])
    group = group[group_index, :]
    xmin = group[0][0]
    ymin = group[0][1]
    xmax = group[-1][2]
    ymax = group[-1][3]
    xcen = int((xmax + xmin) / 2)
    ycen = int((ymax + ymin) / 2)"""
    # ADD DIVERTION
    rand = random.randint(-150, 150)

    xcen_dev = xcen + rand
    ycen_dev = ycen + rand
    if xcen_dev - 304 < 0:
        xcen_dev = 304
    elif xcen_dev + 304 > img.shape[1]:
        xcen_dev = img.shape[1] - 304
    if ycen_dev - 304 < 0:
        ycen_dev = 304
    elif ycen_dev + 304 > img.shape[0]:
        ycen_dev = img.shape[0] - 304

    cut_image = img[int(ycen_dev - 304):int(ycen_dev + 304), int(xcen_dev - 304):int(xcen_dev + 304)]

    bboxes = np.zeros(shape=(1, 2))
    cut_img_v = cut_image.copy()
    annotation = ET.Element('annotaion')
    newline = '\n'
    indent = '\t'
    form = newline + indent
    for box, n in zip(full_box, name):
        '''bx_min = int(box[0] - (xcen - 304))
        if bx_min < 0:
            bx_min = 0
        by_min = int(box[1] - (ycen - 304))
        if by_min < 0:
            by_min = 0
        bx_max = int(box[2] - (xcen - 304))
        if bx_max > 608:
            bx_max = 608
        by_max = int(box[3] - (ycen - 304))
        if by_max > 608:
            by_max = 608'''

        bx_cen = (box[0] + box[2]) / 2
        by_cen = (box[1] + box[3]) / 2
        b_l = box[2] - box[0]
        b_h = box[3] - box[1]
        dx = abs(xcen_dev - bx_cen)
        dy = abs(ycen_dev - by_cen)

        if dx > (b_l / 2 + 608 / 2) or dy > (b_h / 2 + 608 / 2):
            pass
        else:
            if dx <= (608 / 2 - b_l / 2):  # x axis full in
                L = b_l
            else:
                L = b_l / 2 - (dx - 608 / 2)

            if dy <= (608 / 2 - b_h / 2):  # y axis full in
                H = b_h
            else:
                H = b_h / 2 - (dy - 608 / 2)

            if (L * H) / (b_l * b_h) <= 0.5:  # less than 20% of the origin area
                pass
            else:
                if xcen_dev - bx_cen > 0:  # left side
                    if dx < (608 / 2 - b_l / 2):  # full in
                        new_bx_cen = 608 / 2 - dx
                    else:
                        new_bx_cen = L / 2
                else:  # right side
                    if dx < (608 / 2 - b_l / 2):
                        new_bx_cen = 608 / 2 + dx
                    else:
                        new_bx_cen = 608 - L / 2

                if ycen_dev - by_cen < 0:  # down side
                    if dy < (608 / 2 - b_h / 2):
                        new_by_cen = 608 / 2 + dy
                    else:
                        new_by_cen = 608 - H / 2
                else:  # up side
                    if dy < (608 / 2 - b_h / 2):
                        new_by_cen = 608 / 2 - dy
                    else:
                        new_by_cen = H / 2

                bx_min = int(new_bx_cen - L / 2)
                bx_max = int(new_bx_cen + L / 2)
                by_min = int(new_by_cen - H / 2)
                by_max = int(new_by_cen + H / 2)

                object_xml = ET.Element('object')
                object_xml.text = form + indent
                name_xml = ET.Element('name')
                name_xml.text = n.text
                name_xml.tail = form + indent
                object_xml.append(name_xml)

                bndbox = ET.Element('bndbox')
                bndbox.text = form + indent + indent
                xml_xmin = ET.Element('xmin')
                xml_xmin.text = str(bx_min)
                xml_xmin.tail = form + indent + indent
                bndbox.append(xml_xmin)

                xml_ymin = ET.Element('ymin')
                xml_ymin.text = str(by_min)
                xml_ymin.tail = form + indent + indent
                bndbox.append(xml_ymin)

                xml_xmax = ET.Element('xmax')
                xml_xmax.text = str(bx_max)
                xml_xmax.tail = form + indent + indent
                bndbox.append(xml_xmax)

                xml_ymax = ET.Element('ymax')
                xml_ymax.text = str(by_max)
                xml_ymax.tail = form + indent
                bndbox.append(xml_ymax)
                bndbox.tail = form
                object_xml.append(bndbox)

                '''if i <= (len(group) - 1):
                    object_xml.tail = form
                    annotation.append(object_xml)
                else:
                    # object_xml.tail = newline'''
                annotation.append(object_xml)

                width = bx_max - bx_min
                height = by_max - by_min
                size = width * height
                bboxes = np.append(bboxes, np.array([[size, (width / height)]]), axis=0)
                cut_img_v = cv.rectangle(cut_img_v, (bx_min, by_min), (bx_max, by_max), (0, 255, 0), 1)
    bboxes = bboxes[1:]
    return cut_image, cut_img_v, bboxes, annotation


def plot_result(num_bbox, bboxes):
    csv_path = FLAGS.csv_output
    # print(csv_path)

    x1, y1 = np.unique(num_bbox, return_counts=True)
    csv_path_1 = '/'.join([csv_path, '_1.csv'])
    cs = open(csv_path_1, 'w')
    writer = csv.writer(cs)
    writer.writerow(['num of bbox', 'num of img'])
    for i, j in zip(x1, y1):
        writer.writerow([i, j])
    cs.close()
    plt.figure(num=1)
    plt.scatter(x1, y1)
    plt.xlabel('num of bbox')
    plt.ylabel('num of img')
    plt.title("The Distribution of BBoxes' amounts")

    plt.figure(num=2)
    # print(bboxes)
    y2 = []
    for i in bboxes:
        for j in i:
            y2.append(j[0])
    # print(y2)
    y2 = np.array(y2)
    y2 = np.sort(y2)
    x2 = np.arange(1, len(y2) + 1)
    csv_path_2 = '/'.join([csv_path, '_2.csv'])
    cs = open(csv_path_2, 'w')
    writer = csv.writer(cs)
    writer.writerow(['size of bbox'])
    for i in y2:
        writer.writerow([i])
    cs.close()
    plt.plot(x2, y2)
    plt.ylabel('size of bbox')
    plt.title("The Distribution of BBoxes' Size")

    plt.figure(num=4)
    '''y4 = np.array([j[0][1] for j in [i for i in bboxes]])
    print(y4)'''
    y4 = []
    for i in bboxes:
        for j in i:
            y4.append(j[1])
    # print(y2)
    y4 = np.array(y4)
    y4 = np.sort(y4)
    x4 = np.arange(1, len(y4) + 1)
    csv_path_4 = '/'.join([csv_path, '_4.csv'])
    cs = open(csv_path_4, 'w')
    writer = csv.writer(cs)
    writer.writerow(['width / height'])
    for i in y4:
        writer.writerow([i])
    cs.close()
    plt.plot(x4, y4)
    plt.ylabel('width / height')
    plt.title('The Distribution of BBoxes width and Height')

    plt.show()


def main(_argv):
    xml_list = os.listdir(FLAGS.xml_path)
    datalist = []
    print('There is %d XMLs need to be process' % len(xml_list))
    con_1 = multiprocessing.Manager().list()
    con_2 = multiprocessing.Manager().list()

    for xml in xml_list:
        xml_path = os.path.join(FLAGS.xml_path, xml)
        img_name = xml.split(".")[0]
        im_p_l = os.path.join(FLAGS.img_path, img_name + '.jpg')
        im_p_h = os.path.join(FLAGS.img_path, img_name + '.JPG')
        datalist.append([xml_path, im_p_l, im_p_h, img_name, con_1, con_2])
        # data_visualize([xml_path, im_p_l, im_p_h, img_name, [], []])

    p = Pool(multiprocessing.cpu_count() * 2)
    # p = Pool(1)
    result = p.map(data_visualize, datalist)
    p.close()
    p.join()
    num_bbox = np.array(con_2)
    bboxes = np.array(con_1)
    # print(bboxes)
    plot_result(num_bbox, bboxes)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
