"""
Created on Sun Jun 16 19:52:15 2019

@author: bhuwan
"""
# Script to convert yolo annotations to voc format

import os
import xml.etree.cElementTree as ET
from PIL import Image

ANNOTATIONS_DIR_PREFIX = "converted_annotation_in_txt\\"

DESTINATION_DIR = "converted_labels\\"

def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root

def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("data_path\\VOC2012\\converted_XML\\{}.xml".format(file_prefix))

def read_file(file_path):
    file_prefix = file_path.split(".jpg")[0]
    file_path_data = "converted_annotation_in_txt\\" + file_path
    print("the file path data", file_path_data)
    image_file_name = "{}.jpg".format(file_prefix)
    img = Image.open("{}/{}".format("data_path\\VOC2012\\new_augmented_image", image_file_name))
    w, h = img.size
    with open(file_path_data, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            print(data[0],data[1],data[2],data[3],data[4])
            voc.append(data[0])
            voc.append(data[1])
            voc.append(data[2])
            voc.append(data[3])
            voc.append(data[4])
            voc_labels.append(voc)
            print(voc_labels)
        create_file(file_prefix, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path))

def start():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
        print(filename)
        if filename.endswith('txt'):
            read_file(filename)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    start()

    
