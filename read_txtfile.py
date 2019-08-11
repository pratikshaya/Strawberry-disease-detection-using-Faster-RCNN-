# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:50:04 2019

@author: bhuwan
"""

import numpy as np

datapath = "bb_box.txt"
savepath = "converted_annotation_in_txt\\"
filepath = "file_name.txt"

def make_dataset(input_path):
    all_imgs = {}
    print("the input path is", input_path)
    with open(input_path,'r') as f:
        print('Parsing annotation files')
        for line in f:
            line_split = line.strip().split(',')
            (filename,x1,y1,x2,y2,class_name) = line_split
            if filename not in all_imgs:
                all_imgs[filename] = {}
                
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['bboxes'] = []
                
            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
            
        all_data = []
        
        for key in all_imgs:
            all_data.append(all_imgs[key])
            
        return all_data
            
dataset = make_dataset(datapath)

for i in range(len(dataset)):
    dataset_path = dataset[i]['filepath']
    dataset_name = dataset_path.split(',')[0]
    sep_dataset = dataset[i]['bboxes']
    
    with open(filepath, 'a') as fn:
        file_name = "{}\n".format(dataset_name)
        fn.write(file_name)
        
    with open(savepath + str(dataset_name)+ ".txt", 'w') as f:
        for j in range(len(sep_dataset)):
            save_dataset = sep_dataset[j]
            
            data = "{} {} {} {} {} \n".format(sep_dataset[j]['class'],sep_dataset[j]['x1'],sep_dataset[j]['y1'], sep_dataset[j]['x2'], sep_dataset[j]['y2'])
            
            f.write(data)
    

           
