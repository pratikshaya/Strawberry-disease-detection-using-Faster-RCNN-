# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:13:34 2019

@author: bhuwan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 00:39:49 2019

@author: bhuwan
"""

#from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import os
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model

from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
#from keras_frcnn.simple_parser import get_data
from keras_frcnn.my_parser_train import get_data
from keras_frcnn import resnet as nn
#from keras_frcnn import resnet101 as nn
#from keras_frcnn import vgg as nn

sys.setrecursionlimit(40000)
record_path = 'records_new.csv'

train_path = "data_path"
C = config.Config()
C.model_path = 'model_frcnn.hdf5'
C.num_rois = 32
#C.base_net_weights = 'F:\\strawberry_disease_detectison\\keras_frcnn\\inception_resnet_v2_plant_net_weights.h5' # for plant net weight
C.base_net_weights = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' # for image net weight
print("image net weight loaded sucessfully")

all_imgs, classes_count, class_mapping = get_data(train_path) # load training data and labels

print("the total number of image ", len(all_imgs))

#print("the output of classes count is", classes_count)
#print("the output of class mapping are", class_mapping)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
    
    
inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = 'config.pickle'

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

#train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
#val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
#train_imgs = all_imgs[0:30000]
#val_imgs = all_imgs[30000:]
train_imgs = all_imgs

#number of data for each class during training
angular_leafspot = [s for s in train_imgs if s['bboxes'][0]['class'] == 'angular_leafspot']
anthracnose_fruit_rot = [s for s in train_imgs if s['bboxes'][0]['class'] == 'anthracnose_fruit_rot']
gray_mold = [s for s in train_imgs if s['bboxes'][0]['class'] == 'gray_mold']
leaf_blight = [s for s in train_imgs if s['bboxes'][0]['class'] == 'leaf_blight']
leaf_scorch = [s for s in train_imgs if s['bboxes'][0]['class'] == 'leaf_scorch']
leaf_spot = [s for s in train_imgs if s['bboxes'][0]['class'] == 'leaf_spot']
powdery_mildew_fruit = [s for s in train_imgs if s['bboxes'][0]['class'] == 'powdery_mildew_fruit']
powdery_mildew_leaf = [s for s in train_imgs if s['bboxes'][0]['class'] == 'powdery_mildew_leaf']
'''
#number of data for each class during validation and testing 
leaf_blight_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'leaf_blight']
tip_burn_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'tip_burn']
mildew_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'mildew']
leaf_spot_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'leaf_spot']
graymold_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'graymold']
angular_leaf_spot_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'angular_leaf_spot']
anthracnose_test = [s for s in val_imgs if s['bboxes'][0]['class'] == 'anthracnose']
'''
print('Num train samples {}'.format(len(train_imgs)))
#print('Num val samples {}'.format(len(val_imgs)))

print('Num angular leafspot samples for training {}'.format(len(angular_leafspot)))
print('Num anthracnose fruit rot samples for training{}'.format(len(anthracnose_fruit_rot)))
print('Num graymold samples for training {}'.format(len(gray_mold)))
print('Num leaf blight samples for training{}'.format(len(leaf_blight)))
print('Num leaf scorch samples for training {}'.format(len(leaf_scorch)))
print('Num leaf spot samples for training {}'.format(len(leaf_spot)))
print('Num powdery mildew fruit samples for training {}'.format(len(powdery_mildew_fruit)))
print('Num powdery milder leaf samples for training {}'.format(len(powdery_mildew_leaf)))

'''
print('Num leaf blight samples for testing {}'.format(len(leaf_blight_test)))
print('Num tip_burn samples for testing {}'.format(len(tip_burn_test)))
print('Num mildew samples for testing {}'.format(len(mildew_test)))
print('Num leaf_spot samples for testing {}'.format(len(leaf_spot_test)))
print('Num graymold samples for testing {}'.format(len(graymold_test)))
print('Num angular leaf spot samples for testing {}'.format(len(angular_leaf_spot_test)))
print('Num anthracnose samples for testing {}'.format(len(anthracnose_test)))
'''

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')

#data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (inceptionresnetv2 here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True) # inception = (?,?,?,1088) vgg =(?,?,?,1024)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
print("number of anchors", num_anchors)
rpn = nn.rpn(shared_layers, num_anchors)
print("rpn is",rpn[:2]) # load only x_class and x_regr from rpn function. ignores base_layers

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
print("classifier is ",classifier)

model_rpn = Model(img_input, rpn[:2])
#print(model_rpn.summary())
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

if not os.path.isfile(C.model_path):
    #If this is the begin of the training, load the pre-traind base network such as vgg-16
    try:
        print('This is the first time of your training')
        print('loading weights from {}'.format(C.base_net_weights))
        
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        print("loading sucess for rpn")
        #print("summary for model rpn", model_rpn.summary())
        model_classifier.load_weights(C.base_net_weights, by_name=True)
        print("loading sucess for classifier")
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')
    
    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    # If this is a continued training, load the trained model from before
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)
    
    # Load the records
    record_df = pd.read_csv(record_path)

    r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
    r_class_acc = record_df['class_acc']
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_loss_class_cls = record_df['loss_class_cls']
    r_loss_class_regr = record_df['loss_class_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    r_mAP = record_df['mAP']

    print('Already train %dK batches'% (len(record_df)))

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

total_epochs = len(record_df)
r_epochs = len(record_df)

epoch_length = 1500
num_epochs = 25
iter_num = 0

total_epochs += num_epochs

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
    
    r_epochs += 1

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_train)
            #print("the shape of x is", X.shape)
            #print("the shape of y is",Y)
            #print("the shape of img_data is",img_data)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)
            #print("rpn shape",P_rpn[0])

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            #print("the shape of X2",X2.shape)
            #print("the shape of Y1",Y1.shape)
            #print("the shape of Y2",Y2.shape)
            #print("the shape of IouS",IouS.shape)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                      ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time()-start_time)/60

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)
                
                new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                           'class_acc':round(class_acc, 3), 
                           'loss_rpn_cls':round(loss_rpn_cls, 3), 
                           'loss_rpn_regr':round(loss_rpn_regr, 3), 
                           'loss_class_cls':round(loss_class_cls, 3), 
                           'loss_class_regr':round(loss_class_regr, 3), 
                           'curr_loss':round(curr_loss, 3), 
                           'elapsed_time':round(elapsed_time, 3), 
                           'mAP': 0}
                
                record_df = record_df.append(new_row, ignore_index=True)
                record_df.to_csv(record_path, index=0)


                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')

###plotting accuracy and loss curve
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
plt.title('mean_overlapping_bboxes')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
plt.title('class_accuracy')

plt.show()

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
plt.title('loss_rpn_cls')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
plt.title('loss_rpn_regr')
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
plt.title('loss_class_cls')
plt.subplot(1,2,2)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
plt.title('loss_class_regr')
plt.show()

plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
plt.title('total_loss')
plt.show()
