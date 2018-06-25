#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:23:31 2018

@author: sw
"""

# utils function
import glob 
import os
import numpy as np
import tensorflow as tf
import random
import scipy.misc as misc

img_dir = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/JPEGImages'
train_img_name = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt'
test_img_name = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt'
label_dir = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/SegmentationGroundTruth'

# RGB
VGG_MEAN = [123.68,116.779,103.939]



def generate_img_label(train=True):
    
    
    if train:
        img_path = train_img_name
        
        with open(img_path,'r') as file:
            img_names = file.readlines()
        random.shuffle(img_names)   
        for img_name in (img_names):
            img_path = os.path.join(img_dir,img_name.strip()+'.jpg')
            img = misc.imread(img_path).astype(np.float32)
            img -= VGG_MEAN
            label = np.load(os.path.join(label_dir,img_name.strip()+'.npy'))
            yield img,label
            
    else:
        img_path = test_img_name
        with open(img_path,'r') as file:
            img_names = file.readlines()
        for i in img_names:
            img_path= os.path.join(img_dir,i.strip()+'.jpg')
            img = misc.imread(img_path).astype(np.float32)
            img -= VGG_MEAN
            img_name = i.strip()
            yield img,img_name
            
 
def generate_train(step):
   
    img_path = train_img_name
    with open(img_path,'r') as file:
        img_names = file.readlines()
    nums = len(img_names)
    batch_size = 20
    
    img_batch = []
    label_batch = []
    
    start = step*batch_size
    end = min(start+batch_size,nums)
    
    if end==nums:
        random.shuffle(img_names)
    
    for img_name in (img_names[start:end]):
        img_path= os.path.join(img_dir,img_name.strip()+'.jpg')
        img = misc.imread(img_path).astype(np.float32)
        img -= VGG_MEAN
        label = np.load(os.path.join(label_dir,img_name.strip()+'.npy'))
        img_batch.append(img)
        # one_hot encoding
        label_batch.append(tf.one_hot(label,depth=21))
        
    return img_batch,label_batch


def get_train_imglist_label():
    img_list = []
    label_list = []
    with open(train_img_name,'r') as file:
        for img_name in file:
            name = img_name.strip()
            img_list.append(os.path.join(img_dir,name+'.jpg'))
            label = np.load(os.path.join(label_dir,name+'.npy'))
            label = tf.convert_to_tensor(label,dtype=tf.uint8)
            label_list.append(label)
    return img_list,label_list
    
      
if __name__ == '__main__':
   imgs,labs = generate_train(0)
    