#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:17:20 2018

@author: sw
"""

# preprocessing for generating ground truth label

import scipy.misc as misc
import numpy as np
import glob
import sys
import os
import matplotlib.pyplot as plt

# pixel value map to class
pixel_mapping_class = {(0,0,0):0}

img_dir = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/SegmentationClass/*'

gt_dir = '/home/sw/Downloads/VOC2007/VOCdevkit/VOC2007/SegmentationGroundTruth'

if not os.path.exists(gt_dir):
    os.mkdir(gt_dir)

def generate_gt(img_dir):
    img_paths = sorted(glob.glob(img_dir))
    total_nums = len(img_paths)
    for k,img_path in enumerate(img_paths):
        img = misc.imread(img_path)
        # label resize to 224x224
        # interp = 'nearest' to assure dont produce new pixel value
        img = misc.imresize(img,size=(224,224),interp='nearest')
        img_name = os.path.basename(img_path)
        view_bar(k+1,total_nums,img_name)
        row,col = img.shape[0],img.shape[1]
        gt = np.zeros((row,col),dtype=np.uint8)
        # pixel value not equal 0 coordinate
        x, y, _ = np.where(img!=0)
        for i in range(len(x)):
            pixel = img[x[i],y[i],:]
            if not sum(np.equal(pixel,[224,224,192])):
                if tuple(pixel) not in pixel_mapping_class.keys():
                    pixel_mapping_class[tuple(pixel)] = len(pixel_mapping_class)
                gt[x[i],y[i]] = pixel_mapping_class[tuple(pixel)]
#        plt.ion()
#        plt.imshow(gt,cmap='Set1')
#        plt.show()
        np.save(os.path.join(gt_dir,os.path.splitext(img_name)[0]+'.npy'),gt)
    # class map to pixel value
    class_mapping_pixel = {key:value for value,key in pixel_mapping_class.items()}
    np.save('./class_mapping_pixel.npy',class_mapping_pixel)

def view_bar(step,total,img_name):
    rate = step/total
    rate_num = round(rate*40)
    r = '\r[%s%s]%d%%\timage_name-%s %d/%s'%('>'*rate_num,'-'*(40-rate_num),
           rate*100,img_name,step,total)
    sys.stdout.write(r)
    sys.stdout.flush()
    

if __name__ == '__main__':
    generate_gt(img_dir)