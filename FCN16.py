#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:13:41 2018

@author: sw
"""

# FCN8

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import scipy.misc as misc

weight_path = './vgg16.npy'

class FCN16():
    
    def __init__(self):
        self.datadic = np.load(weight_path,encoding='latin1').item()
    
    def bulid_model(self,input_tensor,is_train=True,class_num=21):
        self.input_tensor = input_tensor
        with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],
                             weights_regularizer=slim.l2_regularizer(5e-4),
                             activation_fn=tf.nn.relu):
            
            with tf.name_scope('conv1_block'):
                self.conv1_1 = self.conv_layer(self.input_tensor,'conv1_1')
                self.conv1_2 = self.conv_layer(self.conv1_1,'conv1_2')
                self.pool1 = self.max_pool_layer(self.conv1_2,'pool1')
            with tf.name_scope('conv2_block'):
                self.conv2_1 = self.conv_layer(self.pool1,'conv2_1')
                self.conv2_2 = self.conv_layer(self.conv2_1,'conv2_2')
                self.pool2 = self.max_pool_layer(self.conv2_2,'pool2')
            with tf.name_scope('conv3_block'):
                self.conv3_1 = self.conv_layer(self.pool2,'conv3_1')
                self.conv3_2 = self.conv_layer(self.conv3_1,'conv3_2')
                self.conv3_3 = self.conv_layer(self.conv3_2,'conv3_3')
                self.pool3 = self.max_pool_layer(self.conv3_3,'pool3')
            with tf.name_scope('conv4_block'):
                self.conv4_1 = self.conv_layer(self.pool3,'conv4_1')
                self.conv4_2 = self.conv_layer(self.conv4_1,'conv4_2')
                self.conv4_3 = self.conv_layer(self.conv4_2,'conv4_3')
                self.pool4 = self.max_pool_layer(self.conv4_3,'pool4')
            with tf.name_scope('conv5_block'):
                self.conv5_1 = self.conv_layer(self.pool4,'conv5_1')
                self.conv5_2 = self.conv_layer(self.conv5_1,'conv5_2')
                self.conv5_3 = self.conv_layer(self.conv5_2,'conv5_3')
                self.pool5 = self.max_pool_layer(self.conv5_3,'pool5')
            with tf.name_scope('fc6'):
                self.fc6 = self.fc_layer(self.pool5,class_num,'fc6')
                self.fc6 = slim.dropout(self.fc6,keep_prob=0.5,is_training=is_train)
            with tf.name_scope('fc7'):
                self.fc7 = self.fc_layer(self.fc6,class_num,'fc7')
                self.fc7 = slim.dropout(self.fc7,keep_prob=0.5,is_training=is_train)
            with tf.name_scope('fc8'):
                self.score = self.fc_layer(self.fc7,class_num,'fc8')
                
            with tf.name_scope('fuse_pool4'):
                self.upscore2 = self.upscore_layer(self.score,
                                                   tf.shape(self.pool4),
                                                   'upsample_2',
                                                   ksize=4,upscale=2)
                self.score_pool4 = self.score_layer(self.pool4,'score_pool4')
                self.fuse_pool4 = tf.add(self.upscore2,self.score_pool4)
                print('Layer fuse_pool4 output_shape',self.fuse_pool4.get_shape().as_list())
            
            with tf.name_scope('fuse_pool3'):
#                self.upscore4 = self.upscore_layer(self.fuse_pool4,
#                                                   tf.shape(self.pool3),
#                                                   'upsample_4',
#                                                   ksize=4,upscale=2)
#                self.score_pool3 = self.score_layer(self.pool3,'score_pool3')
#                self.fuse_pool3 = tf.add(self.upscore4,self.score_pool3)
#                print('Layer fuse_pool3 output_shape',self.fuse_pool3.get_shape().as_list())
                self.upscore16 = self.upscore_layer(self.fuse_pool4,
                                                   tf.shape(self.input_tensor),
                                                   'umsample_16',
                                                   ksize=32,upscale=16)
                
#            with tf.name_scope('upsample_8'):
#                self.upscore8 = self.upscore_layer(self.fuse_pool3,
#                                                   tf.shape(self.input_tensor),
#                                                   'umsample_8',
#                                                   ksize=16,upscale=8)
#            
#                print('Layer upscore8 output_shape',self.upscore8.get_shape().as_list())                                 
                
        return self.upscore16
                
    def conv_layer(self,bottom,name):
        with tf.variable_scope(name):
            weights = self.datadic[name][0]
            biases = self.datadic[name][1]
            outnums = len(biases)
            weights_init = tf.constant_initializer(weights,dtype=tf.float32)
            biases_init = tf.constant_initializer(biases,dtype=tf.float32)
            output = slim.conv2d(bottom,num_outputs=outnums,
                                 kernel_size=[3,3],stride=1,
                                 weights_initializer=weights_init,
                                 biases_initializer=biases_init)
            print('Layer %s'%name,'weights_shape',weights.shape,'biases_shape',outnums)
            print('Layer %s'%name,'output_shape',output.get_shape().as_list())
            return output
     
        
    def max_pool_layer(self,bottom,name):
        with tf.variable_scope(name):
            pool = slim.max_pool2d(bottom,kernel_size=[2,2],padding='SAME')
            print('Layer %s'%name,'output_shape',pool.get_shape().as_list())
            return pool
    
    
    def fc_layer(self,bottom,num_class,name):
        
        with tf.variable_scope(name):
            weights,biases = self.weights_biases_reshape(name,num_class)
            outnums = len(biases)
            # print('weights',weights.shape,'bias',outnums)
            weights_init = tf.constant_initializer(weights,dtype=tf.float32)
            biases_init = tf.constant_initializer(biases,dtype=tf.float32)
            
            if name == 'fc6':
                output = slim.conv2d(bottom,num_outputs=outnums,
                                     kernel_size=[7,7],stride=1,
                                     weights_initializer=weights_init,
                                     biases_initializer=biases_init)
            elif name == 'fc7':
                output = slim.conv2d(bottom,num_outputs=outnums,
                                     kernel_size=[1,1],stride=1,
                                     weights_initializer=weights_init,
                                     biases_initializer=biases_init)
            else:
                # note using softmax
                output = slim.conv2d(bottom,num_outputs=outnums,
                                     kernel_size=[1,1],stride=1,
                                     weights_initializer=weights_init,
                                     biases_initializer=biases_init,
                                     activation_fn=None)   
                
            print('Layer %s'%name,'weights_shape',weights.shape,'biases_shape',outnums)
            print('Layer %s'%name,'output_shape',output.get_shape().as_list())
            return output
    
    
    def score_layer(self,bottom,name,num_class=21):
        with tf.variable_scope(name):
            output = slim.conv2d(bottom,num_outputs=num_class,kernel_size=[1,1],
                                 stride=1,padding='SAME',
                                 weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                 activation_fn=None)
        return output
            
            
    def weights_biases_reshape(self,name,num_class=21):
        weights = self.datadic[name][0]
        biases = self.datadic[name][1]
        out_channels = len(biases)
        if name=='fc6':
            weights = np.reshape(weights,newshape=(7,7,512,4096))
            return weights,biases
        elif name=='fc7':
            weights = np.reshape(weights,newshape=(1,1,4096,4096))
            return weights,biases
        elif name=='fc8':
            steps = out_channels//num_class+1
            new_weights = np.zeros(shape=(1,1,4096,num_class),dtype=np.float32)
            new_biases = np.zeros(shape=(num_class),dtype=np.float32)
            weights = np.reshape(weights,newshape=(1,1,4096,1000))
            for i in range(num_class):
                start = i*steps
                end = min(start+steps,1000)
                new_weights[:,:,:,i] = np.mean(weights[:,:,:,start:end],axis=3)
                new_biases[i] = np.mean(biases[start:end])
            return new_weights,new_biases
        else:
            raise Exception('The layer name not exist in model!!!')
        
        
    def get_deconv_filt(self,kernel_shape):
        kernel_size = kernel_shape[1]
        factor = kernel_size//2 + 1
        if kernel_size%2==1:
            center = factor - 1
        else:
            center = factor - 0.5
        # bilinear
        bilinear = np.zeros(shape=(kernel_size,kernel_size),dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                value = (1-abs((i-center)/factor))*(1-abs((j-center)/factor))
                bilinear[i,j] = value
        weights = np.zeros(shape=kernel_shape,dtype=np.float32)
        for i in range(kernel_shape[2]):
            weights[:,:,i,i] = bilinear
        init = tf.constant_initializer(weights,dtype=tf.float32)
        deconv = tf.get_variable('deconv_filt',shape=kernel_shape,
                                 initializer=init)
        return deconv
    
    
    def upscore_layer(self,bottom,out_shape,name,ksize=4,upscale=2):
        with tf.variable_scope(name):
            in_channels = bottom.get_shape()[3].value
            kernel_shape = [ksize,ksize,in_channels,in_channels]
            deconv_weights = self.get_deconv_filt(kernel_shape)
            # deconv filter weight decay
            weight_decay = slim.l2_regularizer(5e-4)(deconv_weights)
            tf.losses.add_loss(weight_decay)
            # output shape
            new_shape = tf.stack([out_shape[0],out_shape[1],out_shape[2],in_channels])
            upscore = tf.nn.conv2d_transpose(bottom,deconv_weights,
                                             output_shape=new_shape,
                                             strides=[1,upscale,upscale,1])
        return upscore
            
        
        
        

if __name__ == '__main__':

    img = np.random.normal(size=(1,224,224,3)).astype(np.float32)
    xs = tf.placeholder(tf.float32,shape=[1,224,224,3])
    fcn16 = FCN16()
    fc16 = fcn16.bulid_model(xs)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output = sess.run(fc16,feed_dict={xs:img})
        lab = np.argmax(output,axis=3)
        misc.imshow(lab[0])