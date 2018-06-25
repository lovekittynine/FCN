#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:21:17 2018

@author: sw
"""

# FCN32 train

import tensorflow as tf
from FCN32 import FCN32
import os
import numpy as np
from utils import get_train_imglist_label
import sys
import time
import keras
slim = tf.contrib.slim


model_dir = './fcn32_model'
VGG_MEAN = [123.68,116.779,103.939]   
 
def fcn32_train():
    
    # Train_Epochs = 20
    with tf.name_scope('placeholder'):
        xs = tf.placeholder(tf.float32,shape=[None,None,None,3])
        ys = tf.placeholder(tf.float32,shape=[None,None,None,21])
    
    with tf.name_scope('get_batch'):
        img_list,label_list = get_train_imglist_label()

        img_path,label = tf.train.slice_input_producer([img_list,label_list],
                                                       num_epochs=200,
                                                       shuffle=True)
        label = slim.one_hot_encoding(label,num_classes=21)
        contents = tf.read_file(img_path)
        image = tf.image.decode_jpeg(contents,channels=3)
        # cubic interp
        image = tf.image.resize_images(image,size=(224,224),method=3)
        
        image = image/1.0
        image -= VGG_MEAN
        
        img_batch,lab_batch = tf.train.batch([image,label],batch_size=16,
                                             num_threads=2,
                                             allow_smaller_final_batch=True)
        
    with tf.name_scope('model'):
        fcn32 = FCN32()
        upscore32 = fcn32.bulid_model(xs)
#        fcn32 = fcn32_vgg.FCN32VGG()
#        fcn32.build(xs,train=True,debug=True)
#        upscore32 = fcn32.upscore
    
    with tf.name_scope('losses'):
        # loss = tf.losses.softmax_cross_entropy(ys,upscore32)
        logits = tf.reshape(upscore32,shape=[-1,21])
        labels = tf.reshape(ys,shape=[-1,21])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()
    
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.000001)
        train_op = optimizer.minimize(total_loss)
        
    with tf.name_scope('pixel_accuracy'):
        pixel_acc = tf.reduce_mean(keras.metrics.categorical_accuracy(ys,upscore32))
    
    with tf.name_scope('save_or_restore'):
        saver = tf.train.Saver(max_to_keep=1)
        restore = tf.train.Saver()
        
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            path = ckpt.model_checkpoint_path
            print('Loading from model %s'%path)
            restore.restore(sess,path)
            # delete early weights
            tf.gfile.DeleteRecursively(model_dir)
        else:
            tf.global_variables_initializer().run()
        try:
            step = 0
            while not coord.should_stop():
                start = time.time()
                imgs,labs = sess.run([img_batch,lab_batch])
                loss_value,_ = sess.run([total_loss,train_op],
                                        feed_dict={xs:imgs,ys:labs})
                end = time.time()
                duration = end-start
                step += 1
                print('\rStep-%5d Batch loss value-%8.4f Elapsed-%4.2f(Sec)'%(step,loss_value,duration),
                      end='',flush=True)
                if step%27==0:
                    accuracy = sess.run(pixel_acc,feed_dict={xs:imgs,ys:labs})
                    print('\nBatch pixel accuracy:%.3f'%accuracy)
                    print('Starting to save model')
                    saver.save(sess,os.path.join(model_dir,'model.ckpt'),global_step=step,
                               write_meta_graph=False)
                    print('Saving model finished!!!')
                  
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            print('\nTraining Finished!!!')  
        coord.join(threads)
                
        
               
       


def view_bar(step,total_step,loss,elapsed):
    
    rate = step/total_step
    rate_num = int(rate*40)
    r = '\r[%s%s]%d%%\tloss-%8.4f time-%.4f(Sec)/batch  %d/%d'%('>'*rate_num,'-'*(40-rate_num),
           rate*100,loss,elapsed,step,total_step)
    sys.stdout.write(r)
    sys.stdout.flush()
    
    '''
    for epoch in range(Train_Epochs):
            generator = generate_img_label()
            sys.stdout.write('\nEpoch-%d/%d\n'%(epoch+1,Train_Epochs))
            sys.stdout.flush()
            accuracy_list = []
            step = 0
            
            for i in generator:
                start = time.time()
                img = i[0]
                img = np.expand_dims(img,axis=0)
                lab = np.expand_dims(i[1],axis=0)
                lab = sess.run(tf.one_hot(lab,depth=21))
                loss_value,acc,_ = sess.run([total_loss,pixel_acc,train_op],
                                            feed_dict={xs:img,ys:lab})
                end = time.time()
                step += 1
                duration = end-start
                accuracy_list.append(acc)
                view_bar(step,422,loss_value,duration)
                             
            sys.stdout.write('\nEpoch-%d pixel_accuracy-%.4f\n'%(epoch+1,np.mean(accuracy_list)))
            sys.stdout.flush()
            # run one epoch save model    
            saver.save(sess,os.path.join(model_dir,'model.ckpt'),global_step=epoch+1,
                       write_meta_graph=False)
    '''
    
   
    
if __name__ == '__main__':
    fcn32_train()