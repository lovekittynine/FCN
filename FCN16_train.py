#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:20:08 2018

@author: sw
"""

# FCN8_train

import tensorflow as tf
from utils import get_train_imglist_label
import time
import os
from FCN16 import FCN16
import keras


VGG_MEAN = [123.68,116.779,103.939]
model_dir = './fcn16_model'

def FCN16_train():
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32,shape=[None,None,None,3])
        ys = tf.placeholder(tf.float32,shape=[None,None,None,21])
        
    with tf.name_scope('get_batch_data'):
        img_list,label_list = get_train_imglist_label()
        img_path,label = tf.train.slice_input_producer([img_list,label_list],
                                                       num_epochs=50)
        contents = tf.read_file(img_path)
        image = tf.image.decode_jpeg(contents,channels=3)
        # must have same data type to excute div or mul
        # resize and convert datatype to float32
        image = tf.image.resize_images(image,size=(224,224),method=2)
        # subtract VGG_MEAN
        image -= VGG_MEAN
        label = tf.one_hot(label,depth=21)
        img_batch,lab_batch = tf.train.batch([image,label],
                                             batch_size=16,
                                             num_threads=2,
                                             allow_smaller_final_batch=True)
    with tf.name_scope('model'):
        fcn16 = FCN16()
        upscore16 = fcn16.bulid_model(xs,is_train=True,class_num=21)
    
    with tf.name_scope('losses'):
        
#        logits = tf.reshape(upscore8,shape=[-1,21])
#        labels = tf.reshape(ys,shape=[-1,21])
#        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                                labels=labels)
#        loss = tf.reduce_mean(cross_entropy)
        
        logits = tf.nn.softmax(upscore16)
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(ys,logits))
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()
        
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0,trainable=False)
        lr = tf.train.exponential_decay(1e-6,global_step,decay_rate=0.99,decay_steps=100)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss,global_step)
    
    with tf.name_scope('pixel_accuracy'):
        pixel_acc = tf.reduce_mean(keras.metrics.categorical_accuracy(ys,upscore16))
        
    with tf.name_scope('saver_or_restore'):
        saver = tf.train.Saver(max_to_keep=1)
        restorer = tf.train.Saver()
        
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            path = ckpt.model_checkpoint_path
            print('>>>>>Loading model from %s'%path)
            restorer.restore(sess,path)
            tf.gfile.DeleteRecursively(model_dir)
        else:
            tf.global_variables_initializer().run()
        try:
            while not coord.should_stop():
                
                start = time.time()
                imgs,labs = sess.run([img_batch,lab_batch])
                loss_value,_ = sess.run([total_loss,train_op],
                                        feed_dict={xs:imgs,ys:labs})
                end = time.time()
                duration = end-start
                step = global_step.eval()
                
                print('\r>>>>>Step-%d Loss_value-%7.4f Elapsed-%.3f(Sec)'%(step,loss_value,duration),
                      end='',flush=True)
                if step%54 == 0:
                    print('\n>>>>>Starting saving model')
                    saver.save(sess,os.path.join(model_dir,'model.ckpt'),step,write_meta_graph=False)
                    print('>>>>>Saving model finished!!!')
                if step%27 == 0:
                    accuracy = sess.run(pixel_acc,feed_dict={xs:imgs,ys:labs})
                    print('\n>>>>>Batch pixel accuracy-%.3f'%accuracy)
        except tf.errors.OutOfRangeError:
            saver.save(sess,os.path.join(model_dir,'model.ckpt'),step,write_meta_graph=False)
            print('\nTraing Finished!!!')
            coord.request_stop()
        coord.join(threads)
                
        
if __name__ == '__main__':
    FCN16_train()
        
        