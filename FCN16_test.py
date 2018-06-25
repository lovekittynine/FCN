'''
Created on Mon May 21 17:43:30 2018

@author: sw

'''

# FCN16_test

import tensorflow as tf
from utils import generate_img_label
import numpy as np
import scipy.misc as misc
from FCN16 import FCN16
import os


model_dir = './fcn16_model'
output_img = './img_output_fcn16/'

def FCN16_test():
    
    class_mapping_pixel = np.load('./class_mapping_pixel.npy').item()
    
    if not tf.gfile.Exists(output_img):
        tf.gfile.MkDir(output_img)
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32,shape=[None,None,None,3])
        
    with tf.name_scope('model'):
        fcn16 = FCN16()
        upscore16 = fcn16.bulid_model(xs,is_train=False)
     
    with tf.name_scope('restore'):
        restorer = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            path = ckpt.model_checkpoint_path
            print('Loading from model %s'%path)
            restorer.restore(sess,path)
        else:
            print('Model dont exits')
            tf.global_variables_initializer().run()
            
        data_gen = generate_img_label(train=False)
        
        step = 0
        for img in data_gen: 
            # misc.imshow(img[0])
            image = np.expand_dims(img[0],axis=0)
            image_name = img[1]
            output16 = sess.run(upscore16,feed_dict={xs:image})
            label = np.argmax(output16[0],axis=2)
            # convert to color map
            row,col = label.shape
            segmented_img = np.zeros((row,col,3),dtype=np.uint8)
            step += 1
            for i in range(row):
                for j in range(col):
                    segmented_img[i,j,:] = class_mapping_pixel[label[i,j]]
            view_bar(step,image_name)
            misc.imsave(os.path.join(output_img,image_name+'.png'),segmented_img)
        print('saving done!!!')



def view_bar(step,img_name,total_nums=210):
    rate = step/total_nums
    rate_num = int(rate*40)
    r = '\r[%s%s]%d%%\timage_name-%s %d/%d'%('>'*rate_num,'-'*(40-rate_num),
           rate*100,img_name,step,total_nums)
    print(r,end='',flush=True)

    
if __name__ == '__main__':
    FCN16_test()