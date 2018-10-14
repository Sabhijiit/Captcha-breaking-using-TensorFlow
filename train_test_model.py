# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:32:37 2018

@author: Sabhijiit
"""

import cv2 # to load the images
import numpy as np # to do matrix manipulations 
from os.path import isfile, join # to manupulate file paths
from os import listdir # get list of all the files in a directory
from random import shuffle # shuffle the data (file paths)
import tensorflow as tf
import timeit

# sess = tf.InteractiveSession()

class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.folder_names = self.get_folder_names()
        self.file_names = self.get_data_paths()

    def get_folder_names(self):
        folder_names = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                folder_names.append(filename)
        return folder_names

    def get_data_paths(self):
        data_paths = []
        for label in self.folder_names:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'png':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths

generator_object = DataSetGenerator(r"C:\Users\Sabhijiit\Desktop\captchured_project_directory\data_sets\TrainDataSetFinal")    # Path of train data set

my_dict = {'a':0 , 'b':1 , 'c':2, 'd':3 , 'e': 4 , 'f':5 , 'g':6 , 'h':7 , 'i':8 , 'j':9 , 'k':10 , 'l':11 , 'm':12 , 'n':13 , 'o':14 , 'p':15 , 'q':16 , 'r':17 , 's':18 , 't':19 , 'u':20 , 'v':21 , 'w':22 , 'x':23, 'y':24 , 'z':25}

print("===STARTED===")
print()

n =  39000   # number of input images

input_images = np.zeros([n,784])    
input_labels = np.zeros([n,26])     
i = 0
for j in range(len(generator_object.folder_names)):
    for path in generator_object.file_names[j]:
        img = cv2.imread(path,0)
        input_images[i] = np.reshape(img, 784)
        input_labels[i,my_dict[generator_object.folder_names[j]]] = 1
        i=i+1
    print("processed" , generator_object.folder_names[j])

print() 
print("===FINISHED===")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
Understand why they have taken the number of features as 32 - TODO, not mentioned in the tutorial
'''

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_ = tf.placeholder(tf.float32, [None, 26])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

sess = tf.InteractiveSession()

start = timeit.default_timer()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Here, gradient descent optimizer is replaced with the more sophisticated ADAM optimizer.

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""to save model"""
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init_op)

n = 3001

for i in range(n):
    index = np.random.choice(input_images.shape[0], 50, replace=False)
    x_random = input_images[index]
    y_random = input_labels[index]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: x_random, y_: y_random, keep_prob: 1.0})
        print("step {0}, training accuracy {1} %".format(i, train_accuracy*100))
    train_step.run(feed_dict={x: x_random, y_: y_random, keep_prob: 0.5})
    
finish = timeit.default_timer()

time = (finish-start)/60

print("Total time taken for {0} iterations: {1} mins".format(n, time))

save_path = saver.save(sess, r"C:\Users\Sabhijiit\Desktop\captchured_project_directory\CNNmodel\model.ckpt")    # Give new path to save model
print("Model saved in path: %s" % save_path)

generator_object1 = DataSetGenerator(r"C:\Users\Sabhijiit\Desktop\captchured_project_directory\data_sets\TestDataSetFinal")    # Path to test data set

print("===STARTED===")
print()

m =  13000   # number of input images

test_images = np.zeros([m,784])
test_labels = np.zeros([m,26])
l=0
for j in range(len(generator_object1.folder_names)):
    for path in generator_object1.file_names[j]:
        img = cv2.imread(path,0)
        test_images[l] = np.reshape(img, 784)
        test_labels[l,my_dict[generator_object1.folder_names[j]]] = 1
        l=l+1
    print("processed" , generator_object1.folder_names[j])

print() 
print("===FINISHED===")
print()
print("==> Computing test accuracy")

accuracy = accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
print("===> Testing accuracy is {0} %".format(accuracy*100))
