# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:27:36 2018

@author: Sabhijiit
"""

import cv2
import numpy as np
import pandas as pd
from math import floor, ceil
import tensorflow as tf
import glob
import re
import random 
import time

random.seed(0)

# dictionary with integer keys
label_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 
                 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 
                 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}

tf.reset_default_graph()

def prepare_image(img, threshold, toggle, session):
    ''' This function prepares the image for segmentation by highlighting the borders between different characters. '''
        
    # If the first and last rows as well as the first and last columns all contain black pixels only, then they form a border and are to be removed
    if not img[0].any() and not img[-1].any() and not img[:,0].any() and not img[:,-1].any():
        img = np.delete(img, (0), axis=0)
        img = np.delete(img, (-1), axis=1)
        img = np.delete(img, (-1), axis=0)
        img = np.delete(img, (0), axis=1)

    crop_img = image_cropper(img, 0)    # Crops the image to remove the extraneous white part
    init_cropped = image_cropper(img, 0)    # A copy of the cropped image for later use
    
    if toggle == 1:
        thresh_predict = segment_predictor(init_cropped, session, 0)[1]
        thresh_prob = segment_predictor(init_cropped, session, 0)[0]
#        print(thresh_prob)
        if thresh_predict == "w" and thresh_prob > 400:
#            print("Pre Thresh")
            return "w"
        elif thresh_predict == "m" and thresh_prob > 400:
#            print("Pre Thresh")
            return "m"
    
    ret,crop_img = cv2.threshold(crop_img, threshold, 255, cv2.THRESH_BINARY)  
        
    # Creates a column vector that is appended to crop_img to be used as an index column for crop_df
    index = np.zeros([crop_img.shape[0],1])
    for i in range(crop_img.shape[0]):
        index[i] = i
    
    crop_img = np.concatenate((index, crop_img), axis=1)
    
    crop_df = pd.DataFrame(data=crop_img[0:,1:], index=crop_img[0:,0])
    
    num = 0
    for column in crop_df:
        if not any(crop_df[column] != 255):     # Check if the given column only has 255 valued pixels
            crop_df[column] = 0                 # If it does, convert all pixel values to 0 to make a border
        else:                                   
            num = num + 1                       # Else increment num by 1. Num will be used to check if for a segment larger than 35 pixels in size, is overlapping segmentation required.
            
    # If num is equal to the number of columns in the segment, it means there are no boundaries, therefore the segment either contains one character or there is overlap between multiple characters.
    if num == crop_df.shape[1]:
        # There might be a chance that a very wide character, like "m" or "w" is present, rather than multiple overlapping characters. To check for this, they can be predicted first
        overl_predict = segment_predictor(init_cropped, session, 0)[1]
        overl_prob = segment_predictor(init_cropped, session, 0)[0]
        if overl_predict == "w" and overl_prob > 400:
#            print("Pre Overlap")
            return "w"
        elif overl_predict == "m" and overl_prob > 400:
#            print("Pre Overlap")
            return "m"
        # If neither "m" nor "w" proceed for attempting overlapping segmentation
        else:
#            print("ENTERING")
            return overlapping_segmentation(init_cropped, session)
    
    crop_img = crop_df.values   # convert dataframe back to numpy array
    # adds a column full of 0 valued pixels to the left and right edge of the image to ensure the first and last character are always extracted. IMPORTANT - DO NOT REMOVE
    crop_df.insert(0, '-1', 0)  
    crop_df.insert(crop_df.shape[1], '+1', 1) 
    
#    cv2.imshow("img", crop_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
    
    # Gets a list of indices of all the columns that are entirely filled with 0 values pixels
    index_list = []
    for column in crop_df:
        if not any(crop_df[column] == 255): 
            index_list.append(crop_df.columns.get_loc(column))
            
    # Gets a list of tuples of index_list values that act as borders of the segments, ie consecutive entries in index_list whose difference is greater than 1
    border_list = []
    for i in range(1, len(index_list)):
        if index_list[i] - index_list[i-1] > 1:
            border_list.append((index_list[i-1], index_list[i]))
 
    predicted_captcha = ""
    
    for i in border_list:
        if i[1] - i[0] <= 35:
            # If the difference of a given tuple from border_list is not greater than 35, it can be assumed the segment contains only one character.
            segment = init_cropped[:,i[0]:i[1]]
            # The following try-except block is to handle cases where the segment becomes empty after thresholding in shape_resizer() 
            try:
                prediction = segment_predictor(segment, session, 0)[1]
                probs = segment_predictor(segment, session, 0)[0]            
                if prediction == "m" or prediction == "w" or prediction == "d" and probs < 400:
#                    or prediction == "p" or prediction == "h" or prediction == "v" or prediction == "q" and probs < 350:
#                    print("Pre Prediction")
                    predicted_captcha = predicted_captcha + prepare_image(segment, 100, toggle, session)
#                elif prediction == "n" and probs < 250:
##                    print("N Prediction")
#                    predicted_captcha = predicted_captcha + prepare_image(segment, 100, toggle, session)
                else:
#                    print("skipped everything")
                    predicted_captcha = predicted_captcha + prediction
            except:
                predicted_captcha = predicted_captcha
        else:
            toggle = 1
            # If the difference of a given tuple is greater than 35, two cases are possible. Case One is the segment might contain the character "m" or "w" (which are usually of width > 35) or Case Two which is that there are multiple characters in the segment
            segment = init_cropped[:,i[0]:i[1]]
            predicted_captcha = predicted_captcha + prepare_image(segment, 100, toggle, session)   # Here, a recursive call to prepare_image is made to repartition the segment, using a smaller threshold value.
            
    return predicted_captcha


def segment_predictor(img_segment, session, interpolation_toggle):
    ''' Calls the TensorFlow CNN model to predict the character in the given image segment. '''
    # resizes the image to shape (28, 28) which is the required input format for the predictor model
    img_segment = shape_resizer(img_segment, interpolation_toggle)
    
    return cnn_model(img_segment, session)


def overlapping_segmentation(img_segment, session):
    ''' Is used to partition segments where characters are overlapping '''
    # Makes copies of img_segment for later use
    init_img = img_segment    
        
    ret, img_segment = cv2.threshold(img_segment,175,255,cv2.THRESH_BINARY)
    ret, img_segment1 = cv2.threshold(init_img,175,255,cv2.THRESH_BINARY)
    right_segment = np.zeros((img_segment.shape[0], img_segment.shape[1]))  # Creates an array of the same shape as img_segment, will be used for extraction of the right character.
    left_segment = np.zeros((img_segment.shape[0], img_segment.shape[1]))   # Creates an array of the same shape as img_segment, will be used for extraction of the left character
    
    # Initializes the mouse which is used for path tracing. Initial corrdinates are the last row, middle column
    mouse = (img_segment.shape[0]-1, img_segment.shape[1]//2)
    mouse_toggle = True
    j = 1
    # Checks if the cell the current mouse coordinates point to has a 255 valued pixel
    if img_segment[mouse[0]][mouse[1]] == 255:
        img_segment[mouse[0]][mouse[1]] = np.uint8(9)
        right_segment[mouse[0]][mouse[1]:] = 1
        left_segment[mouse[0]][:mouse[1]+1] = 1
    # If not, first shift left, then right to find a cell with a 255 valued pixel and set the coordinates of the mouse to point to that cell
    else:
        while mouse_toggle:
            if img_segment[mouse[0]][mouse[1]-j] == 255:
                mouse = (mouse[0], mouse[1]-j)
                img_segment[mouse[0]][mouse[1]] = np.uint8(9)
                right_segment[mouse[0]][mouse[1]:] = 1
                left_segment[mouse[0]][:mouse[1]+1] = 1
                mouse_toggle = False
            elif img_segment[mouse[0]][mouse[1]+j] == 255:
                mouse = (mouse[0], mouse[1]+j)
                img_segment[mouse[0]][mouse[1]] = np.uint8(9)
                right_segment[mouse[0]][mouse[1]:] = 1
                left_segment[mouse[0]][:mouse[1]+1] = 1
                mouse_toggle = False
            else:
                j=j+1
    mouse_init = mouse
    
    # Draws a path along a line of 255 valued pixels from the bottom to the top, that partitions the segment. The priority of the mouse in this snippet is to turn towards the right. This helps avoid getting stuck in right facing curves. Loop stops running if the mouse reaches the top row or completes 45 iterations
    path_toggle = True
    iterations = 0
    while path_toggle:
        if mouse[0] == 0 or iterations == 45:
            path_toggle = False
            break
        if img_segment[mouse[0]-1][mouse[1]] == 255:
            mouse = (mouse[0]-1,mouse[1])
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]-1][mouse[1]+1] == 255:
            mouse = (mouse[0]-1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != 0 and img_segment[mouse[0]-1][mouse[1]-1] == 255:
            mouse = (mouse[0]-1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]][mouse[1]+1] == 255:
            mouse = (mouse[0],mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != 0 and img_segment[mouse[0]][mouse[1]-1] == 255:
            mouse = (mouse[0],mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]+1][mouse[1]+1] == 255:
            mouse = (mouse[0]+1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and mouse[1] != 0 and img_segment[mouse[0]+1][mouse[1]-1] == 255:
            mouse = (mouse[0]+1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and img_segment[mouse[0]+1][mouse[1]] == 255:
            mouse = (mouse[0]+1,mouse[1])
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        iterations = iterations + 1
    # If 45 iterations were completed but the mouse was not able to reach the top, reinitialize the mouse
    if mouse[0] != 0 and path_toggle == False:
        path_toggle = True
        mouse = mouse_init
        iterations = 0   
        img_segment = img_segment1
    # This time, the priority of the mouse is to turn towards the left. This helps avoid getting stuck in left facing curves
    while path_toggle:
        if mouse[0] == 0 or iterations == 45:
            path_toggle = False
            break
        if img_segment[mouse[0]-1][mouse[1]] == 255:
            mouse = (mouse[0]-1,mouse[1])
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != 0 and img_segment[mouse[0]-1][mouse[1]-1] == 255:
            mouse = (mouse[0]-1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]-1][mouse[1]+1] == 255:
            mouse = (mouse[0]-1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != 0 and img_segment[mouse[0]][mouse[1]-1] == 255:
            mouse = (mouse[0],mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]][mouse[1]+1] == 255:
            mouse = (mouse[0],mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and mouse[1] != 0 and img_segment[mouse[0]+1][mouse[1]-1] == 255:
            mouse = (mouse[0]+1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and mouse[1] != img_segment.shape[1]-1 and img_segment[mouse[0]+1][mouse[1]+1] == 255:
            mouse = (mouse[0]+1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif mouse[0] != img_segment.shape[0]-1 and img_segment[mouse[0]+1][mouse[1]] == 255:
            mouse = (mouse[0]+1,mouse[1])
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        iterations = iterations + 1 
    
#    cv2.imshow("img", img_segment)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
    
    if mouse[0] != 0:   # If mouse still has not reached the top, then simply send the original image for prediction
        return segment_predictor(init_img, session, 0)[1]
    else:   # Else extract left and right characters from the segment
        # Extracts right character
        right_final = np.multiply(img_segment, right_segment)
        border_list = []
        for (x,y), value in np.ndenumerate(right_final):
            if right_final[x][y] == 9:
                border_list.append(y)
                right_final[x,:y+1] = 255
        right_final = right_final[:,min(border_list):]
        
        # Extracts left character
        left_final = np.multiply(img_segment, left_segment)
        border_list = []
        for (x,y), value in np.ndenumerate(left_final):
            if left_final[x][y] == 9:
                border_list.append(y)
                left_final[x, y:] = 255
        left_final = left_final[:,:max(border_list)] 
 
        return segment_predictor(left_final, session, 1)[1] + segment_predictor(right_final, session, 1)[1]


def cnn_model(char_img, sess):
    ''' Loads the saved TensorFlow CNN model from the disk '''
    img = np.reshape(char_img, (1,784))
    
    index = prediction.eval(feed_dict={x: img, keep_prob: 1.0}, session=sess)
    label = label_mapping[index[0]]
    probability = confidence.eval(feed_dict={x: img, keep_prob: 1.0}, session=sess)
#    print("Label: ", label, " -> Probability: ", probability)
    
#    cv2.imshow("img", char_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
    
    return probability, label


def shape_resizer(img, interpolation_toggle):
    ''' Resizes the input image to 28*28 size for input into the CNN model for prediction'''
    height = img.shape[0]
    width = img.shape[1]
    
    ret, img_thresh = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    
    img = image_cropper(img_thresh, 1)
    
    # Squares the image
    if height > width: # If height is greater, columns have to be added to increase width
        left_padding  = int(ceil((height - width)/2.0))
        right_padding  = int(floor((height - width)/2.0))
        left = np.full((img.shape[0], left_padding), np.uint8(255))
        right = np.full((img.shape[0], right_padding), np.uint8(255))
        squared_img = np.c_[left, img, right]
    else:   # If width is greater, rows have to be added to increase height
        top_padding  = int(ceil((width - height)/2.0))
        bottom_padding  = int(floor((width - height)/2.0))
        top = np.full((top_padding, img.shape[1]), np.uint8(255))
        bottom = np.full((bottom_padding, img.shape[1]), np.uint8(255))
        squared_img = np.r_[top, img, bottom]
    
    if interpolation_toggle == 1:
        new_img = cv2.resize(squared_img, (20,20), interpolation = cv2.INTER_NEAREST) # Resizes the image to shape (24,24) using Inter_Nearest interpolation for the already thresholeded image coming from overlapping segmentation
    else:
        new_img = cv2.resize(squared_img, (20,20), interpolation = cv2.INTER_CUBIC) # Resizes the image to shape (24,24) using Inter_cubic interpolation
    height = new_img.shape[0]
    width = new_img.shape[1]
    
    # Pads new_img with columns of 255 valued pixels to increase the size to (24,28)
    left_padding  = int(ceil((28 - width)/2.0))
    right_padding  = int(floor((28 - width)/2.0))
    left = np.full((new_img.shape[0],left_padding), np.uint8(255))
    right = np.full((new_img.shape[0],right_padding), np.uint8(255))
    img_width_resize = np.c_[left, new_img, right]
    
    resized_width = img_width_resize.shape[1]
    
    # Pads img_width_resize with rows of 255 valued pixels to increase the size to (28,28)
    top_padding  = int(ceil((28 - height)/2.0))
    bottom_padding  = int(floor((28 - height)/2.0))
    top = np.full((top_padding, resized_width), np.uint8(255))
    bottom = np.full((bottom_padding, resized_width), np.uint8(255))
    resized_img = np.r_[top, img_width_resize, bottom]
    
    return resized_img


def image_cropper(img, crop_toggle):
    ''' This function crops the image by removing all the extraneous whitespace near the borders '''
    # Extracts the top-most, bottom-most, left-most and right-most points non 255 values points respectively, in the captcha image, to be used for segmentation
    itemindex = np.where(img<255)    # Get index of all cells where pixel value is not 255.
    x_min = np.amin(itemindex[0])
    if x_min > 1:
        x_min -= 2
    elif x_min > 0:
        x_min -= 1
    x_max = np.amax(itemindex[0])
    if x_max < img.shape[0]-2:
        x_max += 2
    elif x_max <= img.shape[0]-1:
        x_max += 1
    y_min = np.amin(itemindex[1])
    y_max = np.amax(itemindex[1])
    if y_max < img.shape[1]-2:
        y_max += 2
    elif y_max < img.shape[1]-1:
        y_max += 1
    elif y_max == img.shape[1]-1 and crop_toggle == 1:
        y_max += 1
        
    crop_img = img[x_min:x_max, y_min:y_max]
    
    return crop_img


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

with tf.Session() as sess:
    # TensorFlow Variables, Placeholders and Operation Tensors    
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
    
    prediction = tf.argmax(y_conv, 1)
    confidence = tf.reduce_max(y_conv)
    
#    tf.train.Saver().restore(sess, r"captchured_project_directory\CNNmodel\model.ckpt")
    
    file_list = glob.glob(r"data_sets\Images\*.png")
    
    sub_list = random.sample(file_list, 2000)
    
#    test_list = ["spawist", "harvas", "smaped", "poidest", "arccks", "minpped", "buswer", "boofee", "horstly", "pagked", "jewlds", "liquary", "equily", "quading", "boxigua", "traeze", "cutzes", "yelzing", "mixfort", "jaznder", "judtest"]
#    test_list = ["madads", "madarly", "madcked", "madght", "madhers", "madhing", "mading", "madking", "madnds", "madses", "madtuck" , "madwer", "makbers", "makged", "makhes", "makhter", "makkel", "makkers", "makkes", "myswer"]
#    test_list = ["waiboat", "waiced", "waices", "waided", "waider", "waidest", "waiels", "waiered", "waiest", "waieszes", "waiing", "waiion", "waikers", "waikes", "waiking", "waiming", "waiost", "wairly", "waissy", "waiteps"]
#    test_list = ["damined", "daming", "damlers", "damoing", "damreded", "damrly", "damting", "damtly", "damway", "danced"]
#    test_list = ["ablteen", "acrfer", "actday", "addave", "addvers", "adjded", "adjlds", "aftager", "aftght",                 "aftner", "aftrtly", "aftted", "airally", "airday", "airden", "airdges", "airdly", "aireat", "airerve", "airews", "airghs", "airings", "airnge", "amofing", "amorney", "amucker", "angred", "aniger", "anirded", "ankfit", "harvas", "minpped", "smaped"]
    i = 0
    count = 0
    start = time.time()
    for img_path in file_list:
#        if img_path != "smaped":
#            continue
#        img_path = r"C:\Users\Sabhijiit\Desktop\captchured_project_directory\data_sets\Images\\"+img_path+".png"
        try:
            captcha_string = re.findall(r"\\[^\\][a-zA-Z]+.png", img_path)[0]
            captcha_string = captcha_string[1:-4].lower()
            if captcha_string != "buswer":
                continue
            count += 1
            print("\n=x=x=\n", count, " : ", captcha_string)
            img = cv2.imread(img_path, 0)
            
            result = prepare_image(img, 235, 0, sess) 
            print("CAPTCHA-string: ", captcha_string, ", CAPTCHA-predicted: ", result, "\n")
            if result == captcha_string:
                i += 1
                print("CORRECT => YES")
            else:
#                print("\n=x=x=\n", count, " : ", captcha_string)
#                print("CAPTCHA-string: ", captcha_string, ", CAPTCHA-predicted: ", result, "\n")
                print("CORRECT => NO")
#                print("=x=x=x=x=x=x=x=x=\n") 
            print("=x=x=x=x=x=x=x=x=\n") 
        except Exception as e:
#            print(e)
            pass
    end = time.time()
    print()
#    print("Accuracy: ", (i/len(file_list))*100)
    print("Accuracy: ", (i/len(sub_list)*1000))
    print("Time: ", end-start)
