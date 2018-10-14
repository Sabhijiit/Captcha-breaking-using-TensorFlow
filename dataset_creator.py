# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:15:40 2018

@author: Sabhijiit
"""

import cv2
import numpy as np
import pandas as pd
from math import floor, ceil
import glob
import re

char_name_dict = {'a':-1, 'b':-1, 'c':-1, 'd':-1, 'e':-1, 'f':-1, 'g':-1, 'h':-1, 'i':-1, 
                  'j':-1, 'k':-1, 'l':-1, 'm':-1, 'n':-1, 'o':-1, 'p':-1, 'q':-1, 'r':-1, 
                  's':-1, 't':-1, 'u':-1, 'v':-1, 'w':-1, 'x':-1, 'y':-1, 'z':-1}

def prepare_image(img, string, threshold):
    
    captcha_string = string
    
    if not img[0].any() and not img[-1].any() and not img[:,0].any() and not img[:,-1].any():
        img = np.delete(img, (0), axis=0)
        img = np.delete(img, (-1), axis=1)
        img = np.delete(img, (-1), axis=0)
        img = np.delete(img, (0), axis=1)
    
    crop_img = image_cropper(img)
    init_cropped = image_cropper(img)
    
    ret,crop_img = cv2.threshold(crop_img,threshold,255,cv2.THRESH_BINARY)

    index = np.zeros([crop_img.shape[0],1])
    for i in range(crop_img.shape[0]):
        index[i] = i
        
    crop_img = np.concatenate((index, crop_img), axis=1)
    
    crop_df = pd.DataFrame(data=crop_img[0:,1:], index=crop_img[0:,0])
    
    num = 0
    for column in crop_df:
        if not any(crop_df[column] != 255):     
            crop_df[column] = 0                 
        else:                                   
            num = num + 1

#    if num == crop_df.shape[1]:
#        return overlapping_segmentation(init_cropped, captcha_string)
        # make predictions for "m" and "w" here to avoid any possible cut-through errors as well as entering into overlapping segmentation for these charcters
#         early_prediction = segment_predictor(init_cropped)
#         if not early_prediction == "m" or early_prediction == "w":
#             return overlapping_segmentation(init_cropped)
#         else:
#             predicted_captcha = predicted_captcha + early_prediction
        
    crop_img = crop_df.values
    crop_df.insert(0, '-1', 0)
    crop_df.insert(crop_df.shape[1], '+1', 1)
    
    index_list = []
    for column in crop_df:
        if not any(crop_df[column] == 255): 
            index_list.append(crop_df.columns.get_loc(column))
            
    border_list = []
    for i in range(1, len(index_list)):
        if index_list[i] - index_list[i-1] > 1:
            border_list.append((index_list[i-1], index_list[i]))
            
    if len(captcha_string) != len(border_list):
        return "ignoring"
                
    predicted_captcha = ""
    for index, i in enumerate(border_list):
#        print(index)
#        if i[1] - i[0] <= 35:
        segment = init_cropped[:,i[0]:i[1]]
        char = captcha_string[index]
#            index += 1
#            segment = image_cropper(segment)
        predicted_captcha = predicted_captcha + segment_predictor(segment, char)
#        else:
#            segment = init_cropped[:,i[0]:i[1]]
#            if captcha_string[index] == "m" or captcha_string[index] == "w":
#                captcha_string = captcha_string[index]
#            else:
#                captcha_string = captcha_string[index:]
#            predicted_captcha = predicted_captcha + prepare_image(segment, captcha_string, 100)
    return predicted_captcha

def segment_predictor(img_segment, char):
    ''' Calls the TensorFlow CNN model to predict the character in the given image segment. '''
    img_segment = shape_resizer(img_segment, char)
    return cnn_model(img_segment)

def overlapping_segmentation(img_segment, char_string):
    ''' Is used to partition segments where characters are overlapping '''
    init_img = img_segment
    ret, img_segment = cv2.threshold(img_segment,175,255,cv2.THRESH_BINARY)
    right_segment = np.zeros((img_segment.shape[0], img_segment.shape[1]))
    left_segment = np.zeros((img_segment.shape[0], img_segment.shape[1]))
    
    mouse = (img_segment.shape[0]-1, img_segment.shape[1]//2)
    mouse_toggle = True
    j = 1
    if img_segment[mouse[0]][mouse[1]] == 255:
        img_segment[mouse[0]][mouse[1]] = np.uint8(9)
        right_segment[mouse[0]][mouse[1]:] = 1
        left_segment[mouse[0]][:mouse[1]+1] = 1
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
        elif img_segment[mouse[0]-1][mouse[1]+1] == 255:
            mouse = (mouse[0]-1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif img_segment[mouse[0]-1][mouse[1]-1] == 255:
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
        elif img_segment[mouse[0]+1][mouse[1]+1] == 255:
            mouse = (mouse[0]+1,mouse[1]+1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif img_segment[mouse[0]+1][mouse[1]-1] == 255:
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
        
    if mouse[0] != 0 and path_toggle == False:
        path_toggle = True
        mouse = mouse_init
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
        elif img_segment[mouse[0]-1][mouse[1]-1] == 255:
            mouse = (mouse[0]-1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif img_segment[mouse[0]-1][mouse[1]+1] == 255:
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
        elif img_segment[mouse[0]+1][mouse[1]-1] == 255:
            mouse = (mouse[0]+1,mouse[1]-1)
            img_segment[mouse[0]][mouse[1]] = np.uint8(9)
            right_segment[mouse[0]][mouse[1]:] = 1
            left_segment[mouse[0]][:mouse[1]+1] = 1
        elif img_segment[mouse[0]+1][mouse[1]+1] == 255:
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
    
    if mouse[0] != 0:
        return segment_predictor(init_img, captcha_string)
    else:
        right_final = np.multiply(img_segment, right_segment)
        border_list = []
        for (x,y), value in np.ndenumerate(right_final):
            if right_final[x][y] == 9:
                border_list.append(y)
                right_final[:,:y] = 255
        right_final = init_img[:,min(border_list):]
        
        left_final = np.multiply(img_segment, left_segment)
        border_list = []
        for (x,y), value in np.ndenumerate(left_final):
            if left_final[x][y] == 9:
                border_list.append(y)
                left_final[:,y:] = 255
        left_final = init_img[:,:min(border_list)]

        return segment_predictor(left_final, char_string[0]) + segment_predictor(right_final, char_string[1])

def cnn_model(char_img):
    ''' Loads the saved TensorFlow CNN model from the disk '''
    img = np.reshape(char_img, (1,784))
    
#    with tf.Session() as sess:
#        tf.train.Saver().restore(sess, "/home/ec2-user/jupyter_notebooks/Nayan/tensorflow/Cnn_model2/model.ckpt")
#        prediction=tf.argmax(y_conv,1)
#        index = prediction.eval(feed_dict={x: img, keep_prob: 1.0}, session=sess)
#        label = label_mapping[index[0]]
    
    return "a"

def shape_resizer(img, char):
    ''' Resizes the input image to 28*28 size for input into the CNN model for prediction '''
    height = img.shape[0]
    width = img.shape[1]
    
    ret, img_thresh = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    img = image_cropper(img_thresh)
    
    if height > width:
        left_padding  = int(ceil((height - width)/2.0))
        right_padding  = int(floor((height - width)/2.0))
        left = np.full((img.shape[0], left_padding), np.uint8(255))
        right = np.full((img.shape[0], right_padding), np.uint8(255))
        squared_img = np.c_[left, img, right]
    else:
        top_padding  = int(ceil((width - height)/2.0))
        bottom_padding  = int(floor((width - height)/2.0))
        top = np.full((top_padding, img.shape[1]), np.uint8(255))
        bottom = np.full((bottom_padding, img.shape[1]), np.uint8(255))
        squared_img = np.r_[top, img, bottom]
    
    new_img = cv2.resize(squared_img, (20,20), interpolation = cv2.INTER_CUBIC)
        
    height = new_img.shape[0]
    width = new_img.shape[1]
    
    left_padding  = int(ceil((28 - width)/2.0))
    right_padding  = int(floor((28 - width)/2.0))
    left = np.full((new_img.shape[0],left_padding), np.uint8(255))
    right = np.full((new_img.shape[0],right_padding), np.uint8(255))
    img_width_resize = np.c_[left, new_img, right]
    
    resized_width = img_width_resize.shape[1]
    
    top_padding  = int(ceil((28 - height)/2.0))
    bottom_padding  = int(floor((28 - height)/2.0))
    top = np.full((top_padding, resized_width), np.uint8(255))
    bottom = np.full((bottom_padding, resized_width), np.uint8(255))
    resized_img = np.r_[top, img_width_resize, bottom]
    
#     ret, resized_img = cv2.threshold(resized_img, 200, 255, cv2.THRESH_BINARY)
    global char_name_dict
    if char in char_name_dict:
        char_name_dict[char] += 1
        print("char : ", char)
        print("count :", char_name_dict[char])
    path = r"C:\Users\Sabhijiit\Desktop\captchured\captcha_segmented_characters\0{0}{1}.png".format(char, char_name_dict[char])
    cv2.imwrite(path, resized_img)
# #     sys.exit(0)
#     global number
#     number = number + 1
#     if number == 6:
#         sys.exit(0)
#    print("Final Image")
#    cv2.imshow("resized_img.jpg",resized_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows

    return resized_img

def image_cropper(img):
    ''' This function crops the image by removing all the extraneous whitespace near the borders '''

    itemindex = np.where(img<255)
    x_min = np.amin(itemindex[0])
    if x_min > 1:
        x_min -= 2
    x_max = np.amax(itemindex[0])
    if x_max < img.shape[0]-2:
        x_max += 2
    y_min = np.amin(itemindex[1])
    if y_min > 1:
        y_min -= 2
    y_max = np.amax(itemindex[1])
    if y_max < img.shape[1]-2:
        y_max += 2
        
    crop_img = img[x_min:x_max, y_min:y_max]    
    
    return crop_img

file_list = (glob.glob(r"data_sets\Images\ablded.png"))
#print(len(file_list))
#img_path = r"C:\Users\Sabhijiit\Documents\Zauba\TensorFlowOCR\ImageSegmentation\Images\spawist.png"
#sys.exit(0)
j = 0
for i in file_list:
    try:
        captcha_string = re.findall(r'\\[^\\][a-zA-Z]+.png', i)[0]
        captcha_string = captcha_string[1:-4].lower()
        print(captcha_string)
        img = cv2.imread(i, 0)
        result = prepare_image(img, captcha_string, 235)       
    except Exception as e:
        print("-----EXCEPTION-----")
        print(e)
        pass
    print("CAPTCHA: ", result)
    print("length = ", len(result))
    print("")