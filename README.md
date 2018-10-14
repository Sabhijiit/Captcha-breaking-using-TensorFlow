# Captcha-breaking using TensorFlow

This project aims to take captcha images and predict the string of alphabets given there.

It has three parts:
1. The first part involves segmenting the captcha images given in data_sets using dataset_creator.py and generating a data set for training and testing.
2. The second part involves training and testing a CNN on the data set generated from the previous step using train_test_model.py. The CNN has two convolutional and pooling layers each.
3. The third and last step involves taking in as input a random captcha and running it through captcha_breaker.py to predict the string.
