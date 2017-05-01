from __future__ import print_function
import keras
# from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import scipy.io
import numpy as np
import os

import h5py

test_data_mat = scipy.io.loadmat(
    '/Users/peachie/Desktop/SEM1-10601-ML//HWs/HW9/keras/data_mat/test_data_reshaped.mat')
test_data = test_data_mat['data']
test_data = test_data.astype('float32')

model_path = '/Users/peachie/Desktop/SEM1-10601-ML//HWs/HW9/keras/trained_model/cifar3.h5'
# load model
model = keras.models.load_model(model_path)
# test performance

predictions = model.predict(test_data, verbose=1)
# predictions = model.predict_on_batch(test_data)
predictions_label = np.argmax(predictions, axis=1)
predictions_label = np.reshape(predictions_label,(predictions_label.shape[0],1))
np.savetxt('results.csv',predictions_label)

# TODO: remove when submitting
# compare ground truth
true_labels_mat = scipy.io.loadmat(
    '/Users/peachie/Desktop/SEM1-10601-ML//HWs/HW9/keras/data_mat/ground_truth.mat')
true_labels = true_labels_mat['labels']

# true_labels = keras.utils.to_categorical(true_labels, 3)

# acr = model.evaluate(test_data,true_labels,verbose=1)

# print(acr)

true_labels = np.array(true_labels)
acr = np.true_divide(np.count_nonzero(true_labels==predictions_label),true_labels.shape[0])
print(acr)

