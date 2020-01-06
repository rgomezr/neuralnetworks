import numpy as np # library used for array operations
from os import listdir # library from the system used to read files from a directory
from os import path # library used to check the veracity of files and folders
import time # library used to control time spent on training operations
import matplotlib.pyplot as plt # library used to plot graphs and figures
import seaborn as sb # library used to make heatmaps from confusion matrices
from matplotlib import image # library used to import an image as a vector
import math # library used for math operations
import tensorflow as tf # machine learning library
from keras.utils import to_categorical # function from keras to make the one_hot matrix from the labels
from keras.models import Sequential # function from keras to initialize a sequential model
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, Dropout, MaxPooling2D # layers to include in a keras model
from keras import backend as K # backend from keras
from sklearn.model_selection import train_test_split # used for splitting the data set into train and test set

# initializing variables
images = [] # list where images for training will be saved
targets = [] # list where labels for the corresponding images will be saved

# different paths for different dataset of training

#mainpath = '/Users/rgomezr/Documents/OneDrive/DOCUMENTS/TFG/dev/CNN/images_data/TRAIN_4_TEST_1_9/train_without_ham9/' # local mainpath #train with 4 hams path
#mainpath = '/Users/rgomezr/Documents/OneDrive/DOCUMENTS/TFG/dev/CNN/images_data/TRAIN_4_TEST_1_14/train_without_ham14/' # local mainpath #train with 4 hams path
#mainpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_9/train_without_ham9/' # remote mainpath # train with 4 hams and test with ham 9
#mainpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_14/train_without_ham14/' # remote mainpath # train with 4 hams and test with ham 14
#mainpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_1/train_without_ham1/' # remote mainpath # train with 4 hams and test with ham 1
mainpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_24/train_without_ham24/' # remote mainpath # train with 4 hams and test with ham 24
#mainpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_25/train_without_ham25/' # remote mainpath # train with 4 hams and test with ham 25

print('TRAIN PATH: ', mainpath) # to see through console path dataset

# going through folders of labels for the training data
# checking the veracity of each folder to be a folder
# going through each file image within the folder
# checking the veracity of each file to be a file of interest
# importing images and labels in images and targets respectively
for folder in listdir(mainpath): 
	if path.isdir(mainpath + folder): 
		for filename in listdir(mainpath + folder):
			if path.isfile(mainpath + folder + '/' + filename) and filename != '.DS_Store': 
				img = image.imread(mainpath + folder + '/' + filename)
				images.append(img)
				targets.append(filename[0])

# Creating array of input images
data = np.asarray(images)
data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2], 1]) # reshaping training images for the keras model
# Creating array of labels
targets = np.asarray(targets)
labels = to_categorical(targets) # enconding the labels with the one_hot method


print("training data shape: ", data.shape) # data.shape = (720, 512, 512, 1)
print("training labels shape: ", labels.shape) # labels.shape = (720, 6)

# to see an image of the training set
"""
plt.imshow(data[4, :, :], cmap='gray')
plt.show()
"""

# creating Keras model

model = Sequential()

# model 1

# Feature learning part
model.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (512, 512, 1)))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (3,3)))
model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (3, 3)))
# Classifier part
model.add(Flatten())
model.add(Dense(316, activation = 'relu'))
model.add(Dense(316, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

# showing a summary of the layers and parameters of the model created

model.summary()

# compiling the model defining the optimizer and loss function

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


print("Number of examples for permutation: ", data.shape[0])

# randomizing the training data making a simple permutation
permutation = list(np.random.permutation(data.shape[0]))
random_data = data[permutation, :, :, :]
random_labels = labels[permutation, :]

tic = time.time() # getting the tic time in order to know the start time of the process of training

# Fitting the randomized data and labels into the keras model
# Setting number of epochs and batch size for the training
# model.fit also randomizes the data
h = model.fit(random_data, random_labels, epochs = 15, batch_size = 32) # training method in which data and targets are passed with some specific parameters

toc = time.time() # getting the toc time in order to know the end time of the process of training

print("Time spent: ", (toc-tic)/60)

#predictions

# read data for test data for one ham

# different testpaths for different testing with different hams

#testpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_9/test_with_ham9/'
#testpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_14/test_with_ham14/'
#testpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_1/test_with_ham1/'
testpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_24/test_with_ham24/'
#testpath = '/mnt/shared/rgomez/TRAIN_4_TEST_1_25/test_with_ham25/'

# reading testing images and labels into a list

test_images = []
targets_data = []

for test_image in listdir(testpath):
    if path.isfile(testpath + test_image) and test_image != '.DS_Store':
        test_images.append(image.imread(testpath + test_image))
        targets_data.append(test_image[0])


test_data = np.asarray(test_images)
test_data = np.reshape(test_data, [test_data.shape[0], test_data.shape[1], test_data.shape[2], 1])
targets_data = np.asarray(targets_data)
labels_data = to_categorical(targets_data)


print("Test data shape: ", test_data.shape)
print("Test labels shape: ", labels_data.shape)


# evaluation processes
"""
getting predictions with the trained model with the test data as input and 
getting the index of the max value in order to compare it with the labels
"""
preds = np.argmax(model.predict(test_data), axis = 1) 
labels = np.argmax(labels_data, axis = 1) # getting the index of the max value of each column to compare it with the predictions in order to get pred results
results = preds == labels

"""
print(preds)
print(labels)
print(results)
"""

# making simple stats according to the results obtained in the predictions

correct = np.sum(results == True)
incorrect = np.sum(results == False)
print("Correct: ", correct, " Correct Acc: ", (correct/len(results))*100)
print("Incorrect: ", incorrect, " Incorrect Acc: ", (incorrect/len(results))*100)


# plotting 


#confusion matrix
confusion_matrix = tf.math.confusion_matrix(preds, labels)
cm = plt.figure(1)
heat_map = sb.heatmap(confusion_matrix, cmap="YlGnBu")
plt.ylabel('Predicted values')
plt.xlabel('Real values')
#cm.savefig('/mnt/shared/rgomez/testing/22_confusion_matrix.png')

#accuracy per epoch graph
accuracies = h.history['accuracy']
acc = plt.figure(2)
plt.plot(np.squeeze(accuracies))
plt.ylabel('Accuracy')
plt.xlabel('epochs')
#acc.savefig('/mnt/shared/rgomez/testing/22_accuracy')

#loss per epoch graph
losses = h.history['loss']
loss = plt.figure(3)
plt.plot(np.squeeze(losses))
plt.ylabel('Loss')
plt.xlabel('epochs')
#loss.savefig('/mnt/shared/rgomez/testing/22_loss')

#visualizing intermmediate activations output

get_linear_filters = K.function([model.layers[0].input, K.learning_phase()], model.layers[0].output)
get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[1].output)
filters_applied = get_linear_filters([random_data[0:2, :, :, :],0])
activations_output = get_activations([random_data[0:2, :, :, :],0])


filters = plt.figure(figsize=(8,8))
for i in range(32):
	ax = filters.add_subplot(6, 6, i + 1)
	ax.imshow(filters_applied[0][:, :, i], cmap = 'gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()

#filters.savefig('/mnt/shared/rgomez/testing/20_filters_applied')

activations = plt.figure(figsize=(8,8))
for i in range(32):
	ax = activations.add_subplot(6, 6, i + 1)
	ax.imshow(activations_output[0][:, :, i], cmap = 'gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.tight_layout()

#activations.savefig('/mnt/shared/rgomez/testing/20_activations')






