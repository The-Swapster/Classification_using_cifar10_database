import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#print the data type of x_train
print(type(x_train))
#print the data type of y_train
print(type(y_train))
#print the data type of x_test
print(type(x_test))
#print the data type of y_test
print(type(y_test))
#Get the shape of x_train
print('x_train shape:', x_train.shape)
#Get the shape of y_train
print('y_train shape:', y_train.shape)
#Get the shape of x_test
print('x_test shape:', x_test.shape)
#Get the shape of y_test
print('y_test shape:', y_test.shape)
#looking at the first image
index = 0
print(x_train[index])
#looking as an image
img = plt.imshow(x_train[index])
#printing the label of the image
print('The image label is: ', y_train[index])
#show the label classification in relation to the number
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print the image class
print('The image class is: ', classification[y_train[index][0]])
#one hot encoding to convert the labels into a set of 10 numbers
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
#print all of the new labels
print(y_train_one_hot)
#print an example of the new labels
print('The one hot label is: ', y_train_one_hot[0])
#Normalize the pixels
x_train = x_train/255
x_test = x_test/255
#building the convolution neural network model
#we need to create an architecture using Sequential()
model = Sequential()
#adding first layer, a convolution layer to extract features from the input image
#and create 32, 5 x 5 ReLu convoluted features aldo known as feature maps
#Since this is the first layer we must input the dimension shape which is a 32 x 32
#pixel image with ddepth = 3 (RGB)
model.add(Conv2D(32, (5, 5), activation = 'relu', input_shape = (32, 32, 3)))
#max pooling with a 2 x 2 pixel filter
model.add(MaxPooling2D(pool_size = (2, 2)))
#creating one more convolution layer and pooling layer without the input_shape
model.add(Conv2D(64, (5, 5), activation = 'relu'))
model.add(Conv2D(32, (5, 5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
#add a flattening layer, to reduce the image to a linear also known
#as a one 1-Dimension vector to feed into and connect with the neural network
model.add(Flatten())
#first layer has 1000 neurons and the activation function ReLu
model.add(Dense(1000, activation = 'relu'))
#add a drop out layer with 50% drop out
model.add(Dropout(0.5))
#neural network where the first layer has 500 neurons and the activation function ReLu
model.add(Dense(500, activation = 'relu'))
#drop out layer with 50% drop out
model.add(Dropout(0.5))
#neural network where the first layer has 250 neurons and activation function ReLu
model.add(Dense(250, activation = 'relu'))
#last layer of this neural nerwork with 10 neurons with teh softmax function
model.add(Dense(10, activation = 'softmax'))
#complile the model
#give it the categorical_crossentropy loss function which is used for classes greater than 2, adam optimizer, and the accuracy of the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Train the model using the fit() method, which is other word for
#train. We will train the model on the training data with batch size = 256, 
#epochs = 20, and split the data into training on 80% of the data and
#using the other 20% as validation. Training may take some time to finish.
#batch: Total number of training examples present in a single batch
#Epoch: The number of iterations when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
hist = model.fit(x_train, y_train_one_hot, batch_size = 256, epochs = 20, validation_split = 0.1)
#get the models accuracy on the test data
model.evaluate(x_test, y_test_one_hot)[1]
#visualize teh models accuracy for both the training and validaation data
#visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()
#Visualize the models loss for both the training and validation data
#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()
my_image = plt.imread('mhcv-construck.jpg') 
#show the uploaded image
img = plt.imshow(my_image)
#resize the image
from skimage.transform import resize
my_image_resize = resize(my_image, (32, 32, 3))
img = plt.imshow(my_image_resize)
#get the probability for each class
import numpy as np
probabilities = model.predict(np.array([my_image_resize,]))
#print the probabilities
probabilities
#number to class 
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0, :])
print('Most likely class: ', number_to_class[index[9]], '--probability: ', probabilities[0, index[9]])
print('Second most likely class: ', number_to_class[index[8]], '--probability: ', probabilities[0, index[8]])
print('Third most likely class: ', number_to_class[index[7]], '--probability: ', probabilities[0, index[7]])
print('Forth most likely class: ', number_to_class[index[6]], '--probability: ', probabilities[0, index[6]])
print('Fifth most likely class: ', number_to_class[index[5]], '--probability: ', probabilities[0, index[5]])
