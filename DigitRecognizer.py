import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns


np.random.seed(2)
num_classes=10

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#sns.set(style='white', context='notebook', palette='deep')

# reading the data
test = pd.read_csv('/Users/rohit/Desktop/all/test.csv')
train = pd.read_csv('/Users/rohit/Desktop/all/train.csv')

Y_train = train['label']
X_train = train.drop(labels=["label"],axis=1)

# normalising the data values
X_train = X_train / 255.0
test = test / 255.0
#images converted to grayscale 

# reshape which will convert to a 28x28x1 array. height = width = 28. Gray scale so 1. RGB => 3
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# onehot encoding for label 
Y_train = to_categorical(Y_train, num_classes = 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=23)

model=Sequential()
model.add(Conv2D(filters = 32, kernel_size=(5,5),padding = "Same", activation = "relu",input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size=(5,5),padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size=(5,5),padding = "Same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size=(5,5),padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

optimizer1 = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss = 'categorical_crossentropy',optimizer=optimizer1,metrics = ['accuracy'])

# monitor: quantity to be monitored. factor: factor by which the learning rate will be reduced. 
# new_lr = lr * factor
# patience: number of epochs with no improvement after which learning rate will be reduced. 
# verbose: int, 0: quiet, 1: update messages.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

epochs = 2 #Increase epochs to say 25-30 to get an accuracy of 99.7
batch_size = 100

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] 
                              , callbacks=[learning_rate_reduction])

results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


