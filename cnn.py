import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising CNN
classifier = Sequential()

#step1 - convolution
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu')) 
#                         (3x3)rgb matrix      ( , ,3rgb color)         relu= single point

#step2 - Pooling
classifier.add(MaxPooling2D(pool_size= (2,2)))

#adding a 2nd convolutional layer
classifier.add(Conv2D(32, (3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size= (2,2)))

#step3 - Flattening
classifier.add(Flatten())

#step4 - Full connection
classifier.add(Dense(units = 128, activation= 'relu'))
classifier.add(Dense(units = 1, activation= 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#                              ^supports windows/ operatng sys

#Part2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True) #keep img horizontal

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:/.../training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

training_set.class_indices


test_set = test_datagen.flow_from_directory(r'C:/..../test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
#batch_size = resize 32 img at once
# binary = more than 2 category

classifier.fit_generator(training_set, steps_per_epoch = 50, epochs = 10, validation_data= test_set, validation_steps = 50)

#steps_per_epoch = 100(no. of imgs)    
#epochs = 2  retians previous accuray and runs 2nd time thus increases accuracy 


#Part3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:/.../cat3.jpg', target_size= (64,64))
#dog = test_image
test_image = image.img_to_array(test_image)
#dog1 = test_image
#print(dog)
# in opencv we get direct pixel val not in keras so we use above func
test_image = np.expand_dims(test_image, axis = 0) # flattening in single col 1d array
print(test_image)
result = classifier.predict(test_image)
training_set.class_indices
result
if result[0][0] >= 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)




























