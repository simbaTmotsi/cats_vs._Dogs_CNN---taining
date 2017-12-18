
# coding: utf-8

# # Using keras on the Dog vs. Cats dataset

# In[ ]:

# imports
import os
import distutils.dir_util
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.layers.core import Activation, Dropout
from keras.models import model_from_json
import glob
import numpy as np
from keras.preprocessing import image


# The assumption is you already have the data set if you don't you can download it

# here is the link ->https://www.kaggle.com/c/dogs-vs-cats/download/train.zip

# In[ ]:

# making directories to store our training data
distutils.dir_util.mkpath('Animal-data/train/dog')
distutils.dir_util.mkpath('Animal-data/train/cat')


# # Appending files to the directories

# In[ ]:

dest_dir = "Animal-data/train/dog/"

print ('Named explicitly:')
for image in glob.glob('Animal-data/train/dog*'):
    print ('\t', image)    
    shutil.move(image, dest_dir)    
print ("...Done moving dogs!")


# In[ ]:

dest_dir = "Animal-data/train/cat/"

print ('Named explicitly:')
for image in glob.glob('Animal-data/train/cat*'):
    print ('\t', image)    
    shutil.move(image, dest_dir)    
print ("...Done moving cats_!")


# # making the test directory

# In[ ]:


distutils.dir_util.mkpath('Animal-data/test/dog')
distutils.dir_util.mkpath('Animal-data/test/cat')


# ### adding files to the test directories

# In[ ]:

src = 'Animal-data/train/cat'
cat_test_dst = 'Animal-data/test/cat'

for path, subdirs, files in os.walk(src):
    cat_count = 0
    for name in files:
        f = path + '/' + name
        if name.split('.')[0] == 'cat':
            if cat_count < 2500: # 2500 are 20% of the training set [we are using 80-20 approach]
                cat_count += 1
                shutil.move(f, cat_test_dst)
            
print ("...Done making test set for cats...")


# In[ ]:

src = 'Animal-data/train/dog'
dog_test_dst = 'Animal-data/test/dog'

for path, subdirs, files in os.walk(src):
    dog_count = 0
    for name in files:
        f = path + '/' + name
        if name.split('.')[0] == 'dog':
            if dog_count < 2500:
                dog_count +=1
                shutil.move(f, dog_test_dst)
            
print ("___Done making test set for dogs___")


# # Done with packaging....

# ## Defining the CNN

# In[ ]:

model = Sequential()


# In[ ]:

model.add(Conv2D(32, (3, 3),padding='same', input_shape = (64, 64, 3), activation = 'relu'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))


# ### Defining our hidden layers

# In[ ]:

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))


# ### Defining the fully connected layer

# In[ ]:

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))


# Locating the training data

# In[ ]:

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True)

#rgb or grayscale
training_set = train_datagen.flow_from_directory('Animal-data/train/',
                                                 color_mode = "rgb",
                                                 target_size = (64, 64),
                                                 batch_size = 20,
                                                 shuffle = True,
                                                 class_mode = 'binary')


# Locating the test data

# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True)

testing_datagen = ImageDataGenerator(rescale = 1./255)

#rgb or grayscale
testing_set = testing_datagen.flow_from_directory('Animal-data/test/',
                                                 color_mode = "rgb",
                                                 target_size = (64, 64),
                                                 batch_size = 20,
                                                 shuffle = True,
                                                 class_mode = 'binary')


# In[ ]:

model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])


# #### fitting the model and now training

# In[ ]:

# steps_per_epoch - It should typically be equal to the number of samples of your dataset divided by the batch size. 
# validation_steps - It should typically be equal to the number of samples of your validation dataset divided by the batch size. 
history = model.fit_generator(training_set,
                          steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = testing_set,
                         validation_steps = 250)


# In[ ]:

print(history.history['val_acc'])


# # Our model in .json format and the weights in .h5 format

# In[ ]:

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
 serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

