import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

# set directories to be used for creation of generators
base_dir = '/Users/NavSha/Documents/tensorflow-projects/cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

#augment training data
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range = 40, width_shift_range = 0.2, height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
# rescalling but no augmentation required for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)    #validation data is not augmented
#create generators for training and validation
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')

#define the model
conv_base = VGG16(weights = 'imagenet',include_top=False, input_shape=(150,150,3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

#freeze the convolution base to avoid its weights from getting updated during
#training. It has to be done be done before model compilation
conv_base.trainable = False

#compile the model
model.compile(optimizer = optimizers.RMSprop(lr=2e-5),loss = 'binary_crossentropy',metrics=['acc'])

#train the model
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

# save the model for later use
model.save('cats_and_dogs_small_4.h5')

#let's plot the training and validation losses and accuracies
import matplotlib.pyplot as plt

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs,val_acc,'bo',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'ro', label = 'Training loss')
plt.plot(epochs,val_acc,'ro',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
