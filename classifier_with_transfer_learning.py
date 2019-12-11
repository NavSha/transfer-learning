import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

def create_model():
    #define the model
    conv_base = VGG16(weights = 'imagenet',include_top=False, input_shape=(150,150,3))
    model = models.Sequential()
    model.add(conv_base)
    #freeze the convolution base to avoid its weights from getting updated during training.
    conv_base.trainable = False
    # add dense layers on top of the conv base
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model


def train_model():
    model = create_model()
    #compile the model
    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['acc'])
    #callbacks to get insight into the model during training
    callbacks_list = [EarlyStopping(monitor = 'val_loss',patience = 4, verbose = 1),ModelCheckpoint(filepath = 'dogs_and_cats_tl_weights.h5',monitor = 'val_loss',verbose = 1, save_best_only=True), ReduceLROnPlateau(monitor = 'val_loss',factor = 0.1, patience = 3)]
    #train the model
    history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50, callbacks = callbacks_list)
    # save the model wights for later use
    model.save('dogs_and_cats_tl_weights.h5')
    # save the model as json
    model_json = model.to_json()
    with open("dogs_and_cats_tl_model.json", "w") as json_file:
        json_file.write(model_json)
    return history

def plot_loss_and_accuracy():
    #let's plot the training and validation losses and accuracies
    history = train_model()
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc)+1)

    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs,val_acc,'b',label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'ro', label = 'Training loss')
    plt.plot(epochs,val_acc,'r',label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_loss_and_accuracy()
