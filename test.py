
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os

#load the model
model = load_model('dogs_and_cats_tl_weights.h5')
# summary of the model
model.summary()

#set up base and test directories
base_dir = '/Users/NavSha/Documents/tensorflow-projects/cats_and_dogs_small'
test_dir = os.path.join(base_dir,'test')

# loads the model and prints its accuracy over the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator,steps = 50)
print('Test accuracy: ',test_acc)
