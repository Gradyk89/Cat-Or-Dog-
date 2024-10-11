


import os
import shutil
import pandas as pd
import glob
import cv2
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.getcwd()
os.chdir(r"C:\Users\kellg\Desktop\CS451_Final_Project")

""""
train_files = glob.glob('train/*')
len_train = len(train_files)
print("length of train data:", len_train)

test_files = glob.glob('test1/*')
len_test = len(test_files)
print("length of test data:", len_test)

data_train = {}

data_train['cat'] = glob.glob('train/cat*')
data_train['dog'] = glob.glob('train/dog*')

print(f"count of cat in train data :  {len(data_train['cat'])}")
print(f"count of dog in train data :  {len(data_train['dog'])}")

count = 0
plt.figure(figsize= (15,15))
for label in data_train.keys():
    for path in data_train[label]:
        sp = plt.subplot(2 , 4 , count+1)
        sp.set_title(label)
        image = cv2.imread(path)
        plt.axis('off')
        plt.imshow(image)
        count+=1
        if count % 4 == 0:
            break
plt.show()

print("hello")

train_dir = ('training')
#os.mkdir(train_dir) #make a new train directory inside my base directory

print("hello1")

valid_dir= ('validation')
#os.mkdir(valid_dir) #make a new validation directory inside my base directory

print("hello2")

test_dir = ('testing')
#os.mkdir(test_dir) #make a new test directory inside my base directory

train_cats_dir = os.path.join(train_dir , 'cats')
#os.mkdir(train_cats_dir) #make a new cats directory inside my train directory

train_dogs_dir = os.path.join(train_dir , 'dogs')
#os.mkdir(train_dogs_dir) #make a new dogs directory inside my train directory

valid_cats_dir = os.path.join(valid_dir, 'cats')
#os.mkdir(valid_cats_dir) #make a new cats directory inside my validation directory

valid_dogs_dir = os.path.join(valid_dir , 'dogs')
#os.mkdir(valid_dogs_dir) #make a new dogs directory inside my validation directory

test_cats_dir = os.path.join(test_dir , 'cats')
#os.mkdir(test_cats_dir) #make a new cats directory inside my test directory

test_dogs_dir = os.path.join(test_dir , 'dogs')
#os.mkdir(test_dogs_dir) #make a new dogs directory inside my test directory

original_dataset_dir = 'train'


fnames = ['cat.{}.jpg'.format(i) for i in range(10000)] #Move 10,000 cat images from the original dataset to the train cat directory
for fname in fnames:
  src = os.path.join(original_dataset_dir , fname)
  dst = os.path.join(train_cats_dir , fname)
  shutil.move(src , dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(10000,11250)] #Move 1250 cat images from the original dataset to the validation cat directory
for fname in fnames:
  src = os.path.join(original_dataset_dir , fname)
  dst = os.path.join(valid_cats_dir , fname)
  shutil.move(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(11250,12500)] #Move 1250 cat images from the original dataset to the test cat directory
for fname in fnames:
  src= os.path.join(original_dataset_dir , fname)
  dst= os.path.join(test_cats_dir, fname)
  shutil.move(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(10000)] #Move 10,000 dog images from the original dataset to the train cat directory
for fname in fnames:
  src = os.path.join(original_dataset_dir , fname)
  dst = os.path.join(train_dogs_dir ,fname)
  shutil.move(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(10000,11250)] #Move 1250 dog images from the original dataset to the validation cat directory
for fname in fnames:
  src = os.path.join(original_dataset_dir,fname)
  dst = os.path.join(valid_dogs_dir , fname)
  shutil.move(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(11250,12500)]#Move 1250 dog images from the original dataset to the test cat directory
for fname in fnames:
  src= os.path.join(original_dataset_dir, fname)
  dst = os.path.join(test_dogs_dir , fname)
  shutil.move(src, dst)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

"""
data_generator = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.1,
    zoom_range=0.3,
)
generator = data_generator.flow_from_directory(
    'training',
    target_size=(250, 250),
    batch_size=20,
    class_mode='binary'
)

data_generator_test = ImageDataGenerator(
    rescale=1.0/255.0,
)
generator_valid = data_generator_test.flow_from_directory(
    'validation',
    target_size=(250, 250),
    batch_size=10,
    class_mode='binary'
)

generator_test = data_generator_test.flow_from_directory(
    'testing',
    target_size=(250, 250),
    batch_size=10,
    class_mode='binary'
)


generator_out = data_generator_test.flow_from_directory(
    'Photos',
    target_size=(250, 250),
    batch_size=1,
    class_mode='binary'
)


"""
data = cv2.imread('Photos/cat1.jpg')
data = np.expand_dims(data, axis=0)
generator_out.fit(data)

figure = plt.figure(figsize=(10, 10))

for i in range(9):
    output = generator_out.flow(data)[0]
    img =  np.reshape(output, output.shape[1:4])
    plt.subplot(3, 3, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
"""






# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# es = EarlyStopping(monitor='val_loss', mode='min', patience=7)
# rlrop = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=0.001)
# mch = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)

# history = model.fit(generator, steps_per_epoch=len(generator), epochs=100,
#                     validation_data=generator_valid, validation_steps=len(generator_valid), callbacks=[es, rlrop, mch])

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# plt.plot(history.history['accuracy'], color='blue', label = 'Training Accuracy')
# plt.plot(history.history['val_accuracy'], color='green' , label = 'Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# best_val_accuracy = max(history.history['val_accuracy'])
# print("Best Validation Accuracy:", best_val_accuracy)

model = tf.keras.models.load_model('model.h5')
#model.evaluate(generator_test)
model.evaluate(generator_out)


