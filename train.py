import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import cv2,os

num_of_students_in_class = 4;
total_train = num_of_students_in_class*95;
total_val = num_of_students_in_class*25;
IMG_SHAPE = 400;
BATCH_SIZE  = 8;
image_gen_train = ImageDataGenerator(rescale=1./255,
      #rotation_range=40,
      #width_shift_range=0.1,
      #height_shift_range=0.1,
      shear_range=0.15,
      zoom_range=0.15)
      #horizontal_flip=True,
      #fill_mode='nearest')
      
train_data_gen = image_gen_train.flow_from_directory(directory='/home/yani/Documents/projects/Face-Recognition_based-Attendence-System/datagen/train',
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True, 
                                                     color_mode='grayscale',
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='categorical',
                                                     seed =42)

image_gen_val = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.15,
                                   zoom_range = 0.15)

val_data_gen = image_gen_val.flow_from_directory(directory='/home/yani/Documents/projects/Face-Recognition_based-Attendence-System/datagen/validate',            
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 color_mode = 'grayscale',
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='categorical',
                                                 seed = 42)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE,1)),
    Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.2),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(rate=0.2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(rate=0.2),
    Dense(num_of_students_in_class,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs=5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()