import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import cv2,os
#from load_imgdata import load_data
import sklearn

IMG_SHAPE = 256
nClass = 4
BATCH_SIZE  = 8;
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

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
    Dense(nClass,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train,y_train , batch_size=8, epochs=5)
model.save('my_model_actor.h5')
prediction  = model.predict_classes(x_test)
h = np.argmax(prediction,axis=1)
j = np.argmax(y_test,axis=1)
ac=np.sum(h==j)/len(x_test)
print(ac)




# =============================================================================
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# 
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# 
# epochs_range = range(epochs)
# 
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# 
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
# 
# from keras.models import load_model
# model.save('my_model_actor.h5')
# =============================================================================






    
