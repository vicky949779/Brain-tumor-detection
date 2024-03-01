import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

image_directory='datasets/'
no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE = 64

# print(no_tumor_images)
# path='no0.jpg'
# print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)

y_train=to_categorical(y_train,num_classes=2)
y_test=to_categorical(y_test,num_classes=2)

#model training 

model=Sequential()

model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE,INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
hist =model.fit(x_train, y_train,
batch_size=16,verbose=1, epochs=10, validation_data=(x_test,y_test),
shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')

h=hist.history
h.keys()
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c="red")
plt.xlabel("accuracy")
plt.ylabel("val_accuracy")
plt.title("acc Vs Val_acc")
plt.show()

# plt.plot(h['loss'],label="loss")
# plt.plot(h['val_loss'] , c="green", label="val_loss")
# plt.xlabel("loss")
# plt.ylabel("val_loss")
# plt.title("Loss Vs Val_loss")
# plt.show()

# -----------------------------------old code--------------------------------
# ----------------------------------             --------------------------------
# import cv2
# import os
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.utils import normalize, to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dense, Flatten, Dropout
# import matplotlib.pyplot as plt

# image_directory = 'datasets/'
# no_tumor_images = os.listdir(image_directory + 'no/')
# yes_tumor_images = os.listdir(image_directory + 'yes/')
# dataset = []
# label = []

# INPUT_SIZE = 64

# label_mapping = {'no': 0, 'yes': 1}

# for i, image_name in enumerate(no_tumor_images):
#     if image_name.split('.')[1] == 'jpg':
#         image = cv2.imread(image_directory + 'no/' + image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE, INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append('no')

# for i, image_name in enumerate(yes_tumor_images):
#     if image_name.split('.')[1] == 'jpg':
#         image = cv2.imread(image_directory + 'yes/' + image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE, INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append('yes')

# dataset = normalize(np.array(dataset), axis=1)
# label = np.array([label_mapping[l] for l in label])

# x_train, x_temp, y_train, y_temp = train_test_split(dataset, label, test_size=0.2, random_state=0)
# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

# y_train = to_categorical(y_train, num_classes=2)
# y_val = to_categorical(y_val, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)

# model = Sequential()

# model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10,
#                  validation_data=(x_val, y_val), shuffle=False)

# model.save_weights('BrainTumor10EpochsCategorical_weights.h5')

# h = hist.history
# plt.plot(h['loss'], label="loss")
# plt.plot(h['val_loss'], c="green", label="val_loss")
# plt.xlabel("loss")
# plt.ylabel("val_loss")
# plt.title("Loss Vs Val_loss")
# plt.show()

