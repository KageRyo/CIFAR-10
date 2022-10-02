import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Activation, Dense,MaxPooling2D,Flatten,Dropout,MaxPool2D,GlobalAveragePooling2D,Lambda, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization

#!nvidia-smi

print(tf.__version__)
print(keras.__version__)

train_datagen = ImageDataGenerator(rescale=1/255)#影像正規畫
test_datagen = ImageDataGenerator(rescale=1/255)
train_data = train_datagen.flow_from_directory(
                "../input/cifar10-pngs-in-folders/cifar10/train", # 目標目錄
                target_size=(32,32), # 所有影象調整為32x32
                batch_size=20,
                seed=10)
validation_data = test_datagen.flow_from_directory(
                "../input/cifar10-pngs-in-folders/cifar10/test", # 目標目錄
                target_size=(32,32), # 所有影象調整為32x32
                batch_size=20,
                seed=10)

train_data.class_indices #Label對應表

class_names = list(train_data.class_indices.keys())
print(class_names)

train_imgs, labels = next(train_data)
print(train_imgs.shape) #顯示資料維度

plt.imshow(train_imgs[10])
plt.xlabel(class_names[np.argmax(labels[10])])#顯示 Label與影像資料

plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(labels[i])])
plt.show()

def frac_max_pool(x):
    return tf.nn.fractional_max_pool(x, [1.0, 1.41, 1.41, 1.0], pseudo_random=True, overlapping=True)[0]

#模型
LeNet = Sequential()

LeNet.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.3))

LeNet.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.35))

LeNet.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.35))

LeNet.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.4))

LeNet.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.45))

LeNet.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(Lambda(frac_max_pool))
LeNet.add(Dropout(0.5))

LeNet.add(Conv2D(filters=192, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform'))
LeNet.add(LeakyReLU())
LeNet.add(BatchNormalization())
LeNet.add(GlobalAveragePooling2D())
LeNet.add(Dense(units=10, kernel_initializer='he_uniform', activation='softmax'))
LeNet.summary()

plot_model(LeNet, to_file='LeNet.png', show_shapes=True, show_layer_names=True)

LeNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = LeNet.fit_generator(train_data,steps_per_epoch=50000/20,epochs=150,validation_data=validation_data, validation_steps = 10000//20)#epochs=150

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

test_imgs, test_labels = next(validation_data)
predictions = LeNet.predict(test_imgs)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    true_label1 = np.argmax(true_label)
    if predicted_label == true_label1:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label1]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    true_label1 = np.argmax(true_label)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label1].set_color('green')

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_imgs)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

    LeNet.save('my_LeNet.h5')