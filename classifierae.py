# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 01:20:27 2019

@author: tanma
"""

import numpy as np
import keras
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = np.amax(y_train) + 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
filters = 16
latent_dim = 16

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    filters = filters * 2
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               padding='same')(x)
    x = MaxPooling2D()(x)

shape = x.shape.as_list()

x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

encoder = Model(inputs, latent, name='encoder')
encoder.summary()
plot_model(encoder, to_file='classifier-encoder.png', show_shapes=True)

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)


for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        padding='same')(x)
    x = UpSampling2D()(x)
    filters = int(filters / 2)

x = Conv2DTranspose(filters=1, kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)


decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='classifier-decoder.png', show_shapes=True)

latent_inputs = Input(shape=(latent_dim,), name='classifier_input')
x = Dense(512)(latent_inputs)
x = Activation('relu')(x)
x = Dropout(0.4)(x)
x = Dense(256)(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)
x = Dense(num_labels)(x)
classifier_outputs = Activation('softmax', name='classifier_output')(x)
classifier = Model(latent_inputs, classifier_outputs, name='classifier')
classifier.summary()
plot_model(classifier, to_file='classifier.png', show_shapes=True)

autoencoder = Model(inputs,
                    [classifier(encoder(inputs)), decoder(encoder(inputs))],
                    name='autodecoder')
autoencoder.summary()
plot_model(autoencoder, to_file='classifier-autoencoder.png', show_shapes=True)

autoencoder.compile(loss=['categorical_crossentropy', 'mse'],
                    optimizer='adam',
                    metrics=['accuracy', 'mse'])

autoencoder.fit(x_train, [y_train, x_train],
                validation_data=(x_test, [y_test, x_test]),
                epochs=2, batch_size=batch_size,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

y_predicted, x_decoded = autoencoder.predict(x_test)
print(np.argmax(y_predicted[:8], axis=1))

imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.show()