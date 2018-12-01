import pandas as pd
import sys 
import numpy as np
import math as m
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam , SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
import h5py

table = pd.read_csv(sys.argv[1], encoding='big5')
table = pd.DataFrame(table)

#print(table)
total = np.array(table)
train_x = []
for i in total:
	t = i[1].split(' ')
	train_x.append(t)

y = pd.Series(table[table.columns[0]])
y = pd.get_dummies(y)
train_y = np.asfarray(y, int)
train_x = np.asfarray(train_x, float)

train = np.concatenate((train_y, train_x), axis=1)
np.random.shuffle(train)

train_y = train[:, 0:7]
train_x = train[:, 7:]

train_x = train_x.reshape(28709, 48, 48, 1)
train_x = train_x / 255
#print(train_x)

valx = train_x[0:5500, :, :]
train_x = train_x[5500:, :, :]

valy = train_y[0:5500, :]
train_y = train_y[5500:, :]



datagen = ImageDataGenerator(rotation_range = 25,
                              width_shift_range = 0.15,
                              height_shift_range = 0.15,
                              shear_range = 0.15,
                              zoom_range = [0.8, 1.2],
                              horizontal_flip = True,
                              fill_mode = 'nearest')
datagen.fit(train_x)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()
#adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1E-8, decay=0.0000001, amsgrad=False)
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
callbacks_list = [checkpoint]
#checkpoint = ModelCheckpoint(filepath="best_virgin", verbose=1, save_best_only=True)
training = model.fit_generator(datagen.flow(train_x , train_y , batch_size = 128) ,steps_per_epoch=10*train_x.shape[0]//128, epochs = 300, callbacks=callbacks_list, verbose = 1 ,
validation_data = (valx, valy))



