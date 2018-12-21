import pandas as pd
import sys 
import numpy as np
import math as m
import jieba
from gensim.models import word2vec
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Input, BatchNormalization, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import LeakyReLU


jieba.load_userdict(sys.argv[3])
table = pd.read_csv(sys.argv[1], sep='delimiter', encoding='utf8')
table = pd.DataFrame(table)
table = np.array(table)
table = table.reshape([len(table)])
ran = len(table)
for i in range(ran):
	table[i] = table[i].split(',', 1)[1]
	table[i] = jieba.cut(table[i])
	
	table[i] = list(table[i])
model =  word2vec.Word2Vec.load("word2vec_256.model")
temp = []
v = np.zeros(256)
for line in table:
	templ = []
	for word in line:
		if word in model:
			templ.append(model[word])
		
	templ = np.array(templ)	
	temp.append(templ)
train_x = pad_sequences(temp, maxlen=48, padding='post', truncating='post', value=model[' '])
tabley = pd.read_csv(sys.argv[2], encoding='utf8')
tabley = pd.DataFrame(tabley)
train_y = np.asfarray(tabley[tabley.columns[1]], int)
print(train_x.shape)
q_input = Input(shape=(48, 256))
inner = LSTM(256 , return_sequences = True, input_length=48, input_dim=256, dropout=0.5, recurrent_dropout=0.5, kernel_initializer='he_normal')(q_input)
inner = LSTM(256 , return_sequences = False, input_length=48, input_dim=256, dropout=0.5, recurrent_dropout=0.5, kernel_initializer='he_normal')(inner)
inner = Dense(512, activation='relu')(inner)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Dense(512, activation='relu')(inner)
inner = BatchNormalization()(inner)
inner = Dropout(0.5)(inner)
inner = Dense(output_dim=1)(inner)
y_pred = Activation('sigmoid')(inner)
model = Model(q_input, y_pred)
model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(optimizer=adam,
            loss='binary_crossentropy',
			metrics=['accuracy'])
model.fit(train_x,train_y,batch_size=64,epochs=100,validation_split=.2, verbose=1, callbacks=callbacks_list, shuffle=True)

