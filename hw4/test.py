import pandas as pd
import sys 
import numpy as np
import math as m
from keras.models import load_model
import jieba
from gensim.models import word2vec
import h5py
from keras.preprocessing.sequence import pad_sequences

jieba.load_userdict(sys.argv[2])

test_x = pd.read_csv(sys.argv[1], sep='delimiter', encoding='utf8')
test_x = pd.DataFrame(test_x)
test_x = np.array(test_x)
table = test_x.reshape([len(test_x)])
ran = len(table)
for i in range(ran):
	table[i] = table[i].split(',', 1)[1]
	table[i] = jieba.cut(table[i])

model =  word2vec.Word2Vec.load("word2vec_256.model")

temp = []
v = np.zeros(256)
for line in table:
	templ = []
	for word in line:
		if word in model:
			if word in model:
				templ.append(model[word])
		
	templ = np.array(templ)	
	temp.append(templ)

test_x = pad_sequences(temp, maxlen=48, dtype='int32', padding='post', truncating='post', value=model[' '])

model = load_model("model.h5")
y_predict = model.predict(test_x)

ran = len(y_predict)
for i in range(ran):
	if y_predict[i] > 0.5:
		y_predict[i] = 1
	else:
		y_predict[i] = 0
y_predict = y_predict.astype(int)

index = np.array([[str(i)] for i in range(ran)])
solution = np.hstack((index, y_predict))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'label']
solution.to_csv(sys.argv[3] , columns = ['id' , 'label'] , index = False , sep = ',')