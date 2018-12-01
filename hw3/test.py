import pandas as pd
import sys 
import numpy as np
import math as m
from keras.models import load_model

import h5py

model = load_model("model.h5")


test_x_file = sys.argv[1]
test_x = pd.read_csv(test_x_file , encoding = 'big5')
test_x = pd.DataFrame(test_x)
test_x = np.array(test_x)
temppp = []
for i in test_x:
    temp = i[1].split(' ')
    temppp.append(temp)
x_test = np.array(temppp).reshape(7178 , 48 , 48 ,1).astype(float)
x_test = x_test/255

#model = load_model("best_virgin")
y_predict = model.predict(x_test)
y = y_predict.argmax(axis = -1).reshape(7178,1)




index = np.array([[str(i)] for i in range(7178)])
solution = np.hstack((index,y))
solution = pd.DataFrame(solution)
solution.columns = ['id' , 'label']
solution.to_csv(sys.argv[2] , columns = ['id' , 'label'] ,  index = False , sep = ',')
