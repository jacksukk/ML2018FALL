import pandas as pd
import sys 
import numpy as np
import math as m

df = pd.read_csv(sys.argv[1], encoding='utf-8')
table = pd.DataFrame(df)
###########################onehot########################
sex = pd.Series(table[table.columns[1]])
sex = pd.get_dummies(sex)
sex = np.asfarray(sex, int)

edu = pd.Series(table[table.columns[2]])
edu = pd.get_dummies(edu)
edu = np.asfarray(edu, int)

mar = pd.Series(table[table.columns[3]])
mar = pd.get_dummies(mar)
mar = np.asfarray(mar, int)

pay0 = pd.Series(table[table.columns[5]])
pay0 = pd.get_dummies(pay0)
pay0 = np.asfarray(pay0, int)
print(pay0.shape)
pay2 = pd.Series(table[table.columns[6]])
pay2 = pd.get_dummies(pay2)
pay2 = np.asfarray(pay2, int)
pay2 = np.concatenate((pay2, np.zeros([20000,1])), axis=1)
print(pay2.shape)
pay3 = pd.Series(table[table.columns[7]])
pay3 = pd.get_dummies(pay3)
pay3 = np.asfarray(pay3, int)
print(pay3.shape)
pay4 = pd.Series(table[table.columns[8]])
pay4 = pd.get_dummies(pay4)
pay4 = np.asfarray(pay4, int)
print(pay4.shape)
pay5 = pd.Series(table[table.columns[9]])
pay5 = pd.get_dummies(pay5)
pay5 = np.asfarray(pay5, int)
print(pay5.shape)
pay6 = pd.Series(table[table.columns[10]])
pay6 = pd.get_dummies(pay6)
pay6 = np.asfarray(pay6, int)
print(pay6.shape)
######################################################
table = table.drop([table.columns[2], table.columns[5], table.columns[6], table.columns[7], table.columns[8], table.columns[9], table.columns[10]], axis=1)

train = np.array(table)
train_x = np.asfarray(train, int)

train_x = np.concatenate((train_x, edu, pay0, pay2),axis=1)

df = pd.read_csv(sys.argv[2], encoding='utf-8')
table2 = pd.DataFrame(df)
train_y = np.array(table2)
train_y = np.asfarray(train_y, int)

maxstore = []
minstore = []
train_x = train_x.transpose()
ran = len(train_x)

for i in range(ran):
    maxstore.append(np.amax(train_x[i]))
    minstore.append(np.amin(train_x[i]))
    if maxstore[i] - minstore[i] != 0:
        train_x[i] = (train_x[i] - minstore[i])/(maxstore[i] - minstore[i])
maxstore.extend(minstore)
maxmin = np.array(maxstore)
np.save(sys.argv[3], maxmin)

weight = np.concatenate((np.ones(1)/9, np.random.rand(ran)), axis = 0)
print(weight.shape)
train_x = train_x.transpose()

temp = np.ones([20000, 1])

traingdata = np.concatenate((temp, train_x), axis=1)
#print(traingdata)
val = np.concatenate((train_y, traingdata), axis=1)
np.random.shuffle(val)

yval = val[0:4000, 0:1]
xval = val[0:4000, 1:]
#print(xval)
traingdata = val[4000:, 1:]

ydata = val[4000:, 0:1]
yval = yval.reshape(4000)
ydata = ydata.reshape(16000)

loss = 1E20
gradientsum = 100
itera = 1E6
count = 1
ada = 0
llast = 1E20
lr = 0.000012
while loss - llast <= 0:
    itera -= 1
    llast = loss

    ydata = ydata.reshape(16000, 1)
    temp = np.concatenate((ydata, traingdata), axis=1)
    np.random.shuffle(temp)
    ydata = temp[:, 0:1]
    ydata = ydata.reshape(16000)
    traingdata = temp[:, 1:]
    for i in range(250):
        sigmoid = 1/(1 + np.exp(-np.dot(traingdata[i*64:(i+1)*64,:], weight)))
        gradient = -np.dot(traingdata[i*64:(i+1)*64,:].transpose(), ydata[i*64:(i+1)*64] - sigmoid)
        weight = np.subtract(weight, lr*gradient)


    sigmoid2 = 1/(1 + np.exp(-np.dot(xval, weight)))
    loss = -(np.dot(yval.transpose(), np.log(sigmoid2)) + np.dot((1 - yval).transpose(), np.log(1 - sigmoid2)))
    count += 1
    lr = lr*0.999999
    print(loss, gradientsum, itera)
print(weight)
ffname = sys.argv[4]
np.save(ffname, weight)
