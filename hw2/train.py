import pandas as pd
import sys 
import numpy as np
import math as m

df = pd.read_csv(sys.argv[1], encoding='utf-8')
table = pd.DataFrame(df)
table = np.array(table)
table = np.asfarray(table, int)

df = pd.read_csv(sys.argv[2], encoding='utf-8')
table2 = pd.DataFrame(df)
table2 = np.array(table2)
table2 = np.asfarray(table2, int)

a = []
b = []
for i in range(20000):
	if table2[i][0] == 1:
		a.append(table[i])
	else:
		b.append(table[i])

a = np.asfarray(a, int)
b = np.asfarray(b, int)

rana = len(a)
ranb = len(b)


a = a.transpose()
u1 = np.ones([23, 1])
for i in range(23):
	u1[i] = np.sum(a[i])/rana

a = a.transpose()

sigma1 = np.zeros([23, 23])


#print((a[i] - u1.reshape([1, 23])).transpose())
for i in range(rana):
	sigma1 += np.dot(((a[i] - u1.reshape([1, 23])).transpose()), (a[i] - u1.reshape([1, 23])))

sigma1 /= rana


b = b.transpose()
u2 = np.ones([23, 1])
for i in range(23):
	u2[i] = np.sum(b[i])/ranb

b = b.transpose()
sigma2 = np.zeros([23, 23])
for i in range(ranb):
	sigma2 += np.dot((b[i] - u2.reshape([1, 23])).transpose(), (b[i] - u2.reshape([1, 23])))
sigma2 = sigma2/ranb

df = pd.read_csv(sys.argv[3], encoding='utf-8')
table3 = pd.DataFrame(df)
table3 = np.array(table3)
table3 = np.asfarray(table3, int)

sigma1 = (sigma1*rana + sigma2*ranb)/(rana+ranb)
#print(sigma1)

k = np.dot((u1 - u2).transpose(), np.linalg.inv(sigma1))

z = np.dot(table3, k.T) - 0.5*np.dot(np.dot(u1.transpose(), np.linalg.inv(sigma1)), u1) +  0.5*np.dot(np.dot(u2.transpose(), np.linalg.inv(sigma1)), u2) + np.log(rana/ranb)

yy = 1/(1 + np.exp(-z))


for i in range(10000):
	if yy[i] < 0.5:
		yy[i] = 0
	else:
		yy[i] = 1
yy = yy.astype(int)

yy = yy.reshape(10000, 1)
index = np.array([["id_"+str(i)] for i in range(10000)])
sol = np.concatenate((index, yy), axis = 1)
#print(sol)
sol = pd.DataFrame(sol)
sol.columns = ['id', 'Value']
sol.to_csv(sys.argv[4], columns = ['id', 'Value'], index = False, sep = ',')

