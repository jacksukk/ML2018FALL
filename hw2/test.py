import pandas as pd
import sys 
import numpy as np

df = pd.read_csv(sys.argv[1], encoding='utf-8')
table = pd.DataFrame(df)


#############################
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
#print(pay0.shape)
pay2 = pd.Series(table[table.columns[6]])
pay2 = pd.get_dummies(pay2)
pay2 = np.asfarray(pay2, int)
#print(pay2.shape)
pay3 = pd.Series(table[table.columns[7]])
pay3 = pd.get_dummies(pay3)
pay3 = np.asfarray(pay3, int)
pay3 = np.concatenate((pay3, np.zeros([10000,1])), axis=1)
#print(pay3.shape)
pay4 = pd.Series(table[table.columns[8]])
pay4 = pd.get_dummies(pay4)
pay4 = np.asfarray(pay4, int)
#print(pay4.shape)
pay5 = pd.Series(table[table.columns[9]])
pay5 = pd.get_dummies(pay5)
pay5 = np.asfarray(pay5, int)
pay5 = np.concatenate((pay5, np.zeros([10000,1])), axis=1)

pay6 = pd.Series(table[table.columns[10]])
pay6 = pd.get_dummies(pay6)
pay6 = np.asfarray(pay6, int)
pay6 = np.concatenate((pay6, np.zeros([10000,1])), axis=1)

#############################
table = table.drop([table.columns[2], table.columns[5], table.columns[6], table.columns[7], table.columns[8], table.columns[9], table.columns[10]], axis=1)
table = np.array(table)
table = np.asfarray(table, int)

table = np.concatenate((table, edu, pay0, pay2), axis=1)

table = table.transpose()
ran = len(table)
mm = np.load(sys.argv[2])

for i in range(ran):
	if mm[i] - mm[i+ran] != 0:
		table[i] = (table[i] - mm[i+ran])/(mm[i] - mm[i+ran])

table = table.transpose()

xx = np.concatenate((np.ones([10000, 1]), table), axis=1)
ww = np.load(sys.argv[3])
#print(xx)
yy = np.dot(xx, ww)

yy = 1/(1 + np.exp(-yy))


for i in range(10000):
	if yy[i] <= 0.46:
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
