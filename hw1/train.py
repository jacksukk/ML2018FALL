import pandas as pd
import sys 
import numpy as np
import math as m
fill = sys.argv[1]
df = pd.read_csv(fill, encoding='big5')
table = pd.DataFrame(df)
table2 = pd.DataFrame(df)
table3 = pd.DataFrame(df)
table4 = pd.DataFrame(df)
table5 = pd.DataFrame(df)
table5 = table5.replace("NR", "0")

table = table.drop([i for i in table.index if (i-8)%18 != 0])
table = table.drop(columns = table.columns[0:3])
table.index = (table.index - 8) // 18

table2 = table2.drop([i for i in table2.index if (i-9)%18 != 0])
table2 = table2.drop(columns = table2.columns[0:3])
table2.index = (table2.index - 9) // 18


table3 = table3.drop([i for i in table3.index if (i-2)%18 != 0])
table3 = table3.drop(columns = table3.columns[0:3])
table3.index = (table3.index - 12) // 18

table4 = table4.drop([i for i in table4.index if (i-12)%18 != 0])
table4 = table4.drop(columns = table4.columns[0:3])
table4.index = (table4.index - 12) // 18

table5 = table5.drop([i for i in table5.index if (i-10)%18 != 0])
table5 = table5.drop(columns = table5.columns[0:3])
table5.index = (table5.index - 10) // 18
#print(table5)

np1 = np.array(table)
np1 = np1.reshape(5760)
k = np.array([np1[i:i+10] for i in range(471)])
for j in range(2, 13):
    temp = np.array([np1[i:i+10] for i in range((j-1)*480, j*480-9)])
    k = np.concatenate((k, temp), axis = 0)
np1 = np.asfarray(k, float)

np2 = np.array(table2)
np2 = np2.reshape(5760)
t = np.array([np2[i:i+10] for i in range(471)])
for j in range(2, 13):
    temp = np.array([np2[i:i+10] for i in range((j-1)*480, j*480-9)])
    t = np.concatenate((t, temp), axis = 0)
np2 = np.asfarray(t, float)

np3 = np.array(table3)
np3 = np3.reshape(5760)
t = np.array([np3[i:i+10] for i in range(471)])
for j in range(2, 13):
    temp = np.array([np3[i:i+10] for i in range((j-1)*480, j*480-9)])
    t = np.concatenate((t, temp), axis = 0)
np3 = np.asfarray(t, float)

np4 = np.array(table4)
np4 = np4.reshape(5760)
t = np.array([np4[i:i+10] for i in range(471)])
for j in range(2, 13):
    temp = np.array([np4[i:i+10] for i in range((j-1)*480, j*480-9)])
    t = np.concatenate((t, temp), axis = 0)
np4 = np.asfarray(t, float)

np5 = np.array(table5)
np5 = np5.reshape(5760)
t = np.array([np5[i:i+10] for i in range(471)])
for j in range(2, 13):
    temp = np.array([np5[i:i+10] for i in range((j-1)*480, j*480-9)])
    t = np.concatenate((t, temp), axis = 0)
np5 = np.asfarray(t, float)


print(2.5*np.std(np2)+np.mean(np2))
print(2.5*np.std(np1) + np.mean(np1))
print(2.5*np.std(np3) + np.mean(np3))
print(2.5*np.std(np4) + np.mean(np4))
mustdelete = []
for i in range(5652):
    if np.max(np2[i]) >= 150 or np.min(np2[i]) < 0:
        mustdelete.append(i)
        continue
    
    for j in range(10):
        if np1[i][j] == 0 and np2[i][j] == 0 and np3[i][j] == 0 and np4[i][j] == 0:
            mustdelete.append(i)
            break
    for j in range(10):
        if np3[i][j] < 0 or np4[i][j] < 0:
            mustdelete.append(i)
            break
#print(np2)
np2 = np.delete(np2, mustdelete, axis=0)
np1 = np.delete(np1, mustdelete, axis=0)
np3 = np.delete(np3, mustdelete, axis=0)
np4 = np.delete(np4, mustdelete, axis=0)
np5 = np.delete(np5, mustdelete, axis=0)
print(np2.shape)
for i in range(5286):
    for j in range(10):
        if np1[i][j] == 0:
            if j == 0:
                np1[i][j] = (np1[i][1]+np1[i][2])/2
            elif j == 1:
                np1[i][j] = (np1[i][0]+np1[i][2])/2
            else:
                np1[i][j] = (np1[i][j-1]+np1[i][j-2])/2
"""for i in range(5286):
    for j in range(10):
        if np2[i][j] == 0 and np3[i][j] == 0 and np4[i][j] == 0:
            if j == 0:
                np2[i][j] = (np2[i-1][9]+np2[i-1][8])/2
                np3[i][j] = (np3[i-1][9]+np3[i-1][8])/2
                np4[i][j] = (np4[i-1][9]+np4[i-1][8])/2
            elif j == 1:
                np2[i][j] = (np2[i][j-1]+np2[i-1][9])/2
                np3[i][j] = (np3[i][j-1]+np3[i-1][9])/2
                np4[i][j] = (np4[i][j-1]+np4[i-1][9])/2
            else:
                np2[i][j] = (np2[i][j-1]+np2[i][j-2])/2
                np3[i][j] = (np3[i][j-1]+np3[i][j-2])/2
                np4[i][j] = (np4[i][j-1]+np4[i][j-2])/2 """



weight = np.concatenate((np.ones(1)/9, np.random.rand(54)), axis = 0)

pm10 = np1[:, 0:9]
pm25 = np2[:, 0:9]
co = np3[:, 0:9]
so2 = np4[:, 0:9]
rf = np5[:, 0:9]
pm25_2 = np.power(pm25, 2)
pm10_2 = np.power(pm10, 2)
print(pm25_2)
pm25_3 = np.power(pm25, 3)
pm25_4 = np.power(pm25, 4)
max10 = np.amax(pm10)
min10 = np.amin(pm10)
max25 = np.amax(pm25)
min25 = np.amin(pm25)
maxco = np.amax(co)
minco = np.amin(co)
max25_2 = np.amax(pm25_2)
min25_2 = np.amin(pm25_2)
max10_2 = np.amax(pm10_2)
min10_2 = np.amin(pm10_2)
maxso2 = np.amax(so2)
minso2 = np.amin(so2)
max25_3 = np.amax(pm25_3)
min25_3 = np.amin(pm25_3)
max25_4 = np.amax(pm25_4)
min25_4 = np.amin(pm25_4)
maxrf = np.amax(rf)
minrf = np.amin(rf)
normal = np.array([max10, min10, max25, min25, maxco, minco, max25_2, min25_2, max10_2, min10_2, maxso2, minso2, maxrf, minrf])
np.save(sys.argv[3], normal)
"""
pm10 = (pm10 - min10)/(max10 - min10)
pm25 = (pm25 - min25)/(max25 - min25)
co = (co - minco)/(maxco - minco)

pm25_2 = (pm25_2 - min25_2)/(max25_2 - min25_2)
pm10_2 = (pm10_2 - min10_2)/(max10_2 - min10_2)
so2 = (so2 - minso2)/(maxso2 - minso2)
pm25_4 = (pm25_4 - min25_4)/(max25_4-min25_4)
pm25_3 = (pm25_3-min25_3)/(max25_3 - min25_3)
#rf = (rf - minrf)/(maxrf - minrf)"""
temp = np.ones([5286, 1])

traingdata = np.concatenate((temp, pm25_2, pm25, pm10, co, so2, rf), axis = 1)
ydata = np2[:, 9]
y = ydata.reshape(5286, 1)

val = np.concatenate((y, traingdata), axis=1)

np.random.shuffle(val)

yval = val[0:1100, 0:1]
xval = val[0:1100, 1:]
traingdata = val[1100:5286, 1:]
ydata = val[1100:5286, 0:1]
yval = yval.reshape(1100)
ydata = ydata.reshape(4186)
loss = 1E20
gradientsum = 100
itera = 1E6
count = 1
ada = 0
llast = 1E20
while loss - llast <= 0:
    itera -= 1
    llast = loss
    gradient = 2*np.dot(np.dot(traingdata.transpose(), traingdata), weight) - 2*np.dot(ydata.transpose(), traingdata) - 2*0.01*weight
    gradientsum = np.dot(gradient, gradient.transpose())
    ada += gradientsum
    #print("fuck")
    lr = 10000/m.sqrt(ada)
    weight = np.subtract(weight, lr*gradient)
    ll = np.dot(weight, xval.transpose()) - yval
    loss = np.dot(ll, ll.transpose())
    #print(loss, llast)
    count += 1
    print(m.sqrt(loss/1100), gradientsum, itera)
print(weight)
ffname = sys.argv[2]
np.save(ffname, weight)
