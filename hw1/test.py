import pandas as pd
import sys 
import numpy as np
fill = sys.argv[1]
df = pd.read_csv(fill, encoding='big5', header = None)
table = pd.DataFrame(df)
table2 = pd.DataFrame(df)
table3 = pd.DataFrame(df)
table4 = pd.DataFrame(df)
table5 = pd.DataFrame(df)
table5 = table5.replace("NR", "0")
table = table.drop([i for i in table.index if (i-8)%18 != 0])
table = table.drop(columns = table.columns[0:2])
table.index = (table.index - 8) // 18


table2 = table2.drop([i for i in table2.index if (i-9)%18 != 0])
table2 = table2.drop(columns = table2.columns[0:2])
table2.index = (table2.index - 9) // 18


table3 = table3.drop([i for i in table3.index if (i-2)%18 != 0])
table3 = table3.drop(columns = table3.columns[0:2])
table3.index = (table3.index - 2) // 18

table4 = table4.drop([i for i in table4.index if (i-12)%18 != 0])
table4 = table4.drop(columns = table4.columns[0:2])
table4.index = (table4.index - 12) // 18

table5 = table5.drop([i for i in table5.index if (i-10)%18 != 0])
table5 = table5.drop(columns = table5.columns[0:2])
table5.index = (table5.index - 10) // 18

pm10 = np.array(table).reshape([260, 9])
pm25 = np.array(table2).reshape([260, 9])
co = np.array(table3).reshape([260, 9])
so2 = np.array(table4).reshape([260, 9])
rf = np.array(table5).reshape([260, 9])
pm10 = np.asfarray(pm10, float)
pm25 = np.asfarray(pm25, float)
co = np.asfarray(co, float)
so2 = np.asfarray(so2, float)
rf = np.asfarray(rf, float)
#traing6 = np.asfarray(traing6, float)
for i in range(260):
    for j in range(9):
        if pm10[i][j] == 0:
            if j == 0:
                pm10[i][j] = (pm10[i-1][8]+pm10[i-1][7])/2
            elif j == 1:
                pm10[i][j] = (pm10[i][j-1]+pm10[i-1][8])/2
            else:
                pm10[i][j] = (pm10[i][j-2]+pm10[i][j-1])/2


#nnnx 2.5
normal = np.load(sys.argv[4])

max10 = normal[0]
min10 = normal[1]
max25 = normal[2]
min25 = normal[3]
maxco = normal[4]
minco = normal[5]

pm10_2 = np.power(pm10, 2)
pm25_2 = np.power(pm25, 2)

so2 = (so2 - normal[11])/(normal[10] - normal[11])
pm25_2 = (pm25_2 - normal[7])/(normal[6] - normal[7])
pm10_2 = (pm10_2 - normal[9])/(normal[8] - normal[9])
pm10 = (pm10 - min10)/(max10 - min10)
pm25 = (pm25 - min25)/(max25 - min25)
co = (co - minco)/(maxco - minco)
#rf = (rf - normal[13])/(normal[12] - normal[13])




#print(normal)
#traing = np.power(x, 1)

#traing3 = np.power(nnx, 1)
#traing6 = traing5

#x = (x - np.mean(x))/np.std(x)
#nx = (nx - np.mean(nx))/np.std(nx)
#nnnx = (nnnx - np.mean(nnnx))/np.std(nnnx)
#nnx = (nnx - np.mean(nnx))/np.std(nnx)
bb = np.ones([260, 1])
x = np.concatenate((bb, pm25_2, pm25, pm10, co,  so2, rf), axis = 1)

#x = np.concatenate((bb, x), axis = 1)
#print(x)
#print(x.shape)
ww = np.load(sys.argv[2])
total = x.dot(ww)
#np.save(sys.argv[4], total)
#print(ww)
#print(total)
total = total.reshape(260, 1)
index = np.array([["id_"+str(i)] for i in range(260)])
sol = np.concatenate((index, total), axis = 1)
#print(sol)
sol = pd.DataFrame(sol)
sol.columns = ['id', 'value']
sol.to_csv(sys.argv[3], columns = ['id', 'value'], index = False, sep = ',')
