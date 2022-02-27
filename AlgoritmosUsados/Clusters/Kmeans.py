import matplotlib.pyplot as plt
import numpy as np 
import timeit
import Aneis
import time

from seaborn import scatterplot as scatter 
from sklearn import datasets as Datas
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics as sm
from jqmcvi import dunn_fast
from jqmcvi import dunn
from s_dbw import S_Dbw
from S_Dbw import DBCV
#--------------------------------------------------------------------------------------------------------

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#BANCO DE DADOS------------------------------------------------------------------------------------------

#Teste com Iris DataSet:
data,target = Datas.load_iris(return_X_y=True)

#Teste com Boston:
#data,target = Datas.load_boston(return_X_y=True)

#Teste com Wine:
#data, target =  Datas.load_wine(return_X_y=True)

#Teste com Circles01:
#data = Aneis.Circles01()

#Teste com Circles02:
#data = Aneis.Circles02()

#Teste com Circles03:
#data = Aneis.Circles03()

#Teste com Circles04:
#data = Aneis.Circles04()

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTES NÃO SUPERVISIONADOS------------------------------------------------------------------------------

X = np.array(1)

#calculo das metricas para cada K cluster:
Y=[0]
Z=[0]
K=[0]
D=[1]
U=[0]
S=[-1]
for i in range(2,10):
    clusters = KMeans(n_clusters=i)
    clusters.fit(data)
    X = np.append(X, i)
    Y.append(clusters.inertia_)
    Z.append(sm.silhouette_score(X=data, labels=clusters.labels_))
    K.append(sm.calinski_harabasz_score(X=data, labels=clusters.labels_))
    D.append(sm.davies_bouldin_score(X=data, labels=clusters.labels_))
    U.append(dunn_fast(points=data, labels=clusters.labels_))
    S.append(DBCV(X=data, labels=clusters.labels_))

    plt.subplot(231)
    plt.plot(i, clusters.inertia_, color="b", marker="s",markersize = 3) #(Joelho)
    plt.subplot(232)
    plt.plot(i, sm.silhouette_score(X=data, labels=clusters.labels_), color="g", marker="s",markersize = 3) #(MAX)
    plt.subplot(233)
    plt.plot(i, sm.calinski_harabasz_score(X=data, labels=clusters.labels_), color="r", marker="s", markersize = 3) #(MAX)
    plt.subplot(234)
    plt.plot(i, sm.davies_bouldin_score(X=data, labels=clusters.labels_), color="black", marker="s", markersize = 3) #(MIN)
    plt.subplot(235)
    plt.plot(i, dunn_fast(points=data, labels=clusters.labels_), color="pink", marker="s",markersize = 3) #(MAX)
    plt.subplot(236)
    plt.plot(i, DBCV(X=data, labels=clusters.labels_), color="orange", marker="s",markersize = 3) #(MAX)

#plot do gráfico de linhas:
plt.subplot(231)
plt.plot(X,Y, color="b")
plt.subplot(232)
plt.plot(X,Z, color="g")
plt.subplot(233)
plt.plot(X, K, color="r")
plt.subplot(234)
plt.plot(X, D, color="black")
plt.subplot(235)
plt.plot(X, U, color="pink")
plt.subplot(236)
plt.plot(X, S, color="orange")

#Marcar os melhores resultados:
plt.subplot(232)
c = X[np.argmax(Z)]
plt.plot([c, c], [np.min(Z), np.max(Z)], color="orange")
plt.subplot(233)
c = X[np.argmax(K)]
plt.plot([c, c], [np.min(K), np.max(K)], color="orange")
plt.subplot(234)
c = X[np.argmin(D)]
plt.plot([c, c], [np.min(D), np.max(D)], color="orange")
plt.subplot(235)
c = X[np.argmax(U)]
plt.plot([c, c], [np.min(U), np.max(U)], color="orange")
plt.subplot(236)
c = X[np.argmax(S)]
plt.plot([c, c], [np.min(S), np.max(S)], color="orange")

plt.show()

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTE SUPERVISIONADO------------------------------------------------------------------------------------

clusters = KMeans(n_clusters=i)
clusters.fit(data)
labels = clusters.labels_
print("Acuracia = {0}".format(sm.accuracy_score(target, labels))) #Não sei se funciona corretamente ainda
print("Rand = {0}".format(sm.adjusted_rand_score(target, labels))) #comparação de concordâncias e discordâncias. (max)
print("Completeness = {0}".format(sm.completeness_score(target, labels))) #computa de acordo com se os pontos de uma mesma label continuam em um mesmo grupo. (max)
print("Homogeneity = {0}".format(sm.homogeneity_score(target, labels))) #computa de acordo com a diversidade de labels dentro de um mesmo cluster. (max)
print("FolkesMallows = {0}".format(sm.fowlkes_mallows_score(target, labels))) #computa de acordo com o formato geomértico dos clusters(pares de pontos presentes no mesmo cluster ou em diferentes).
print("Mutual Info. = {0}".format(sm.normalized_mutual_info_score(target,labels,"geometric"))) #Medida estatistica usada para avaliar o co-ocorrência de dados em varios lugares(no caso clusters)
print("V score = {0}".format(sm.v_measure_score(target, labels)))#média harmônica entre a homogeneidade e completância.

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#color = ['#00091a', '#001b4d', '#002d80', '#003eb3', '#0050e6', '#1a6aff', '#4d8bff', '#80acff', '#b3cdff', '#e6eeff'] #AZUL
#color = ['#003325', '#ffcccc', '#009970', '#ff6666', '#00ffbf', '#ff0000', '#66ffd9', '#990000', '#ccfff2', '#330000'] #GREEN RED
color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
saidas= ["Iris", "Wine", "Boston"]

# file = open("Costumer.txt", 'r')
# if(file.mode == 'r'):
#       X = 0
#       for line in file:
#             if(X == 0):
#                   info =  line.split()
#                   for i in info:
#                         info[info.index(i)] = float(i)
#                   data = np.array([info])
#                   X += 1
#             else:
#                   info =  line.split()
#                   if(len(info) == 4):
#                         for i in info:
#                               info[info.index(i)] = float(i)
#                         data = np.concatenate([data, [info]])

def CVSopen(filePath):
      file = open(filePath, 'r')
      if(file.mode == 'r'):
            X = 0
            for line in file:
                  if(X == 0):
                        info =  line.split()
                        for i in info:
                              info[info.index(i)] = float(i)
                        data2 = np.array([info])
                        X += 1
                  else:
                        info =  line.split()
                        for i in info:
                              info[info.index(i)] = float(i)
                        data2 = np.concatenate([data2, [info]])
      file.close()
      return data2

data = CVSopen("insurence.txt")

#APLICAÇÃO DO ALGORITMO----------------------------------------------------------------------------------

#Algoritmo de Agrupamento:
inicio = timeit.time.perf_counter()
KMcluster = KMeans(n_clusters=2) #2 2 3
KMcluster.fit(data)
fim = timeit.time.perf_counter()
labels = KMcluster.labels_
#print("Duração: {0} segundos".format(fim - inicio))

colors = []
for i in range(len(data)):
    colors.append(color[int(labels[i])])

#Gráfico:
'''
plt.scatter(data[:,6], data[:,5], marker="o", c=colors)
plt.title("{0} Tempo = {1} segundos".format(saidas[2], round((fim - inicio),5)))
plt.show()  
'''

for i in range(7):
  for j in range(i+1,7):
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    plt.title("{0} Tempo = {1} segundos".format(saidas[0], round((fim - inicio),5)))
    axes[0].scatter(data[:,i], data[:,j], marker="o", c="#0099cc",)
    axes[1].scatter(data[:,1], data[:,0], marker="o", c=colors) 
    plt.show()
    plt.clf()

print("{0}".format(round(sm.silhouette_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.calinski_harabasz_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.davies_bouldin_score(X=data, labels=labels), 5)))
print("{0}".format(round(dunn_fast(points=data, labels=labels), 5)))
#--------------------------------------------------------------------------------------------------------

