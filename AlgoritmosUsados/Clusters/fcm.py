from __future__ import division, print_function

import matplotlib.colors
import matplotlib.pyplot as plt
import FuzzyIndex as fi
import numpy as np
import timeit
import Aneis
import time

from seaborn import scatterplot as scatter
from skfuzzy.cluster import cmeans as fcm
from scipy.spatial.distance import cdist
from sklearn import datasets as Datas
from sklearn import metrics as sm
from tabulate import tabulate  
from jqmcvi import dunn_fast
from jqmcvi import dunn

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#BANCO DE DADOS------------------------------------------------------------------------------------------

#teste com Iris DataSet:
data,target = Datas.load_iris(return_X_y=True)
S = []
for i in range(4):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Boston: 
data,target = Datas.load_boston(return_X_y=True)
S = []
for i in range(13):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Wine:
data, target =  Datas.load_wine(return_X_y=True)
S = []
for i in range(13):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Circles01:
data = Aneis.Circles01()
S = []
for i in range(2):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Circles02:
data = Aneis.Circles02()
S = []
for i in range(2):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Circles03:
data = Aneis.Circles03()
S = []
for i in range(2):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#Teste com Circles04:
data = Aneis.Circles04()
S = []
for i in range(2):
    S.append(data[:,i])
alldata = np.vstack((S[:]))

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTE PARA O MELHOR FPC---------------------------------------------------------------------------------

for i in range(2, 10):
    cntr, u, u0, d, jm, p, fpc = fcm(data=alldata, c=i, m=2, error=0.01, maxiter=1000)
    print("{0} - fpc: {1}".format(i,fpc))

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTES NÃO SUPERVISIONADOS------------------------------------------------------------------------------

#calculo das metricas para cada K cluster:
X = np.array(1)
Y=[0]
Z=[1]
W=[1]
A=[1]
H=[1]
B=[0]
for i in range(2,10):
    cntr, u, u0, d, jm, p, fpc = fcm(data=alldata, c=i, m=2, error=0.01, maxiter=1000)
    labels = np.argmax(u, axis=0)
    X = np.append(X, i)
    y = fi.pc(None, u, None, None) #(MAX)
    z = fi.fhv(data, u, cntr, 2) #(MIN)
    w = fi.fs(data, u, cntr, 2) #(MIN) 
    a = fi.xb(data, u, cntr, 2) #(MIN)
    h = fi.bh(data, u, cntr, 2) #(MIN)
    b = fi.bws(data, u, cntr, 2) #(MAX)
    Y.append(y)
    Z.append(z)
    W.append(w)
    A.append(a)
    H.append(h)
    B.append(b)
    plt.subplot(231)
    plt.plot(i, y, color="blue", marker="s",markersize = 3)
    plt.subplot(232)
    plt.plot(i, z, color="red", marker="s",markersize = 3)
    plt.subplot(233)
    plt.plot(i, w, color="yellow", marker="s",markersize = 3)
    plt.subplot(234)
    plt.plot(i, a, color="black", marker="s",markersize = 3)
    plt.subplot(235)
    plt.plot(i, h, color="pink", marker="s",markersize = 3)
    plt.subplot(236)
    plt.plot(i, b, color="green", marker="s",markersize = 3)
    
#plot do gráfico de linhas:
plt.subplot(231)
plt.plot(X,Y, color="blue")
plt.subplot(232)
plt.plot(X, Z, color="red")
plt.subplot(233) 
plt.plot(X, W, color="yellow")
plt.subplot(234)
plt.plot(X, A, color="black")
plt.subplot(235) 
plt.plot(X, H, color="pink")
plt.subplot(236)
plt.plot(X, B, color="green")

#Marcar os melhores resultados:
plt.subplot(231)
c = X[np.argmax(Y)]
plt.plot([c, c], [np.min(Y), np.max(Y)], color="orange")
plt.subplot(232)
c = X[np.argmin(Z)]
plt.plot([c, c], [np.min(Z), np.max(Z)], color="orange")
plt.subplot(233)
c = X[np.argmin(W)]
plt.plot([c, c], [np.min(W), np.max(W)], color="orange")
plt.subplot(234)
c = X[np.argmin(A)]
plt.plot([c, c], [np.min(A), np.max(A)], color="orange")
plt.subplot(235)
c = X[np.argmin(H)]
plt.plot([c, c], [np.min(H), np.max(H)], color="orange")
plt.subplot(236)
c = X[np.argmax(B)]
plt.plot([c, c], [np.min(B), np.max(B)], color="orange")

plt.show()

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTE SUPERVISIONADO-------------------------------------------------------------------------------------

cntr, u, u0, d, jm, p, fpc = fcm(data=alldata, c=i, m=2, error=0.01, maxiter=1000)
labels = np.argmax(u, axis=0)
print("Acuracia = {0}".format(sm.accuracy_score(target,labels))) #Não sei se funciona corretamente ainda
print("Rand = {0}".format(sm.adjusted_rand_score(target, labels))) #comparação de concordâncias e discordâncias. (max)  
print("Completeness = {0}".format(sm.completeness_score(target, labels))) #computa de acordo com se os pontos de uma mesma label continuam em um mesmo grupo. (max)
print("Homogeneity = {0}".format(sm.homogeneity_score(target,labels))) #computa de acordo com a diversidade de labels dentro de um mesmo cluster. (max)
print("FolkesMallows = {0}".format(sm.fowlkes_mallows_score(target,labels))) #computa de acordo com o formato geomértico dos clusters(pares de pontos presentes no mesmo cluster ou em diferentes).
print("Mutual Info. = {0}".format(sm.normalized_mutual_info_score(target,labels,"geometric"))) #Medida estatistica usada para avaliar o co-ocorrência de dados em varios lugares(no caso clusters)
print("V score = {0}".format(sm.v_measure_score(target,labels)))#média harmônica entre a homogeneidade e completância.

#--------------------------------------------------------------------------------------------------------
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
#color = ['#00091a', '#001b4d', '#002d80', '#003eb3', '#0050e6', '#1a6aff', '#4d8bff', '#80acff', '#b3cdff', '#e6eeff'] #AZUL
#color = ['#003325', '#ffcccc', '#009970', '#ff6666', '#00ffbf', '#ff0000', '#66ffd9', '#990000', '#ccfff2', '#330000'] #GREEN RED
color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
saidas= ["Triade", "Grupo Aneis", "Interlaço", "Anel interno", "Ruidos", "Luas"]
colorainbow = list(matplotlib.colors.cnames.values())

data,_ = Datas.load_iris(return_X_y=True)
alldata = np.array(data).T

#APLICAÇÃO DO ALGORITMO----------------------------------------------------------------------------------

#Algoritmo de Agrupamento:
inicio = timeit.time.perf_counter()
cntr, u, u0, d, jm, p, fpc = fcm(data=alldata, c=3, m=100, error=0.01, maxiter=1000) #6 2 6 3 5 6
fim = timeit.time.perf_counter()
labels = np.argmax(u, axis=0)
#print("Duração: {0} segundos".format(fim - inicio))

colors = []
for i in range(len(data)):
    colors.append(color[int(labels[i])])

#Gráfico:
f, axes = plt.subplots(1, 2, figsize=(11,5))
plt.title("{0} Tempo = {1} segundos".format(saidas[5], round((fim - inicio),5)))
axes[0].scatter(data[:,0], data[:,1], marker="o", c="#0099cc",)
axes[1].scatter(data[:,0], data[:,1], marker="o", c=colors) 
plt.show()

print("{0}".format(round(sm.silhouette_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.calinski_harabasz_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.davies_bouldin_score(X=data, labels=labels), 5)))
print("{0}".format(round(dunn_fast(points=data, labels=labels), 5)))
print("{0}".format(round(fi.pc(data, u, cntr, 2), 5)))
print("{0}".format(round(fi.fhv(data, u, cntr, 2), 5)))
print("{0}".format(round(fi.fs(data, u, cntr, 2), 5)))
print("{0}".format(round(fi.xb(data, u, cntr, 2), 5)))

#Tabela de Pertinência:
#np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#print()
#print(tabulate([["C1", u[0,:]], ["C2", u[1,:]]], headers=['Clusters', 'Pertinência']))
#print()

#--------------------------------------------------------------------------------------------------------


