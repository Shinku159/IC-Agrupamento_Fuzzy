import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import timeit
import Aneis
import time 

from seaborn import scatterplot as scatter
from sklearn.cluster import MeanShift
from sklearn import datasets as Datas
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
#TESTES NAO SUPERVISIONADOS------------------------------------------------------------------------------

clusters = MeanShift()
clusters.fit(data)
print("silhouette = {0}".format(sm.silhouette_score(X=data, labels=clusters.labels_)))
print("CH index = {0}".format(sm.calinski_harabasz_score(X=data, labels=clusters.labels_)))
print("Davies = {0}".format(sm.davies_bouldin_score(X=data, labels=clusters.labels_)))
print("Dunn = {0}".format(dunn_fast(points=data, labels=clusters.labels_)))
print("DBCV = {0}".format(DBCV(X=data, labels=clusters.labels_)))

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#TESTE SUPERVISIONADO------------------------------------------------------------------------------------

clusters = MeanShift()
clusters.fit(data)
labels = clusters.labels_
print("FolkesMallows = {0}".format(sm.fowlkes_mallows_score(target,labels))) #computa de acordo com o formato geomértico dos clusters(pares de pontos presentes no mesmo cluster ou em diferentes).
print("Mutual Info. = {0}".format(sm.normalized_mutual_info_score(target,labels,"geometric"))) #Medida estatistica usada para avaliar o co-ocorrência de dados em varios lugares(no caso clusters)
print("Completeness = {0}".format(sm.completeness_score(target, labels))) #computa de acordo com se os pontos de uma mesma label continuam em um mesmo grupo. (max)
print("Homogeneity = {0}".format(sm.homogeneity_score(target,labels))) #computa de acordo com a diversidade de labels dentro de um mesmo cluster. (max)
print("Acuracia = {0}".format(sm.accuracy_score(target,labels))) #Não sei se funciona corretamente ainda
print("V score = {0}".format(sm.v_measure_score(target,labels)))#média harmônica entre a homogeneidade e completância.
print("Rand = {0}".format(sm.adjusted_rand_score(target, labels))) #comparação de concordâncias e discordâncias. (max)

#--------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#color = ['#00091a', '#001b4d', '#002d80', '#003eb3', '#0050e6', '#1a6aff', '#4d8bff', '#80acff', '#b3cdff', '#e6eeff'] #AZUL
#color = ['#003325', '#ffcccc', '#009970', '#ff6666', '#00ffbf', '#ff0000', '#66ffd9', '#990000', '#ccfff2', '#330000'] #GREEN RED
color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
saidas= ["Triade", "Grupo Aneis", "Interlaço", "Anel interno", "Ruidos", "Luas"]
colorainbow = list(matplotlib.colors.cnames.values())

data,_ = Datas.load_boston(return_X_y=True)
#APLICAÇÃO DO ALGORITMO---------------------------------------------------------------------------------- 

#Algoritmo de Agrupamento:
inicio = timeit.time.perf_counter()
MScluster = MeanShift()
MScluster.fit(data)
fim = timeit.time.perf_counter()
labels = MScluster.labels_ 
#print("Duração: {0} segundos".format(fim - inicio))
colors = []
for i in range(len(data)):
    if(labels[i] != -1):
        colors.append(color[int(labels[i])])
    else:
        colors.append('black')

#Gráfico:
plt.title("{0} Tempo = {1} segundos".format('MeanShift', round((fim - inicio),5)))
plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
plt.show()


print("{0}".format(round(sm.silhouette_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.calinski_harabasz_score(X=data, labels=labels), 5)))
print("{0}".format(round(sm.davies_bouldin_score(X=data, labels=labels), 5)))
print("{0}".format(round(dunn_fast(points=data, labels=labels), 5)))

#--------------------------------------------------------------------------------------------------------
