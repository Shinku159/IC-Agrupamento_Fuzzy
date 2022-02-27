import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#confere qual distância é menor ---------------
def MenorDis(distancias):
    menor = 0
    for j in range(1, len(distancias)):
        if(distancia[menor] > distancias[j]):
            menor = j
            
    return menor

#Entrada ---------------------------
colors = ["g.", "r.", "c.", "b.", "k.", "o."]
Pontos = np.array([[9, 4],
                   [5.27, 7.8],
                   [3.1 , 4.44],
                   [5.8, 7],
                   [9, 9],
                   [10, 1],
                   [10, 10],
                   [7.7, 4],
                   [4, 3]])
k = 2
#-----------------------------------

#Selecionar os K-clusters -------------------------------------
index = np.random.choice(Pontos.shape[0],k, replace = False)
clusters = np.zeros((k, 2), dtype=float) 
for i in range(k):
    clusters[i] = Pontos[index[i]]

#PROCESSO DE AGRUPAMENTO---------------------------------------
distancia = np.zeros(k) #calcula a distancia dos pontos aos kluesters
labels = np.zeros(len(Pontos))
centroids = np.zeros((k, 2), dtype=float)
denominadorMedia = np.zeros(k)
f = 0
while(f == 0):
    centroids = np.zeros((k, 2), dtype=float)
    denominadorMedia = np.zeros(k)
    for i in range(len(Pontos)):
        for j in range(k):
            distancia[j]  = ((abs(Pontos[i,0] - clusters[j,0]) + abs(Pontos[i,1] - clusters[j,1]))**1/2)
        #print(distancia)
        labels[i] = MenorDis(distancias=distancia) #define a qual grupo cada ponto pertence
        centroids[int(labels[i]), 0] += Pontos[i, 0]
        centroids[int(labels[i]), 1] += Pontos[i, 1]
        denominadorMedia[int(labels[i])] += 1
    #print(labels)

    for i in range(k): #calcula as centroids
        centroids[int(labels[i]), 0] = centroids[int(labels[i]), 0]/denominadorMedia[int(labels[i])]
        centroids[int(labels[i]), 1] = centroids[int(labels[i]), 1]/denominadorMedia[int(labels[i])]
    #print(centroids)
    
    #saida do programa
    if(np.array_equal(centroids, clusters)):
        f = 1
    else:
        clusters = centroids
    
#imprime o gráfico do modelo
for i in range(len(Pontos)):
    plt.plot(Pontos[i][0], Pontos[i][1], colors[int(labels[i])], markersize = 10)
plt.scatter(centroids[:,0], centroids[:,1], marker = 'x', s = 150, linewidth = 5)
#plt.scatter(clusters[:,0], clusters[:,1], marker = '*', s = 150, linewidth = 5)
plt.show()


