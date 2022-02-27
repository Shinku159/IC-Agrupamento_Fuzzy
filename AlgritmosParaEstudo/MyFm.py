import numpy as np

from sklearn import datasets as Datas
from sklearn.cluster import KMeans
from sklearn import metrics as sm
#-----------------------------------------
data,target = Datas.load_wine(return_X_y=True)

def FM(true, labels):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)-1):
        for j in range(i+1, len(labels)):
            if((true[i] == true[j]) and (labels[i] == labels[j])):
                TP += 1
            elif((true[i] == true[j]) and (labels[i] != labels[j])):
                FP += 1
            elif((true[i] != true[j]) and (labels[i] == labels[j])):
                FN += 1

    FMscore = TP / (((TP + FP) * (TP + FN))**0.5)

    return FMscore

cluster = KMeans(n_clusters=3)
cluster.fit(data)

print(FM(target, cluster.labels_))
print(sm.fowlkes_mallows_score(target, cluster.labels_))
