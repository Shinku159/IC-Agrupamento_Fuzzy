import numpy as np

from sklearn import datasets as Datas
from sklearn.cluster import KMeans
from sklearn import metrics as sm
#-----------------------------------------
data,target = Datas.load_wine(return_X_y=True)

def dist(x, y):
    r = 0
    for i in range(len(x)):
        r += (abs(x[i] - y[i]))**2
    r = r**(1/2)
    return r

def Fat(n):
    fat = 1
    i = 2
    while i <= n:
        fat = fat*i
        i = i + 1
    
    return fat

def silhoute(data, labels):
    centers = np.zeros(shape=(len(set(labels)), len(data[0])))
    means = np.zeros(len(centers))
    intraC = np.zeros(len(centers))

    for i in range(len(labels)):
        centers[labels[i]] += data[i]
        means[labels[i]] += 1

    for i in range(len(centers)):
        centers[i] = centers[i]/means[i]
    
    a = 0
    s = np.array([])
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            if((labels[i] == labels[j])):
                intraC[labels[i]] += dist(data[i], data[j])

    for i in range(len(intraC)):
        intraC[i] = intraC[i]/(Fat(means[i])/(2*Fat(means[i] - 2)))
 
    for i in range(len(data)):
        b = np.array([])
        for j in range(len(centers)):
            if(labels[i] != j):
                b = np.append(b, dist(data[i], centers[j]))
        b = np.min(b)
        a = intraC[labels[i]]
        s  = np.append(s, (abs(b - a))/(max(a, b)))
        
    s = np.mean(s)

    return s

cluster = KMeans(n_clusters=3)
cluster.fit(data)

print(silhoute(data,cluster.labels_))
print(sm.silhouette_score(X=data, labels=cluster.labels_))

