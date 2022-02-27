import GraficosCod as gc
import FuzzyIndex as fi
import numpy as np
import timeit
import Aneis
import time

from sklearn.cluster import AgglomerativeClustering as Ac
from seaborn import scatterplot as scatter
from skfuzzy.cluster import cmeans as fcm
from sklearn import datasets as Datas
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics as sm
from tabulate import tabulate
from jqmcvi import dunn_fast
from PcmAlg import pcm
from GkAlg import GK
#----------------------------------------------------------------------  

def FinalResults(data, ks, linkage, name):

  funcName=['Kmeans', 'Aglo','Fcm', 'Gk', 'Pcm']
  functions = [KMeans(n_clusters=ks[0]), Ac(n_clusters=ks[1], linkage=linkage), fcm, GK, pcm]
  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  
  for x in range(5):
    if(x <=1):
      inicio = timeit.time.perf_counter()
      cluster = functions[x]
      cluster.fit(data)
      fim = timeit.time.perf_counter()
      labels = cluster.labels_

      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      plt.scatter(data[:,0], data[:,1], marker="o", c=colors)
      plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))

      path = "TESTANDO/{0}_{1}Resultado.png".format(funcName[x], name)
      plt.savefig(path)
      plt.clf()

      path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
      file = open(path, 'w')
      file.write("{0}\n{1}\n{2}\n{3}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
      round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
      round(sm.davies_bouldin_score(X=data, labels=labels), 5),
      round(dunn_fast(points=data, labels=labels), 5)))
      file.close()

      path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
      file = open(path, 'w')
      for b in range(len(labels)):
        file.write("{0} ".format(labels[b]))
      file.close()
    else:
      if(x == 2):
        inicio = timeit.time.perf_counter() 
        cntr, u, u0, d, jm, p, fpc = functions[x](data=np.array(data).T, c=ks[2], m=2, error=0.0001, maxiter=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
      elif(x == 3):
        inicio = timeit.time.perf_counter() 
        cluster = functions[x](n_clusters=ks[3]) 
        cluster.fit(data)
        labels = np.argmax(cluster.u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = cluster.centers  
        u = cluster.u
        
      else:
        inicio = timeit.time.perf_counter() 
        v, v0, u, u0, d, t  = functions[x](np.array(data).T, c=ks[4], m=2, e=0.0001, max_iterations=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = v
        for norm in range(len(u[0])):
              deno = np.sum(u[:,norm])
              for norm1 in range(len(u)):
                u[norm1,norm] = u[norm1,norm]/deno 
      
      
      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      plt.scatter(data[:,0], data[:,1], marker="o", c=colors)
      plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))

      path = "TESTANDO/{0}_{1}Resultado.png".format(funcName[x], name)
      plt.savefig(path)
      plt.clf()

      if(len(set(labels)) != 1):
        path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
        file = open(path, 'w')
        file.write("{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
        round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
        round(sm.davies_bouldin_score(X=data, labels=labels), 5),
        round(dunn_fast(points=data, labels=labels), 5),
        round(fi.pc(data, u, cntr, 2), 5),
        round(fi.fhv(data, u, cntr, 2), 5),
        round(fi.fs(data, u, cntr, 2), 5),
        round(fi.xb(data, u, cntr, 2), 5)))
        file.close()

        path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(labels)):
          file.write("{0} ".format(labels[x]))
        file.close()

        path = "TESTANDO/{0}_{1}Pertinencias.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(u)):
          for y in range(len(u[b])):
            file.write("{0} ".format(u[b][y]))
          file.write("\n")
        file.close()

def LogicalResultsInsurence(data, ks, linkage, name):
  labelsName = ['Idade','Sexo', 'IMC', 'N Filhos','Fumante', 'Regiao', 'Custo']
  funcName=['Kmeans', 'Aglo', 'Meanshift', 'Fcm', 'Gk', 'Pcm']
  functions = [KMeans(n_clusters=ks[0]), Ac(n_clusters=ks[1], linkage=linkage), MeanShift(), fcm, GK, pcm]
  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  
  for x in range(6):
    if(x <=2):
      inicio = timeit.time.perf_counter()
      cluster = functions[x]
      cluster.fit(data)
      fim = timeit.time.perf_counter()
      labels = cluster.labels_

      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      contador = 1
      for k in range(6):
        for j in range(k+1, 7):
          plt.scatter(data[:,k], data[:,j], marker="o", c=colors)
          plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))
          plt.ylabel(labelsName[j])
          plt.xlabel(labelsName[k])
          path = "TESTANDO/{0}_{1}Resultado{2}.png".format(funcName[x], name, contador)
          plt.savefig(path)
          plt.clf()
          contador += 1

      path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
      file = open(path, 'w')
      file.write("{0}\n{1}\n{2}\n{3}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
      round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
      round(sm.davies_bouldin_score(X=data, labels=labels), 5),
      round(dunn_fast(points=data, labels=labels), 5)))
      file.close()

      path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
      file = open(path, 'w')
      for b in range(len(labels)):
        file.write("{0} ".format(labels[b]))
      file.close()
    else:
      if(x == 3):
        inicio = timeit.time.perf_counter() 
        cntr, u, u0, d, jm, p, fpc = functions[x](data=np.array(data).T, c=ks[2], m=2, error=0.0001, maxiter=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
      elif(x == 4):
        inicio = timeit.time.perf_counter() 
        cluster = functions[x](n_clusters=ks[3]) 
        cluster.fit(data)
        labels = np.argmax(cluster.u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = cluster.centers  
        u = cluster.u
        
      else:
        inicio = timeit.time.perf_counter() 
        v, v0, u, u0, d, t  = functions[x](np.array(data).T, c=ks[4], m=2, e=0.0001, max_iterations=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = v
        for norm in range(len(u[0])):
              deno = np.sum(u[:,norm])
              for norm1 in range(len(u)):
                u[norm1,norm] = u[norm1,norm]/deno 
      
      
      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      contador = 1
      for k in range(6):
        for j in range(k+1, 7):
          plt.scatter(data[:,k], data[:,j], marker="o", c=colors)
          plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))
          plt.ylabel(labelsName[j])
          plt.xlabel(labelsName[k])
          path = "TESTANDO/{0}_{1}Resultado{2}.png".format(funcName[x], name, contador)
          plt.savefig(path)
          plt.clf()
          contador += 1

      if(len(set(labels)) != 1):
        path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
        file = open(path, 'w')
        file.write("{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
        round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
        round(sm.davies_bouldin_score(X=data, labels=labels), 5),
        round(dunn_fast(points=data, labels=labels), 5),
        round(fi.pc(data, u, cntr, 2), 5),
        round(fi.fhv(data, u, cntr, 2), 5),
        round(fi.fs(data, u, cntr, 2), 5),
        round(fi.xb(data, u, cntr, 2), 5)))
        file.close()

        path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(labels)):
          file.write("{0} ".format(labels[x]))
        file.close()
          

        path = "TESTANDO/{0}_{1}Pertinencias.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(u)):
          for y in range(len(u[b])):
            file.write("{0} ".format(u[b][y]))
          file.write("\n")
        file.close()

def LogicalResultsCostumer(data, ks, linkage, name):
  labelsName = ['Quatidade','PrecoUni', 'Cliente', 'pais']
  funcName=['Kmeans', 'Aglo', 'Meanshift', 'Fcm', 'Gk', 'Pcm']
  functions = [KMeans(n_clusters=ks[0]), Ac(n_clusters=ks[1], linkage=linkage), MeanShift(), fcm, GK, pcm]
  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  
  for x in range(6):
    if(x <=2):
      inicio = timeit.time.perf_counter()
      cluster = functions[x]
      cluster.fit(data)
      fim = timeit.time.perf_counter()
      labels = cluster.labels_

      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      contador = 1
      for k in range(3):
        for j in range(k+1, 4):
          plt.scatter(data[:,k], data[:,j], marker="o", c=colors)
          plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))
          plt.ylabel(labelsName[j])
          plt.xlabel(labelsName[k])
          path = "TESTANDO/{0}_{1}Resultado{2}.png".format(funcName[x], name, contador)
          plt.savefig(path)
          plt.clf()
          contador += 1

      path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
      file = open(path, 'w')
      file.write("{0}\n{1}\n{2}\n{3}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
      round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
      round(sm.davies_bouldin_score(X=data, labels=labels), 5),
      round(dunn_fast(points=data, labels=labels), 5)))
      file.close()

      path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
      file = open(path, 'w')
      for b in range(len(labels)):
        file.write("{0} ".format(labels[b]))
      file.close()
    else:
      if(x == 3):
        inicio = timeit.time.perf_counter() 
        cntr, u, u0, d, jm, p, fpc = functions[x](data=np.array(data).T, c=ks[2], m=2, error=0.0001, maxiter=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
      elif(x == 4):
        inicio = timeit.time.perf_counter() 
        cluster = functions[x](n_clusters=ks[3]) 
        cluster.fit(data)
        labels = np.argmax(cluster.u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = cluster.centers  
        u = cluster.u
        
      else:
        inicio = timeit.time.perf_counter() 
        v, v0, u, u0, d, t  = functions[x](np.array(data).T, c=ks[4], m=2, e=0.0001, max_iterations=1000)
        labels = np.argmax(u, axis=0)
        fim = timeit.time.perf_counter() 
        cntr = v
        for norm in range(len(u[0])):
              deno = np.sum(u[:,norm])
              for norm1 in range(len(u)):
                u[norm1,norm] = u[norm1,norm]/deno 
      
      
      colors = []
      for j in range(len(data)):
          colors.append(color[int(labels[j])])
      contador = 1
      for k in range(3):
        for j in range(k+1, 4):
          plt.scatter(data[:,k], data[:,j], marker="o", c=colors)
          plt.title("{0} Tempo = {1} segundos".format(funcName[x], round((fim - inicio),5)))
          plt.ylabel(labelsName[j])
          plt.xlabel(labelsName[k])
          path = "TESTANDO/{0}_{1}Resultado{2}.png".format(funcName[x], name, contador)
          plt.savefig(path)
          plt.clf()
          contador += 1

      if(len(set(labels)) != 1):
        path = "TESTANDO/{0}_{1}TableFinale.txt".format(funcName[x], name)
        file = open(path, 'w')
        file.write("{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}".format(round(sm.silhouette_score(X=data, labels=labels), 5),
        round(sm.calinski_harabasz_score(X=data, labels=labels), 5), 
        round(sm.davies_bouldin_score(X=data, labels=labels), 5),
        round(dunn_fast(points=data, labels=labels), 5),
        round(fi.pc(data, u, cntr, 2), 5),
        round(fi.fhv(data, u, cntr, 2), 5),
        round(fi.fs(data, u, cntr, 2), 5),
        round(fi.xb(data, u, cntr, 2), 5)))
        file.close()

        path = "TESTANDO/{0}_{1}Label.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(labels)):
          file.write("{0} ".format(labels[x]))
        file.close()
          

        path = "TESTANDO/{0}_{1}Pertinencias.txt".format(funcName[x], name)
        file = open(path, 'w')
        for b in range(len(u)):
          for y in range(len(u[b])):
            file.write("{0} ".format(u[b][y]))
          file.write("\n")
        file.close()


def plotSaveFuzzy(functions, data, maxy, name):
  funcName=['Fcm', 'Gk', 'Pcm']
  alldata = np.array(data).T
  for x in range(2):
    Y=np.array([])
    Z=np.array([])
    W=np.array([])
    A=np.array([])

    YY=np.array([np.zeros(maxy-2)])
    ZZ=np.array([np.zeros(maxy-2)])
    WW=np.array([np.zeros(maxy-2)])
    AA=np.array([np.zeros(maxy-2)])

    for j in range(30):
      for i in range(2,maxy):
        if(x == 0):
          cntr, u, u0, d, jm, p, fpc = functions[0](data=alldata, c=i, m=2, error=0.0001, maxiter=1000)
        elif(x == 1):
          cluster = functions[1](n_clusters=i)
          cluster.fit(data)
          cntr = cluster.centers
          u = cluster.u
        else:
          v, v0, u, u0, d, t  = functions[2](alldata, c=i, m=2, e=0.0001, max_iterations=1000)
          cntr = v
          for norm in range(len(u[0])):
            deno = np.sum(u[:,norm])
            for norm1 in range(len(u)):
              u[norm1,norm] = u[norm1,norm]/deno
        labels = np.argmax(u, axis=0)
        y = fi.pc(None, u, None, None) #(MAX)
        z = fi.fhv(data, u, cntr, 2) #(MIN)
        w = fi.fs(data, u, cntr, 2) #(MIN) 
        a = fi.xb(data, u, cntr, 2) #(MIN)
        #h = fi.bh(data, u, cntr, 2) #(MIN)
        #b = fi.bws(data, u, cntr, 2) #(MAX)

        Y = np.append(Y, y)
        Z = np.append(Z, z)
        W = np.append(W, w)
        A = np.append(A, a)
        #H = H.append(H, h)
        #B = B.append(B, b)

      YY = np.concatenate((YY, [Y]))
      Y = np.array([])
      ZZ = np.concatenate((ZZ, [Z]))
      Z = np.array([])
      WW = np.concatenate((WW, [W]))
      W = np.array([])
      AA = np.concatenate((AA, [A]))
      A = np.array([])

    X = np.arange(2, maxy)
    YY = np.delete(YY, 0, 0)
    YY = np.mean(YY, axis=0)
    ZZ = np.delete(ZZ, 0, 0)
    ZZ = np.mean(ZZ, axis=0)
    WW = np.delete(WW, 0, 0)
    WW = np.mean(WW, axis=0)
    AA = np.delete(AA, 0, 0)
    AA = np.mean(AA, axis=0)

    j = 0 #TRA DE
    for i in range(2, maxy):
        plt.subplot(221)
        plt.plot(i, YY[j], color="g", marker="s",markersize = 3) #(MAX)
        plt.subplot(222)
        plt.plot(i, ZZ[j], color="r", marker="s",markersize = 3) #(MAX)
        plt.subplot(223)
        plt.plot(i, WW[j], color="blue", marker="s",markersize = 3) #(MIN)
        plt.subplot(224)
        plt.plot(i, AA[j], color="purple", marker="s",markersize = 3) #(MAX)
        j = j+1


    #plot do gráfico de linhas:
    plt.subplot(221)
    plt.plot(X, YY, color="g")
    plt.subplot(222)
    plt.plot(X, ZZ, color="r")
    plt.subplot(223)
    plt.plot(X, WW, color="blue")
    plt.subplot(224)
    plt.plot(X, AA, color="purple")

    #Marcar os melhores resultados:
    plt.subplot(221)
    c = X[np.argmax(YY)]
    plt.plot([c, c], [np.min(YY), np.max(YY)], color="orange")
    plt.title("Part. Coef")
    plt.subplot(222)
    c = X[np.argmin(ZZ)]
    plt.plot([c, c], [np.min(ZZ), np.max(ZZ)], color="orange")
    plt.title("Hypervolume")
    plt.subplot(223)
    c = X[np.argmin(WW)]
    plt.plot([c, c], [np.min(WW), np.max(WW)], color="orange")
    plt.title("Fukuyama")
    plt.subplot(224)
    c = X[np.argmin(AA)]
    plt.plot([c, c], [np.min(AA), np.max(AA)], color="orange")
    plt.title("Xie-Beni")

    if(maxy == 11):
      path = "TESTANDO/{0}_{1}.png".format(funcName[x], name)
    else:
      path = "TESTANDO/{0}_{1}Large.png".format(funcName[x], name)

    plt.savefig(path)
    plt.clf()

    if(maxy == 11):
      #Tabela de Pertinência:
      path = "TESTANDO/{0}_{1}Table.txt".format(funcName[x], name)
      file = open(path, 'w')
      np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
      file.write(tabulate([["Part. Coef", YY[0], YY[1], YY[2], YY[3],YY[4], YY[5], YY[6], YY[7], YY[8]], 
                    ["Hypervolume", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                    ["Fukuyama",  WW[0], WW[1], WW[2], WW[3],WW[4], WW[5], WW[6], WW[7], WW[8]], 
                    ["Xie Beni", AA[0], AA[1], AA[2], AA[3],AA[4], AA[5], AA[6], AA[7], AA[8]]], 
                    headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                    tablefmt="plain"))
      file.close()

def plotSaveCrisp(functions, data, maxy, name, linkage):
  funcName = ['Kmeans', 'Aglo', 'Mean']
  for x in range(3):
    Z=np.array([])
    K=np.array([])
    D=np.array([])
    U=np.array([])

    ZZ=np.array([np.zeros(maxy-2)])
    KK=np.array([np.zeros(maxy-2)])
    DD=np.array([np.zeros(maxy-2)])
    UU=np.array([np.zeros(maxy-2)])

    for j in range(30):
        for i in range(2,maxy):
            if(x == 0):
              clusters = functions[x](n_clusters=i)
            elif(x == 1):
              clusters = functions[x](n_clusters=i, linkage=linkage)
            clusters.fit(data)
            Z = np.append(Z, sm.silhouette_score(X=data, labels=clusters.labels_))
            K = np.append(K, sm.calinski_harabasz_score(X=data, labels=clusters.labels_))
            D = np.append(D, sm.davies_bouldin_score(X=data, labels=clusters.labels_))
            U = np.append(U, dunn_fast(points=data, labels=clusters.labels_))

        ZZ = np.concatenate((ZZ, [Z]))
        Z=np.array([])
        KK = np.concatenate((KK, [K]))
        K=np.array([])
        DD = np.concatenate((DD, [D]))
        D=np.array([])
        UU = np.concatenate((UU, [U]))
        U =np.array([])

    X = np.arange(2, maxy)
    ZZ = np.delete(ZZ, 0, 0)
    ZZ = np.mean(ZZ, axis=0)
    KK = np.delete(KK, 0, 0)
    KK = np.mean(KK, axis=0)
    DD = np.delete(DD, 0, 0)
    DD = np.mean(DD, axis=0)
    UU = np.delete(UU, 0, 0)
    UU = np.mean(UU, axis=0)

    j = 0
    for i in range(2, maxy):
        plt.subplot(221)
        plt.plot(i, ZZ[j], color="g", marker="s",markersize = 3) #(MAX)
        plt.subplot(222)
        plt.plot(i, KK[j], color="r", marker="s",markersize = 3) #(MAX)
        plt.subplot(223)
        plt.plot(i, DD[j], color="blue", marker="s",markersize = 3) #(MIN)
        plt.subplot(224)
        plt.plot(i, UU[j], color="purple", marker="s",markersize = 3) #(MAX)
        j = j+1


    #plot do gráfico de linhas:
    plt.subplot(221)
    plt.plot(X,ZZ, color="g")
    plt.subplot(222)
    plt.plot(X, KK, color="r")
    plt.subplot(223)
    plt.plot(X, DD, color="blue")
    plt.subplot(224)
    plt.plot(X, UU, color="purple")

    #Marcar os melhores resultados:
    plt.subplot(221)
    c = X[np.argmax(ZZ)]
    plt.plot([c, c], [np.min(ZZ), np.max(ZZ)], color="orange")
    plt.title("Silhouette")
    plt.subplot(222)
    c = X[np.argmax(KK)]
    plt.plot([c, c], [np.min(KK), np.max(KK)], color="orange")
    plt.title("Calinski")
    plt.subplot(223)
    c = X[np.argmin(DD)]
    plt.plot([c, c], [np.min(DD), np.max(DD)], color="orange")
    plt.title("Davies")
    plt.subplot(224)
    c = X[np.argmax(UU)]
    plt.plot([c, c], [np.min(UU), np.max(UU)], color="orange")
    plt.title("Dunn")

    if(maxy == 11):
      path = "TESTANDO/{0}_{1}.png".format(funcName[x], name)
    else:
      path = "TESTANDO/{0}_{1}Large.png".format(funcName[x], name)

    plt.savefig(path)
    plt.clf()

    if(maxy == 11):
      #Tabela de Pertinência:
      path = "TESTANDO/{0}_{1}Table.txt".format(funcName[x], name)
      file = open(path, 'w')
      np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
      file.write(tabulate([["silhouette", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                      ["calinski", KK[0], KK[1], KK[2], KK[3],KK[4], KK[5], KK[6], KK[7], KK[8]], 
                      ["davies",  DD[0], DD[1], DD[2], DD[3],DD[4], DD[5], DD[6], DD[7], DD[8]], 
                      ["dunn", UU[0], UU[1], UU[2], UU[3],UU[4], UU[5], UU[6], UU[7], UU[8]]], 
                      headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                      tablefmt="plain"))
      file.close()
      

def GeraGrafico(datas, linkages):
  if(not datas): #len(ks) != len(datas)*5
    print("Erro na Inicialização do procedimento")
    return

  m = len(datas)
  for i in range(m):
    functionsC = [KMeans, Ac]
    functionsF = [fcm, GK, pcm]
    plotSaveCrisp(functionsC, datas[i], 11, name=i, linkage=linkages[i])
    plotSaveFuzzy(functionsF, datas[i], 11, name=i)
    plotSaveCrisp(functionsC, datas[i], 31, name=i, linkage=linkages[i])
    plotSaveFuzzy(functionsF, datas[i], 31, name=i)

def Resultados(datas, linkages, ks):
  if(not datas): #len(ks) != len(datas)*5
    print("Erro na Inicialização do procedimento")
    return

  m = len(datas)
  for i in range(m):
    FinalResults(datas[i], ks[i], linkages[i], i)

def ResultadosReais(datas, linkages, ks):
  if(not datas): #len(ks) != len(datas)*5
    print("Erro na Inicialização do procedimento")
    return

  LogicalResultsInsurence(datas[0], ks[0], linkages[0], 0)
  LogicalResultsCostumer(datas[1], ks[1], linkages[1], 1)

