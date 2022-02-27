import FuzzyIndex as fi
import numpy as np
import timeit
import Aneis
import time

from sklearn.cluster import AgglomerativeClustering as Ac
from scipy.cluster.hierarchy import dendrogram, linkage
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

''' ATENÇÃO!!!!
FUNÇÕES AUXILIARES PARA GERAÇÃO AUTOMATICA DOS GRÁFICOS EM GRAFICOS AVANÇADO
'''

def KmeansGraf(maxy, data):
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
          clusters = KMeans(n_clusters=i)
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

  plt.show()

  #Tabela de Pertinência:
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["silhouette", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                  ["calinski", KK[0], KK[1], KK[2], KK[3],KK[4], KK[5], KK[6], KK[7], KK[8]], 
                  ["davies",  DD[0], DD[1], DD[2], DD[3],DD[4], DD[5], DD[6], DD[7], DD[8]], 
                  ["dunn", UU[0], UU[1], UU[2], UU[3],UU[4], UU[5], UU[6], UU[7], UU[8]]], 
                  headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                  tablefmt="plain"))
  print()

def AglomerativoGraf(maxy, data, linkage):
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
          clusters = Ac(n_clusters=i, linkage = linkage)
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

  plt.show()

  #Tabela de Pertinência:
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["silhouette", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                  ["calinski", KK[0], KK[1], KK[2], KK[3],KK[4], KK[5], KK[6], KK[7], KK[8]], 
                  ["davies",  DD[0], DD[1], DD[2], DD[3],DD[4], DD[5], DD[6], DD[7], DD[8]], 
                  ["dunn", UU[0], UU[1], UU[2], UU[3],UU[4], UU[5], UU[6], UU[7], UU[8]]], 
                  headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                  tablefmt="plain"))
  print()

def AgloDendograma(data):
  L = linkage(data, 'ward')

  dendrogram(L, truncate_mode = 'lastp', p = 12, leaf_rotation = 45, leaf_font_size=15, show_contracted=True)
  plt.title('Dendrograma Hierarquico')
  plt.xlabel('Clusters')
  plt.ylabel('Distancias')

  plt.axhline(y=150)
  plt.show()

def AgLinkTest(K, data):
  Agcluster = Ac(n_clusters=K, linkage = 'single')
  Agcluster.fit(data) 
  print(sm.silhouette_score(data, Agcluster.labels_))

  Agcluster = Ac(n_clusters=K, linkage = 'complete')
  Agcluster.fit(data) 
  print(sm.silhouette_score(data, Agcluster.labels_))

  Agcluster = Ac(n_clusters=K, linkage = 'average')
  Agcluster.fit(data) 
  print(sm.silhouette_score(data, Agcluster.labels_))

  Agcluster = Ac(n_clusters=K, linkage = 'ward')
  Agcluster.fit(data) 
  print(sm.silhouette_score(data, Agcluster.labels_))

def MeanShiftAverage(data):
  clusters = MeanShift()
  clusters.fit(data)

  Z = np.array((sm.silhouette_score(X=data, labels=clusters.labels_)))
  K = np.array((sm.calinski_harabasz_score(X=data, labels=clusters.labels_)))
  D = np.array((sm.davies_bouldin_score(X=data, labels=clusters.labels_)))
  U = np.array((dunn_fast(points=data, labels=clusters.labels_)))

  for i in range(29):
    clusters = MeanShift()
    clusters.fit(data)
    Z = np.append(Z, (sm.silhouette_score(X=data, labels=clusters.labels_)))
    K = np.append(K, (sm.calinski_harabasz_score(X=data, labels=clusters.labels_)))
    D = np.append(D, (sm.davies_bouldin_score(X=data, labels=clusters.labels_)))
    U = np.append(U, (dunn_fast(points=data, labels=clusters.labels_)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["silhouette", Z], 
                  ["calinski", K], 
                  ["davies",  D], 
                  ["dunn", U]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))
  print()
  
def FcmGraf(maxy, data, alldata, m):
  
  Y=np.array([])
  Z=np.array([])
  W=np.array([])
  A=np.array([])
  #H=np.array([])
  #B=np.array([])
  YY=np.array([np.zeros(maxy-2)])
  ZZ=np.array([np.zeros(maxy-2)])
  WW=np.array([np.zeros(maxy-2)])
  AA=np.array([np.zeros(maxy-2)])
  #HH=np.array([np.zeros(maxy-2)])
  #BB=np.array([np.zeros(maxy-2)])
  for j in range(30):
    for i in range(2,maxy):
      cntr, u, u0, d, jm, p, fpc = fcm(data=alldata, c=i, m=2, error=0.01, maxiter=1000)
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

  plt.show()

  #Tabela de Pertinência:
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["Part. Coef", YY[0], YY[1], YY[2], YY[3],YY[4], YY[5], YY[6], YY[7], YY[8]], 
                  ["Hypervolume", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                  ["Fukuyama",  WW[0], WW[1], WW[2], WW[3],WW[4], WW[5], WW[6], WW[7], WW[8]], 
                  ["Xie Beni", AA[0], AA[1], AA[2], AA[3],AA[4], AA[5], AA[6], AA[7], AA[8]]], 
                  headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                  tablefmt="plain"))
  print()

def GkGraf(maxy, data, alldata, m):
  
  Y=np.array([])
  Z=np.array([])
  W=np.array([])
  A=np.array([])
  #H=np.array([])
  #B=np.array([])
  YY=np.array([np.zeros(maxy-2)])
  ZZ=np.array([np.zeros(maxy-2)])
  WW=np.array([np.zeros(maxy-2)])
  AA=np.array([np.zeros(maxy-2)])
  #HH=np.array([np.zeros(maxy-2)])
  #BB=np.array([np.zeros(maxy-2)])
  for j in range(30):
    for i in range(2,maxy):
      GKcluster = GK(n_clusters=i)
      GKcluster.fit(data)  
      u = GKcluster.u
      cntr = GKcluster.centers
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

  j = 0
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
  plt.title("Part. Coef.")
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

  plt.show()

  #Tabela de Pertinência:
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["Part. Coef", YY[0], YY[1], YY[2], YY[3],YY[4], YY[5], YY[6], YY[7], YY[8]], 
                  ["Hypervolume", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                  ["Fukuyama",  WW[0], WW[1], WW[2], WW[3],WW[4], WW[5], WW[6], WW[7], WW[8]], 
                  ["Xie Beni", AA[0], AA[1], AA[2], AA[3],AA[4], AA[5], AA[6], AA[7], AA[8]]], 
                  headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                  tablefmt="plain"))
  print()

def PcmGraf(maxy, data, alldata, m):
  
  Y=np.array([])
  Z=np.array([])
  W=np.array([])
  A=np.array([])
  #H=np.array([])
  #B=np.array([])
  YY=np.array([np.zeros(maxy-2)])
  ZZ=np.array([np.zeros(maxy-2)])
  WW=np.array([np.zeros(maxy-2)])
  AA=np.array([np.zeros(maxy-2)])
  #HH=np.array([np.zeros(maxy-2)])
  #BB=np.array([np.zeros(maxy-2)])
  for j in range(30):
    for i in range(2,maxy):
      v, v0, u, u0, d, t = pcm(alldata, c=i, m=m, e=0.0001, max_iterations=1000)
      #Normaliza os valores de pcm.
      for norm in range(len(u[0])):
          deno = np.sum(u[:,norm])
          for norm1 in range(len(u)):
              u[norm1,norm] = u[norm1,norm]/deno
      
      labels = np.argmax(u, axis=0)
      y = fi.pc(None, u, None, None) #(MAX)
      z = fi.fhv(data, u, v, 2) #(MIN)
      w = fi.fs(data, u, v, 2) #(MIN) 
      a = fi.xb(data, u, v, 2) #(MIN)
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

  j = 0
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
  plt.title("Part. Coef.")
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

  plt.show()

  #Tabela de Pertinência:
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["Part. Coef", YY[0], YY[1], YY[2], YY[3],YY[4], YY[5], YY[6], YY[7], YY[8]], 
                  ["Hypervolume", ZZ[0], ZZ[1], ZZ[2], ZZ[3],ZZ[4], ZZ[5], ZZ[6], ZZ[7], ZZ[8]], 
                  ["Fukuyama",  WW[0], WW[1], WW[2], WW[3],WW[4], WW[5], WW[6], WW[7], WW[8]], 
                  ["Xie Beni", AA[0], AA[1], AA[2], AA[3],AA[4], AA[5], AA[6], AA[7], AA[8]]], 
                  headers=["coeficiente", '2','3','4','5','6','7','8','9','10'], 
                  tablefmt="plain"))
  print()

def KmeansTrue(data, target):
  n_clusters = len(set(target))
  inicio = timeit.time.perf_counter()
  clusters = KMeans(n_clusters=n_clusters)
  clusters.fit(data)
  fim = timeit.time.perf_counter()
  labels = clusters.labels_
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    clusters = KMeans(n_clusters=len(set(target)))
    clusters.fit(data)
    labels = clusters.labels_
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  '''print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))'''
  print(K)
  print(D)
  print(U)
  print(L)
  print(T)
  print(F)
  print()

  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  colors = []
  for i in range(len(data)):
      colors.append(color[int(labels[i])])
  #Gráfico:
  plt.title("Tempo = {0} segundos".format(round((fim - inicio),5)))
  plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
  plt.show()

def AgglomerativeTrue(data, target, linkage):
  n_clusters = len(set(target))
  inicio = timeit.time.perf_counter()
  clusters = Ac(n_clusters=n_clusters, linkage=linkage)
  clusters.fit(data)
  fim = timeit.time.perf_counter()
  labels = clusters.labels_
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    clusters = Ac(n_clusters=n_clusters, linkage=linkage)
    clusters.fit(data)
    labels = clusters.labels_
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  '''print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))'''
  print(K)
  print(D)
  print(U)
  print(L)
  print(T)
  print(F)
  print()

  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  colors = []
  for i in range(len(data)):
      colors.append(color[int(labels[i])])
  #Gráfico:
  plt.title("Tempo = {0} segundos".format(round((fim - inicio),5)))
  plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
  plt.show()

def MeanShiftTrue(data, target):
  n_clusters = len(set(target))
  clusters = MeanShift()
  clusters.fit(data)
  labels = clusters.labels_
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    clusters = MeanShift()
    clusters.fit(data)
    labels = clusters.labels_
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))
  print()

def FcmTrue(data, target):
  n_clusters = len(set(target))
  inicio = timeit.time.perf_counter()
  cntr, u, u0, d, jm, p, fpc = fcm(data=data, c=n_clusters, m=2, error=0.01, maxiter=1000)
  fim = timeit.time.perf_counter()
  labels = np.argmax(u, axis=0)
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    cntr, u, u0, d, jm, p, fpc = fcm(data=data, c=n_clusters, m=2, error=0.01, maxiter=1000)
    labels = np.argmax(u, axis=0)
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  '''print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))'''
  print(K)
  print(D)
  print(U)
  print(L)
  print(T)
  print(F)
  print()

  data = np.array(data).T
  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  colors = []
  for i in range(len(data)):
      colors.append(color[int(labels[i])])
  #Gráfico:
  plt.title("Tempo = {0} segundos".format(round((fim - inicio),5)))
  plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
  plt.show()

def GkTrue(data, target):
  n_clusters = len(set(target))
  inicio = timeit.time.perf_counter()
  GKcluster = GK(n_clusters=n_clusters)
  GKcluster.fit(data)
  fim = timeit.time.perf_counter()
  labels = np.argmax(GKcluster.u, axis=0)
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    GKcluster = GK(n_clusters=n_clusters)
    GKcluster.fit(data)
    labels = np.argmax(GKcluster.u, axis=0)
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))

  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  '''print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))'''
  print(K)
  print(D)
  print(U)
  print(L)
  print(T)
  print(F)
  print()

  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  colors = []
  for i in range(len(data)):
      colors.append(color[int(labels[i])])
  #Gráfico:
  plt.title("Tempo = {0} segundos".format(round((fim - inicio),5)))
  plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
  plt.show()

def PcmTrue(data, target):
  n_clusters = len(set(target))
  inicio = timeit.time.perf_counter()
  v, v0, u, u0, d, t = pcm(data, c=n_clusters, m=2, e=0.0001, max_iterations=1000)
  fim = timeit.time.perf_counter()
  labels = np.argmax(u, axis=0)
  Z = np.array((sm.accuracy_score(target, labels)))
  K = np.array((sm.adjusted_rand_score(target, labels)))
  D = np.array((sm.completeness_score(target, labels)))
  U = np.array((sm.homogeneity_score(target, labels)))
  L = np.array((sm.fowlkes_mallows_score(target, labels)))
  T = np.array((sm.normalized_mutual_info_score(target,labels,"geometric")))
  F = np.array((sm.v_measure_score(target, labels)))

  for i in range(29):
    v, v0, u, u0, d, t = pcm(data, c=n_clusters, m=2, e=0.0001, max_iterations=1000)
    labels = np.argmax(u, axis=0)
    Z = np.append(Z, (sm.accuracy_score(target, labels)))
    K = np.append(K, (sm.adjusted_rand_score(target, labels)))
    D = np.append(D, (sm.completeness_score(target, labels)))
    U = np.append(U, (sm.homogeneity_score(target, labels)))
    L = np.append(L, (sm.fowlkes_mallows_score(target, labels)))
    T = np.append(T, (sm.normalized_mutual_info_score(target,labels,"geometric")))
    F = np.append(F, (sm.v_measure_score(target, labels)))
 
  Z = np.mean(Z)
  K = np.mean(K)
  D = np.mean(D)
  U = np.mean(U)
  L = np.mean(L)
  T = np.mean(T)
  F = np.mean(F)
  
  data = np.array(data).T
  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print()
  '''print(tabulate([["Acuracia", Z], 
                  ["Rand", K], 
                  ["Completeness", D], 
                  ["Homogeneity", U],
                  ["FolkesMallows", L],
                  ["Mutual Info.", T],
                  ["V score", F]], 
                  headers=["coeficiente", "value"], 
                  tablefmt="plain"))'''
  print(K)
  print(D)
  print(U)
  print(L)
  print(T)
  print(F)
  print()

  color = ['#0099cc', '#00cc99', '#33cc33', '#ffff66', '#ff8533', '#ff0000', '#ffccef', '#cc33ff', '#333399'] #MULT SOFT
  colors = []
  for i in range(len(data)):
      colors.append(color[int(labels[i])])
  #Gráfico:
  plt.title("Tempo = {0} segundos".format(round((fim - inicio),5)))
  plt.scatter(data[:,0], data[:,1], marker="o", c=colors) 
  plt.show()

