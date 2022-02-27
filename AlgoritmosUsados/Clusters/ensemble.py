#IMPORTS===================================================
import ensembles as es
import numpy as np
import random
import copy
import math
import sys

from sklearn.cluster import AgglomerativeClustering as Ac
from seaborn import scatterplot as scatter
from sklearn.cluster import MeanShift
from sklearn import datasets as Datas
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics as sm
from tabulate import tabulate
from munkres import Munkres
from skfuzzy import cmeans
from PcmAlg import pcm
from GkAlg import GK
#==========================================================

''' ATENÇÃO!!!!!!
APLICAÇÃO CONJUNTA DO ENSEMBLE FUZZY E CRISP
'''
def MembershipNorm(u):
    """Normalizador de pertinência

    Função que assume valores de 0 a 1 para toda a matriz de pertinência
    de um dado agrupamento enviado, a soma de todos os valores ao final
    fica resolvida em 1

    Parâmetros
    ----------
    u: 2D Array
        matriz de pertinência gerada por um agrupamento fuzzy

    Returns
    -------
    u: 2D Arrar
        Matriz de pertinência normalizada
    """

    #para a pertinência de cada cluster, normalize para valores entre 0 e 1 dividindo pela soma de todos os fatores
    for i in range(len(u[0])):
        c = np.sum(u[:,i]) 
        for j in range(len(u)):
            u[j,i] = u[j,i]/c
    return u

def HighLinkage(data, k):
    """Método de escolha de melhor linkage

    Função usa método da silhueta para identificar o possivel
    melhor método de ligação para o algoritmo aglomerativo para
    uma base de dados com 'k' clusters

    Parâmetros
    ----------
    data : 2D np.array
        Um array 2D onde cada célula contem uma sequencia de valores
        númericos contendo o atributo de cada dado a ser agrupado
        (base de dados).
        
    k: inteiro
        Número de divisões a serem feitas pelo agrupamento.

    Returns
    -------
    linkage: string
        mátodo de ligação com melhor pontuação na silhueta

    nota
    ----
    Mais a frente sera desenvolvido um método para que seja enviado
    mais opções de coeficientes para essa avaliação.

    """
    
    linkages = ['single','complete', 'average', 'ward']
    score = np.array([])
    
    Agcluster = Ac(n_clusters=k, linkage = 'ward')
    Agcluster.fit(data) 
    score = np.append(score, sm.silhouette_score(data, Agcluster.labels_))

    Agcluster = Ac(n_clusters=k, linkage = 'single')
    Agcluster.fit(data) 
    score = np.append(score, sm.silhouette_score(data, Agcluster.labels_))

    Agcluster = Ac(n_clusters=k, linkage = 'complete')
    Agcluster.fit(data) 
    score = np.append(score, sm.silhouette_score(data, Agcluster.labels_))


    Agcluster = Ac(n_clusters=k, linkage = 'average')
    Agcluster.fit(data) 
    score = np.append(score, sm.silhouette_score(data, Agcluster.labels_))

    return linkages[np.argmax(score)]

#============================================================

class ensembleFuzzy:
  """Tática de ensemble via votação de métodos Fuzzy

    Esse algoritmo é desenvolvido de maneira a randomificar possiveis
    resultados para o agrupamento de uma base X nos modelos fuzzy do
    método possibilistico do c-médias, o c-médias e a otimização de gustafson kessel.

    Parametros
    ----------
    n_clusters: inteiro
        Número de divisões a serem feitas pelos métodos de agrupamento.

    maxRuns: inteiro, optional
        Número máximo de interações para gerar rótulos de agrupamento entre
        os métodos, de forma a fazer o ensemble. essa variavel é ignorada
        caso 'base' seja verdadeiro. Automaticamente iniciado com 100 caso
        não especificado.

    base: booleana, optional
        Realiza um ensemble básico apenas com a 'melhor' resposta de cada 
        algoritmo para a base de dados proposta, usando apenas uma interação.
        Esse valor é iniciado como falso caso não seja fornecido.

    Atributos
    ---------

    labels: np.array
        retorna uma lista com o rótulo de cada ponto do banco(data) informado,
        após realizar o ensemble.
    
    u: 2D Array
        retorna a matriz de pertinência resultante do agrupamento ensemble.

    Notas
    -----
        
    esse algoritmo não está totalmente otimizado, sendo apenas criado para
    estudos da linguagem de python e de suporte para um pojeto de iniciação
    cientifica.

    Referências
    -----------

    Agradecimentos a plataforma do SkFuzzy pela disposição do algoritmo para
    o fuzzy c-means, e dos usuários do GitHub Holt Skinne e  hmed Nour Jamal El-Din
    pela disposição do Possibilistic c-means e gustafson kessel, respectivamente.

    """

  def __init__(self,  n_clusters, maxRuns=100, base=False):
    self.c = n_clusters
    self.maxRuns = maxRuns
    self.base = base

  def fit(self, data):
    """Compute Fuzzy Ensemble Clustering
    
    Parametros
    ----------
    data : 2D np.array
        Um array 2D onde cada célula contem uma sequencia de valores
        númericos contendo o atributo de cada dado a ser agrupado
        (base de dados).

    Returns
    -------
    self
        Fitted estimator.

    """

    clusters = []
    alldata = np.array(data).T
    #ALGORITMOS BASE------------------------------
    #fuzzy c-means
    cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=self.c, m=2, error=0.0001, maxiter=1000)
    FCMlabels = np.argmax(u, axis=0)

    #possibilistic c-means
    v, v0, u2, u02, d2, t = pcm(alldata, c=self.c, m=2, e=0.0001, max_iterations=1000)
    u2 = MembershipNorm(u2)
    PCMlabels = np.argmax(u2, axis=0)
    
    #gustafson kessel
    GKcluster = GK(n_clusters=self.c)
    GKcluster.fit(data)
    GKlabels = np.argmax(GKcluster.u, axis=0)

    Memberships = [u]
    clusters.append(list(FCMlabels))

    if(len(set(GKlabels)) == self.c):
        Memberships = np.concatenate((Memberships, [GKcluster.u]))
        clusters.append(list(GKlabels))
    if(len(set(PCMlabels)) == self.c):
        Memberships = np.concatenate((Memberships, [u2]))
        clusters.append(list(PCMlabels))
    #----------------------------------------------
    if(not self.base):
        for run in range(self.maxRuns-1):
            #fcm randomizado
            cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=self.c, m=round(random.uniform(1.1,3.1), 1), error=0.0001, maxiter=1000)
            FCMlabels = np.argmax(u, axis=0)

            v, v0, u, u0, d, t = pcm(alldata, c=self.c, m=round(random.uniform(1.1,3.1), 1), e=0.0001, max_iterations=1000) 
            u2 = MembershipNorm(u2)
            PCMlabels = np.argmax(u2, axis=0)

            GKcluster = GK(n_clusters=self.c, max_iter=1000, m=round(random.uniform(1.1,3.1), 1), error=0.0001)
            GKcluster.fit(data)
            GKlabels = np.argmax(GKcluster.u, axis=0)

            if(len(set(FCMlabels)) == self.c and not math.isnan(u[0,0])):
              Memberships = np.concatenate((Memberships, [u]))
              clusters.append(list(FCMlabels))

            if(len(set(GKlabels)) == self.c and not math.isnan(GKcluster.u[0,0])):
                Memberships = np.concatenate((Memberships, [GKcluster.u]))
                clusters.append(list(GKlabels))
            if(len(set(PCMlabels)) == self.c and not math.isnan(u2[0,0])):
                Memberships = np.concatenate((Memberships, [u2]))
                clusters.append(list(PCMlabels))

    for idx in range(1, len(clusters)):
        Memberships[idx] = es.relabelFuzzy(base=clusters[0], array=clusters[idx], M=Memberships[idx])

    u = es.votingFuzzy(Memberships)
    self.labels = np.argmax(u, axis=0)
    self.u = MembershipNorm(u)

    return self

class ensembleCrisp:
  """Tática de ensemble via votação de métodos Crisp

  Esse algoritmo é desenvolvido de maneira a randomificar possiveis
  resultados para o agrupamento de uma base X nos modelos Crisp do
  método hierarquico aglormerativo, K-médias, e MeanShif, esse ultimo
  sendo incluso apenas quando resulta na mesma quantia de rótulos que 
  o valor 'k' exigido aos demais.

  Parametros
  ----------
  data : 2D np.array
      Um array 2D onde cada célula contem uma sequencia de valores
      númericos contendo o atributo de cada dado a ser agrupado
      (base de dados).

  n_clusters: inteiro
      Número de divisões a serem feitas pelos métodos de agrupamento.

  maxRuns: inteiro,
      Número máximo de interações para gerar rótulos de agrupamento entre
      os métodos, de forma a fazer o ensemble. essa variavel é ignorada
      caso 'base' seja verdadeiro. Automaticamente iniciado com 100 caso
        não especificado.

  base: booleana, optional
      Realiza um ensemble básico apenas com a melhor resposta de cada 
      algoritmo para a base de dados proposta, usando apenas uma interação.
      Esse valor é iniciado como falso caso não seja fornecido.

  Atributos
  ---------

  labels: np.array
      retorna uma lista com o rótulo de cada ponto do banco(data) informado,
      após realizar o ensemble.

  Notas
  -----
      
  esse algoritmo não está totalmente otimizado, sendo apenas criado para
  estudos da linguagem de python e de suporte para um pojeto de iniciação
  cientifica.

  Referências
  -----------

  Agradecimentos ao site do scikit pela disponibilização dos algoritmos e ao
  GitHub Kr4t0n de Kyle Yang, pelo algoritmo de votação e rerotulação usado como
  base.

  """
  def __init__(self, n_clusters, maxRuns=100, base=False):
    self.k = n_clusters
    self.maxRuns = maxRuns
    self.base = base

  def fit(self, data):
    """Compute Crisp Ensemble Clustering
    
    Parametros
    ----------
    data : 2D np.array
        Um array 2D onde cada célula contem uma sequencia de valores
        númericos contendo o atributo de cada dado a ser agrupado
        (base de dados).

    Returns
    -------
    self
        Estimadores Aclopados.

    """

    clusters = [] #mantem todos os agrupamentos para o ensemble
    if(self.base):
        #ALGORITMOS BASE------------------------------
        #meanshift
        MScluster = MeanShift()
        MScluster.fit(data)

        #kmeans
        KMcluster = KMeans(n_clusters=self.k)
        KMcluster.fit(data)

        #aglomerativo
        AGcluster = Ac(n_clusters=self.k, linkage=HighLinkage(data, self.k))
        AGcluster.fit(data)
        #----------------------------------------------

        #ENSEMBLE--------------------------------------
        clusters.append(list(KMcluster.labels_)) 
        clusters.append(list(AGcluster.labels_))
        if(len(set(MScluster.labels_)) == self.k): #adiciona o meanshift apenas caso tenha o mesmo número de rótulos que o k fornecido
            clusters.append(list(MScluster.labels_)) 
        #----------------------------------------------
    else:
        #ENSEMBLE--------------------------------------
        linkages = ['complete','average','single', 'ward'] #ligações aceitas pelo algoritmo aglomerativo
        affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] #tipos de calculos de distância aceitas pelo algortimo aglomerativo
        for run in range(self.maxRuns):
            #kmédias randomizado
            KMcluster = KMeans(n_clusters=self.k, init='random' ,n_init=1)
            KMcluster.fit(data)
            clusters.append(list(KMcluster.labels_))

            #para a versão usando a ligação ward, apenas a métrica euclidiana funciona.
            aux = random.randrange(0, 4)
            linkage = linkages[aux]
            affinity = 'euclidean'
            if(aux != 3):
                affinity = affinities[random.randrange(0, 5)]

            #Aglomerativo randomizado
            Agcluster = Ac(n_clusters=self.k, linkage=linkage, affinity=affinity)
            Agcluster.fit(data)
            clusters.append(list(Agcluster.labels_))

            #MeanShift randomizado
            MScluster = MeanShift(bandwidth=round(random.uniform(0.1,100.1), 2))
            MScluster.fit(data)
            if(len(set(MScluster.labels_)) == self.k): #adiciona o meanshift apenas caso tenha o mesmo número de rótulos que o k fornecido
                clusters.append(list(Agcluster.labels_))
        #----------------------------------------------
        
        #faz a re-rotulação dos agrupamentos de acordo com um primeiro agrupamento em comum.
        for idx in range(1,len(clusters)):
            clusters[idx] = es.relabel(base=clusters[0], array=clusters[idx])

        self.labels = es.voting(clusters) #votação entre os rótulos.
        return self;

