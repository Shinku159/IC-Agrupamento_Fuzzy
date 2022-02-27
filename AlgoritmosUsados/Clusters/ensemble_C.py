import ensembles as es
import numpy as np
import random

from sklearn.cluster import AgglomerativeClustering as Ac
from seaborn import scatterplot as scatter
from sklearn.cluster import MeanShift
from sklearn import datasets as Datas
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics as sm
#--------------------------------------------------------------------------------------------------------

def EnsembleC(data, k, maxRuns, base=False):
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
        númericos contendo o valor do atributo de cada dado a ser agrupado
        (base de dados).

    k: inteiro
        Número de divisões a serem feitas pelos métodos de agrupamento.

    maxRuns: inteiro,
        Número máximo de interações para gerar rótulos de agrupamento entre
        os métodos, de forma a fazer o ensemble. essa variavel é ignorada
        caso 'base' seja verdadeiro.

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
    clusters = [] #mantem todos os agrupamentos para o ensemble
    if(base):
        #ALGORITMOS BASE------------------------------
        #meanshift
        MScluster = MeanShift()
        MScluster.fit(data)

        #kmeans
        KMcluster = KMeans(n_clusters=k)
        KMcluster.fit(data)

        #aglomerativo
        AGcluster = Ac(n_clusters=k, linkage='ward') #método de escolha de linkage em desenvolvimento
        AGcluster.fit(data)
        #----------------------------------------------

        #ENSEMBLE--------------------------------------
        clusters.append(list(KMcluster.labels_)) 
        clusters.append(list(AGcluster.labels_))
        if(len(set(MScluster.labels_)) == k): #adiciona o meanshift apenas caso tenha o mesmo número de rótulos que o k fornecido
            clusters.append(list(MScluster.labels_)) 
        #----------------------------------------------
    else:
        #ENSEMBLE--------------------------------------
        linkages = ['complete','average','single', 'ward'] #ligações aceitas pelo algoritmo aglomerativo
        affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] #tipos de calculos de distância aceitas pelo algortimo aglomerativo
        for run in range(maxRuns):
            #kmédias randomizado
            KMcluster = KMeans(n_clusters=k, init='random' ,n_init=1)
            KMcluster.fit(data)
            clusters.append(list(KMcluster.labels_))

            #para a versão usando a ligação ward, apenas a métrica euclidiana funciona.
            aux = random.randrange(0, 4)
            linkage = linkages[aux]
            affinity = 'euclidean'
            if(aux != 3):
                affinity = affinities[random.randrange(0, 5)]

            #Aglomerativo randomizado
            Agcluster = Ac(n_clusters=k, linkage=linkage, affinity=affinity)
            Agcluster.fit(data)
            clusters.append(list(Agcluster.labels_))

            #MeanShift randomizado
            MScluster = MeanShift(bandwidth=round(random.uniform(0.1,100.1), 2))
            MScluster.fit(data)
            if(len(set(MScluster.labels_)) == k): #adiciona o meanshift apenas caso tenha o mesmo número de rótulos que o k fornecido
                clusters.append(list(Agcluster.labels_))
        #----------------------------------------------
        
        #faz a re-rotulação dos agrupamentos de acordo com um primeiro agrupamento em comum.
        for idx in range(1,len(clusters)):
            clusters[idx] = es.relabel(base=clusters[0], array=clusters[idx])

        labels = es.voting(clusters) #votação entre os rótulos.
        return labels; 