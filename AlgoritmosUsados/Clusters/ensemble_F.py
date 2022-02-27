import ensembles as es
import numpy as np
import random
import copy
import math
import sys

from seaborn import scatterplot as scatter
from sklearn import datasets as Datas
from matplotlib import pyplot as plt
from sklearn import metrics as sm
from tabulate import tabulate
from munkres import Munkres
from skfuzzy import cmeans
from PcmAlg import pcm
from GkAlg import GK
#--------------------------------------------------------------------------------------------------------

def MembershipNorm(u):
    """Normalizador de pertinência

    Função que assume valores de 0 a 1 para toda a matriz de pertinência
    de um dado agrupamento enviado, a soma de todos os valores ao final
    fica resolvida em 1

    Parâmetros
    ----------
    u: 2D Array
        matriz de pertinência gerada por um agrupamento fuzzy

    Atributos
    ---------
    u: 2D Arrar
        Matriz de pertinência normalizada
    """

    #para a pertinência de cada cluster, normalize para valores entre 0 e 1 dividindo pela soma de todos os fatores
    for i in range(len(u[0])):
        c = np.sum(u[:,i]) 
        for j in range(len(u)):
            u[j,i] = u[j,i]/c
    return u


def EnsembleF(data, k, maxRuns, base=False):
    """Tática de ensemble via votação de métodos Fuzzy

    Esse algoritmo é desenvolvido de maneira a randomificar possiveis
    resultados para o agrupamento de uma base X nos modelos fuzzy do
    método possibilistico do c-médias, o c-médias e a otimização de gustafson kessel.

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
        Realiza um ensemble básico apenas com a 'melhor' resposta de cada 
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
    clusters = []
    alldata = np.array(data).T
    #ALGORITMOS BASE------------------------------
    #fuzzy c-means
    cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=k, m=2, error=0.0001, maxiter=1000)
    FCMlabels = np.argmax(u, axis=0)

    #possibilistic c-means
    v, v0, u2, u02, d2, t = pcm(alldata, c=k, m=2, e=0.0001, max_iterations=1000)
    u2 = MembershipNorm(u2)
    PCMlabels = np.argmax(u2, axis=0)
    
    #gustafson kessel
    GKcluster = GK(n_clusters=k)
    GKcluster.fit(data)
    GKlabels = np.argmax(GKcluster.u, axis=0)

    Memberships = [u]
    clusters.append(list(FCMlabels))

    if(len(set(GKlabels)) == k):
        Memberships = np.concatenate((Memberships, [GKcluster.u]))
        clusters.append(list(GKlabels))
    if(len(set(PCMlabels)) == k):
        Memberships = np.concatenate((Memberships, [u2]))
        clusters.append(list(PCMlabels))
    #----------------------------------------------
    if(not base):
        for run in range(maxRuns-1):
            #fcm randomizado
            cntr, u, u0, d, jm, p, fpc = cmeans(data=alldata, c=k, m=round(random.uniform(1.1,10.1), 1), error=0.0001, maxiter=1000)
            FCMlabels = np.argmax(u, axis=0)

            v, v0, u, u0, d, t = pcm(alldata, c=k, m=round(random.uniform(1.1,10.1), 1), e=0.001, max_iterations=1000) 
            u2 = MembershipNorm(u2)
            PCMlabels = np.argmax(u2, axis=0)

            GKcluster = GK(n_clusters=k, max_iter=1000, m=round(random.uniform(1.1,10.1), 1), error=0.0001)
            GKcluster.fit(data)
            GKlabels = np.argmax(GKcluster.u, axis=0)

            Memberships = np.concatenate((Memberships, [u]))
            clusters.append(list(FCMlabels))

            if(len(set(GKlabels)) == k and not math.isnan(GKcluster.u[0,0])):
                Memberships = np.concatenate((Memberships, [GKcluster.u]))
                clusters.append(list(GKlabels))
            if(len(set(PCMlabels)) == k and not math.isnan(u2[0,0])):
                Memberships = np.concatenate((Memberships, [u2]))
                clusters.append(list(PCMlabels))

    for idx in range(1, len(clusters)):
        Memberships[idx] = es.relabelFuzzy(base=clusters[0], array=clusters[idx], M=Memberships[idx])

    ur = es.votingFuzzy(Memberships)
    labels = np.argmax(ur, axis=0)
    # u = MembershipNorm(u)
    return labels


