import numpy as np
import copy
import sys

from munkres import Munkres
#--------------------------------------------------------------------------------------------------------

''' ATENÇÃO!!!!!!
FUNÇÕES DE AUXÍLIO PARA OS ENSEMBLES
'''
def relabel(base, array):
    set1 = set(base) #pega apenas os nomes dos clusters usados nos r[otulos do agrupamento base.
    u1 = list(set1) 

    set2 = set(array) #pega apenas os nomes dos clusters usados nos r[otulos do agrupamento que vamos trocar.
    u2 = list(set2)

    matrix = [[0 for i in range(len(u2))]for j in range(len(u1))] #Matriz de comparação dos rótulos, |base|x|array|.

    for i in range(len(base)):
        item_1 = u1.index(base[i]) #adiciona 1 ao indice da tabela que corresponde a classificação do ponto i, tanto na base quanto no array.
        item_2 = u2.index(array[i]) #(EX: se o ponto i é do cluster 1 na base e do cluster 3 no array, adiciona +1 na posição [1][3])

        matrix[item_1][item_2] = matrix[item_1][item_2] + 1 #as maiores correspondencias entre os clusters demonstram qual cluster é parido com qual na base e no array.

    #Cria uma matriz custo, dando custo menor para os valores mais altos na matrix de correspondência. 
    #(Para encontrarmos a solução ótima do problema dos trabalhadores usando a biblioteca do Munkres)
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row = cost_row + [(sys.maxsize - col)]
        cost_matrix = cost_matrix + [cost_row]

    m = Munkres() #resolve o conhecido problema da ordem dos trabalhadores que pode ser aplicado com os rótulos dos clusters.
    indexes = m.compute(cost_matrix)
    replaced = copy.deepcopy(array) #replaced vai ser nosso novo array com os rótulos correspondendo a base.

    for row, col in indexes:
        # u1[row] e u2[col] são correspondentes
        for idx in range(len(array)):
            # se o elemento do array é igual u2[col]
            if array[idx] == u2[col]:
                #o elemento corresponde ao elemento u1[row]
                replaced[idx] = u1[row]

    return replaced

#--------------------------------------------------------------------------------------------------------

def relabelFuzzy(base, array, M):
    set1 = set(base) #pega apenas os nomes dos clusters usados nos r[otulos do agrupamento base.
    u1 = list(set1) 

    set2 = set(array) #pega apenas os nomes dos clusters usados nos r[otulos do agrupamento que vamos trocar.
    u2 = list(set2)

    matrix = [[0 for i in range(len(u2))]for j in range(len(u1))] #Matriz de comparação dos rótulos, |base|x|array|.

    for i in range(len(base)):
        item_1 = u1.index(base[i]) #adiciona 1 ao indice da tabela que corresponde a classificação do ponto i, tanto na base quanto no array.
        item_2 = u2.index(array[i]) #(EX: se o ponto i é do cluster 1 na base e do cluster 3 no array, adiciona +1 na posição [1][3])

        matrix[item_1][item_2] = matrix[item_1][item_2] + 1 #as maiores correspondencias entre os clusters demonstram qual cluster é parido com qual na base e no array.

    #Cria uma matriz custo, dando custo menor para os valores mais altos na matrix de correspondência. 
    #(Para encontrarmos a solução ótima do problema dos trabalhadores usando a biblioteca do Munkres)
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row = cost_row + [(sys.maxsize - col)]
        cost_matrix = cost_matrix + [cost_row]

    m = Munkres() #resolve o conhecido problema da ordem dos trabalhadores que pode ser aplicado com os rótulos dos clusters.
    indexes = m.compute(cost_matrix)
    replaced = copy.deepcopy(M) #replaced vai ser nosso novo array com os rótulos correspondendo a base.

    for row, col in indexes:
        replaced[u1[row]] = M[u2[col]]
        
    return replaced

#--------------------------------------------------------------------------------------------------------

def voting(clusters):
    #Fazemos a transposta da matriz para que cada linha seja o voto de cada algoritmo para o cluster a qual o ponto i deve pertencer.
    clusters = list(map(list, zip(*clusters)))

    voted = []
    for row in clusters:
        #para cada linha de votos(ponto i), list os apenas os cluster que receberam votos.
        u = list(set(row))
        
        #lista de votos.
        counter = [0 for i in u]

        #para cada voto, aumenta 1 no no cluster votado pelo algoritmo no counter(contador).
        for idx in range(len(u)):
            counter[idx] = row.count(u[idx])

        #encontre no counter, o cluster mais votado para o ponto i.
        max_idx = counter.index(max(counter))

        #Define o array com os rótulos definis pela votação a cada ponto.
        voted = voted + [u[max_idx]]

    return voted

#--------------------------------------------------------------------------------------------------------

def votingFuzzy(Memberships):
    ur = copy.deepcopy(Memberships[0])
    for j in range(len(Memberships[0])):
        for i in range(1, len(Memberships)):
            ur[j] += Memberships[i,j]

    return ur

#Créditos---------
#Kyle Yang
#GitHub: Kr4t0n
#link do original: https://github.com/Kr4t0n/Ensemble-Clustering
#-----------------