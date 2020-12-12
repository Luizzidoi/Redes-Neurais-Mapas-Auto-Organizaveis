"""
Redes Neurais - Mapas Auto Organizáveis
Aprendizagem não supervisionada
Detecção de características e agrupamentos
Base de dados Bart e Homer
Classificação e agrupamento de personagens
Identificar outliers em uma base de dados para classificação de personagens

"""


from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


""" Carregar nas variáveis os atributos da base de dados """
base = pd.read_csv('personagens.csv')
X = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)


""" Construção do mata auto organizável """
# tamanho do som = 5*sqrt(293) = 85,58
som = MiniSom(x=9, y=9, input_len=6, sigma=2.0, learning_rate=0.5, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=500)

# Transformação das classes em números para associarmos com os markers e colors abaixo
y[y == 'Bart'] = 0
y[y == 'Homer'] = 1


""" Visualização do mapa """
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
# MID - mean inter neuron distance -> traz quanto parecido um neuronio é de seus vizinhos
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
color = ['r', 'g']


""" Adicionar ao mapa a classificação com os marcadores """
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)

plt.show()


########################################################################################################################
""" 
Detecção de fraudes -> outliers
Buscar no mapa se os registros em amarelo (maior MID) são clientes suspeitos
"""

""" Buscar quais registros estão asssociados a cada um dos neuronios """
mapeamento = som.win_map(X)
# Concatenar dois neuronios escolhidos no mapa (que estão em cor amarela - suspeitos)
# Axix = 0  para concatenação um abaixo do outro
suspeitos = mapeamento[(3, 1)]
suspeitos = normalizador.inverse_transform(suspeitos)


""" Buscar as classes de cada um dos suspeitos para avaliação """
classe = []
# i percorre todos os registros da base (293 registrios)
# j percorre todos os registros da lista de suspeitos
for i in range(len(base)):
    for j in range(len(suspeitos)):
        # procurar se o id encontra na base é igual ao encontrado na lista dos suspeitos
        if ((base.iloc[i, 0] == suspeitos[j, 0]) and (base.iloc[i, 1] == suspeitos[j, 1]) and
            (base.iloc[i, 2] == suspeitos[j, 2]) and (base.iloc[i, 3] == suspeitos[j, 3]) and
            (base.iloc[i, 4] == suspeitos[j, 4]) and (base.iloc[i, 5] == suspeitos[j, 5])):
                classe.append(base.iloc[i, 6])


classe = np.asarray(classe)
# Concatenar os dados dos suspeitos com as suas respectivasa classe
suspeitos_final = np.column_stack((suspeitos, classe))
# Ordenação dos dados para melhor visualização (conforme às clases)
suspeitos_final = suspeitos_final[suspeitos_final[:, 6].argsort()]



print('Fim')