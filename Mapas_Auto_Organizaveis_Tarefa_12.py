"""
Redes Neurais - Mapas Auto Organizáveis
Aprendizagem não supervisionada
Detecção de características e agrupamentos
Base de dados Breast Cancer
Classificação e agrupamento dos tumores maligno e benigno

"""


from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


""" Carregar nas variáveis os atributos da base de dados """
base = pd.read_csv('Entradas_Breast_cancer.csv')
X = base.iloc[:, 0:30].values
base2 = pd.read_csv('Saidas_classes.csv')
y = base2.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)

""" Contrução e configuração do mapa auto organizável """
# tamanho = 5*sqrt(569) = 119,26. Logo: matriz 11x11
som = MiniSom(x=11, y=11, input_len=30, sigma=3.0, learning_rate=0.5, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=1000)


""" Visualização do mapa """
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
# MID - mean inter neuron distance
# T é para matriz transposta
pcolor(som.distance_map().T)
colorbar()
# Criação dos marcadores
markers = ['o', 's']
# criação das cores
color = ['r', 'g']


""" Adicionar ao mapa a classificação com os marcadores """
for i, x in enumerate(X):
    w = som.winner(x)
    # plot(posição x do marcador para um quadradinho da matriz, posição y do marcador, variável com as classe (0 a 177),
    # cor da fonte, tamanho do marcador, preenchimento da cor dos simbolos, configuração da borda)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)

plt.show()



print('Fim')