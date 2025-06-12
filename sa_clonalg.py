import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tsplib95
import time
from copy import deepcopy

from clonalg import clonalg,cost
from simulatedAnnealing import simulatedAnnealing

tsp = tsplib95.load('TSP/att532.tsp')

def res_rota(tsp, sol):
    coord = tsp.node_coords
    x = [coord[i + 1][0] for i in sol] + [coord[sol[0] + 1][0]]
    y = [coord[i + 1][1] for i in sol] + [coord[sol[0] + 1][1]]
    plt.figure()
    plt.plot(x, y, marker='o')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(sol[i % len(sol)] + 1), fontsize=8)
    plt.grid(True)
    plt.title("Melhor Rota CLONALG + SA")
    plt.show()

def results(best, custo, tempo, algoritmo):
    df = pd.DataFrame({
        'Parametros': ['Custo', 'Tempo (s)'],
        'Valor': [custo, tempo]
    })
    print(f'Resultados - {algoritmo}')
    print('Melhor caminho: ', best)
    print(df.to_markdown(index=False))

#clonalg
best_ca,cost_ca,time_ca = clonalg(tsp)
#results(best_ca,cost_ca,time_ca,'CLONALG')
#res_rota(tsp,best_ca)

#sa
best_sa,_,_,_,i,time_sa,cost_sa = simulatedAnnealing(tsp,init_sol=best_ca)
#results(best_sa,cost_sa,time_sa,'SA')
res_rota(tsp,best_sa)
print('TEMPO TOTAL = ',time_sa+time_ca)

print("\nResultados Finais - CLONALG + SA")
print("Melhor Caminho:", best_sa)
df = pd.DataFrame({
    'Parâmetro': ['Custo Clonalg','Custo Clonalg+SA', 'Tempo CLONALG (s)', 'Tempo SA (s)', 'Tempo Total (s)', 'Iterações SA'],
    'Valor': [cost_ca,cost_sa, time_ca, time_sa, time_ca + time_sa, i]
})
print(df.to_markdown(index=False))