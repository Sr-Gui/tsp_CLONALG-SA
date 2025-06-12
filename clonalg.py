import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tsplib95
import time
from copy import deepcopy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#tsp = tsplib95.load('TSP/att48.tsp')

def distance(tsp):
  node=list(tsp.get_nodes())
  populacao=len(node)

  dist=np.zeros((populacao,populacao),dtype=int)
  for i in range(populacao):
    for j in range(populacao):
      if i!=j:
        dist[i][j]=tsp.get_weight(i+1,j+1)
  return dist

def cost(sol,dist):
  populacao=len(sol)
  c=0
  for i in range(populacao-1):
    c+=dist[sol[i]][sol[(i+1)]]
  c += dist[sol[-1]][sol[0]]
  return c

def init_pop(size,n):
  return [random.sample(range(n),n) for _ in range(size)]

def normaliza(afinidades):
  s=sum(afinidades)
  n= [a/s for a in afinidades]
  return n

def avalia(populacao,dist):
  a=[1/cost(i,dist) for i in populacao]
  return a

def seleciona(populacao, afinidade, nSelecao):
  i=np.argsort(afinidade)[-nSelecao:]
  return [populacao[k] for k in i]

def clona(sel,afinidade,beta):
  clones = []
  for i,k in enumerate(sel):
    nClones= int(beta*afinidade[i]*len(sel))+1
    clones += [deepcopy(k) for _ in range(nClones)]
  return clones

def hipermutacao(clones, afins_norm, rho,n):
    for i, clone in enumerate(clones):
        p_mut = rho * (1 - afins_norm[i % len(afins_norm)])
        if random.random() < p_mut:
            a, b = random.sample(range(n), 2)
            clone[a], clone[b] = clone[b], clone[a]
    return clones

def substituicao(pop, afins, clones, afins_clones, n_sel, n_rand,n):
    elite_idx = np.argsort(afins_clones)[-n_sel:]
    novos = [clones[i] for i in elite_idx]
    novos += init_pop(n_rand,n)
    restante = len(pop) - len(novos)
    best_idx = np.argsort(afins)[-restante:]
    novos += [pop[i] for i in best_idx]
    return novos

def clonalg(tsp,beta=10,rho=0.75,tam_pop=100,nAleatorio=5,nSelecao=10,E=100):
  dist=distance(tsp)
  n=len(dist)
  p=init_pop(tam_pop,n)
  a=avalia(p,dist)
  best = p[np.argmax(a)]
  best_custo = cost(best,dist)

  start = time.time()

  for _ in range(E):
    afinidade = normaliza(a)
    selecionados = seleciona(p,a,nSelecao)
    clones = clona(selecionados,afinidade,beta)
    
    afinidade_clones = avalia(clones,dist)
    afinidade_clones = normaliza(afinidade_clones)
    clones=hipermutacao(clones,afinidade_clones,nSelecao,n)
    
    p2=substituicao(p,a,clones,afinidade_clones,nSelecao,nAleatorio,n)
    afinidade = avalia(p2,dist)
    best2=p2[np.argmax(afinidade)]
    best2_custo=cost(best2,dist)

    if best2_custo < best_custo:
      best,best_custo = best2,best2_custo
  end = time.time()
  return best,best_custo,end-start

def results(best, c, t_exec):
  df = pd.DataFrame({
    'Parametros': ['Custo', 'Tempo (s)'],
    'Valor': [c, t_exec]
  })
  print('Resultados')
  print('Melhor caminho: ', best)
  print(df.to_markdown(index=False))

def res_rota(tsp, sol):
    coord = tsp.node_coords
    x = [coord[i + 1][0] for i in sol] + [coord[sol[0] + 1][0]]
    y = [coord[i + 1][1] for i in sol] + [coord[sol[0] + 1][1]]
    plt.figure()
    plt.plot(x, y, marker='o')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(sol[i % len(sol)] + 1), fontsize=8)
    plt.grid(True)
    plt.title("Melhor Rota CLONALG")
    plt.show()

#best, custo, tempo = clonalg(tsp)
#results(best, custo, tempo)
#res_rota(tsp, best)