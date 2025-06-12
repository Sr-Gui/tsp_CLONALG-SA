import numpy as np
import tsplib95
import pandas as pd
import matplotlib.pyplot as plt
import time

#TSP
#tsp=tsplib95.load('TSP/att48.tsp')


def distance(tsp):
  node=list(tsp.get_nodes())
  n=len(node)

  dist=np.zeros((n,n),dtype=int)
  for i in range(n):
    for j in range(n):
      if i!=j:
        dist[i][j]=tsp.get_weight(i+1,j+1)
  return dist

def cost(sol,dist):
  c=0
  n=len(sol)
  for i in range(n-1):
    c+=dist[sol[i]][sol[(i+1)]]
  c += dist[sol[-1]][sol[0]]
  return c

#2-opt
def tweak(sol):
  n=len(sol)
  i,j=sorted(np.random.choice(n,2,replace=False))
  R=sol.copy()
  R[i:j+1]=sol[i:j+1][::-1]
  return R

def simulatedAnnealing(tsp,init_sol=None,t0=10000,t1=0.001,a=0.9999):
  dist = distance(tsp)
  n=len(dist)
  
  if init_sol==None:
    S=np.arange(n)
  else:
    S=init_sol.copy()

  best=S.copy()
  
  T=t0
  i=0
  start=time.time()

  while(t0>t1):
    R=tweak(S)
    d=cost(R,dist)-cost(S,dist)
    if (d<0) or (np.random.rand()<np.exp(-d/t0)):
      S=R
      if cost(S,dist)<cost(best,dist):
        best=S
    t0*=a #resfriamento/decrease
    i+=1
    # print(best,cost(best),t)
    end=time.time()
    time_f=end-start
  return best,T,t1,a,i,time_f,cost(best,dist)

#plot
def results(best,c,t0,t1,a,i,time_f):
  df=pd.DataFrame({
    'Parametros':['T0','T1','a','i','Custo','Tempo'],
    'Value':[t0,t1,a,i,c,time_f]
    })
  print('Resultados')
  print('Melhor caminho: ',best)
  print(df.to_markdown(index=False))

def res_rota(tsp,sol):
  coord=tsp.node_coords
  x=[coord[i+1][0] for i in sol]+[coord[sol[0]+1][0]]
  y=[coord[i+1][1] for i in sol]+[coord[sol[0]+1][1]]

  plt.figure()
  plt.plot(x,y)
  for i,(xi,yi) in enumerate(zip(x,y)):
    plt.text(xi,yi,str(sol[i%len(sol)]+1),fontsize=8)
  plt.grid(True)
  plt.show()
  
#best,t0,t1,a,i,time_f,cost_sa = simulatedAnnealing(tsp)
#results(best,cost_sa,t0,t1,a,i,time_f)
#res_rota(tsp,best)

#Sol otima estimada(pseudo-euclidiana): 10628