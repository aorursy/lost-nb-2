#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./linkern ]]; then\n  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz\n  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c\n  tar xf co031219.tgz\n  (cd concorde && CFLAGS='-O3 -march=native -mtune=native -fPIC' ./configure)\n  (cd concorde/LINKERN && make -j && cp linkern ../../)\n  rm -rf concorde co031219.tgz\nfi")


# In[ ]:


cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])


# In[ ]:


cities1k = cities * 1000


# In[ ]:


def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index+1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities1k, 'cities1k.tsp')


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'time ./linkern -s 42 -S linkern.tour -R 1000000000 -t 18000 ./cities1k.tsp >linkern.log')


# In[ ]:


get_ipython().system("sed -Ene 's/([0-9]+) Steps.*Best: ([0-9]+).*/\\1,\\2/p' linkern.log >linkern.csv")
pd.read_csv('linkern.csv', index_col=0, names=['TSP tour length']).plot();


# In[ ]:


def read_tour(filename):
    tour = open(filename).read().split()[1:]
    tour = list(map(int, tour))
    if tour[-1] == 0: tour.pop()
    return tour

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    primes = list(sympy.primerange(0, len(cities)))
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

def write_submission(tour, filename):
    assert set(tour) == set(range(len(tour)))
    pd.DataFrame({'Path': list(tour) + [0]}).to_csv(filename, index=False)


# In[ ]:


tour = read_tour('linkern.tour')
write_submission(tour, 'submission.csv')


# In[ ]:


score_tour(tour)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20, 20))
plt.plot(cities.X[tour], cities.Y[tour], alpha=0.7)
plt.show()

