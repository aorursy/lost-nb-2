#!/usr/bin/env python
# coding: utf-8

# In[1]:


#For later use
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#networkx graph computing library
import networkx as nx

#load the graph with nx.read_graphml
G = nx.read_graphml('../input/street-network-of-new-york-in-graphml/manhatten.graphml')


# In[2]:


nx.info(G)


# In[3]:


G_simple = nx.Graph(G)
nx.info(G_simple)


# In[4]:


from collections import Counter
degree_dic = Counter(G_simple.degree().values())

degree_hist = pd.DataFrame({"degree": list(degree_dic.values()),
                            "Number of Nodes": list(degree_dic.keys())})
plt.figure(figsize=(20,10))
sns.barplot(y = 'degree', x = 'Number of Nodes', 
              data = degree_hist, 
              color = 'darkblue')
plt.xlabel('Node Degree', fontsize=30)
plt.ylabel('Number of Nodes', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()


# In[5]:


ins = list((G.in_degree()).values())
outs = list((G.out_degree()).values())
degrees = pd.DataFrame({"in-degree": ins, "out-degree": outs})

fig = sns.jointplot(x="in-degree",y="out-degree",data=degrees,kind="kde", color = 'darkblue',size=8)


# In[6]:


nx.density(G)


# In[7]:


#import osmnx 
#ox.plot_graph(G,fig_height= 12, node_size=10, node_zorder=2, node_color = '#808080')


# In[8]:


nx.draw(G, pos=nx.spring_layout(G), node_size=0.01, width=0.1)


# In[9]:


# we cant not just access the nodes with G(0) orso, we must call them by their id
# G.nodes() returns a list of all node ids, e.g., '42459137'

G[G.nodes()[1]]


# In[10]:


nx.diameter(G_simple)


# In[11]:


nx.average_shortest_path_length(G_simple)


# In[12]:


from collections import Counter
import collections
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

in_degrees  = G.in_degree() 
in_h = Counter(in_degrees.values())
in_dic = collections.OrderedDict(sorted(in_h.items()))
in_hist = list(in_dic.values())
in_values =list(in_dic.keys())

out_degrees  = G.out_degree() 
out_h =  Counter(out_degrees.values())
out_dic = collections.OrderedDict(sorted(out_h.items()))
out_hist = list(out_dic.values())
out_values =list(out_dic.keys())

mu = 2.17
sigma = sp.sqrt(mu)
mu_plus_sigma = mu + sigma
x = range(0,10)
prob = stats.poisson.pmf(x, mu)*4426

plt.figure(figsize=(12, 8)) 
plt.grid(True)
plt.loglog(out_values,out_hist,'ro-')  # in-degree
plt.loglog(in_values,in_hist,'bv-')  # in-degree
plt.plot(x, prob, "o-", color="black")
plt.legend(['In-degree','Out-degree','Poission'])
plt.xlabel('Degree')
plt.ylabel('Number of  nodes')
plt.title('Manhatten Street Network')
plt.xlim([0,2*10**2])
plt.show()


# In[13]:


#create two simple graphs from our original directed graph
G_simple = nx.Graph(G)
G_simple2 = nx.Graph(G)


nx.node_connectivity(G_simple)


# In[14]:


nx.algebraic_connectivity(G_simple)


# In[15]:


#compute the betweeness centrality on one of the simple graphs, this can take a while
between =  nx.betweenness_centrality(G_simple)


# In[16]:


#G_projected = ox.project_graph(G)
#max_node, max_bc = max(extended_stats['betweenness_centrality'].items(), key=lambda x: x[1])
#max_node, max_bc


# In[17]:


G['42431099']


# In[18]:


'''
import operator
from random import shuffle
from random import randrange
from random import randint
import random
import matplotlib.ticker as mtick

sorted_x = sorted(between.items(), key=operator.itemgetter(1), reverse=True)
rand_x = list(range(0,4426 ))

random.shuffle(rand_x)
between_giant = []
between_rand = []
avg_degs = []

for x in range(3000):
 
        remove = sorted_x[x]      
        remove2 = sorted_x[rand_x[x]]
        G_simple.remove_nodes_from(remove)
        G_simple2.remove_nodes_from(remove2)
             
        giant = len(max(nx.connected_component_subgraphs(G_simple), key=len))
        giant2 = len(max(nx.connected_component_subgraphs(G_simple2), key=len))

        between_giant.append(giant)
        between_rand.append(giant2)

y1 = between_giant
y2 = between_giant

y1= y1[ :-1]
y2= y2[1: ]

perc = np.linspace(0,100,len(between_giant))
fig = plt.figure(1, (12,8))
ax = fig.add_subplot(1,1,1)

ax.plot(perc, between_giant)
ax.plot(perc, between_rand)

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)
ax.set_xlabel('Fraction of Nodes Removed')
ax.set_ylabel('Giant Component Size')
ax.legend(['betweenness','random'])
plt.show()
'''


# In[19]:


#taken from. https://github.com/gboeing/osmnx
def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):

    phi1 = np.deg2rad(90 - lat1)

    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + np.cos(phi1) * np.cos(phi2))
    arc = np.arccos(cos)

    distance = arc * earth_radius
   
    return distance


def get_nearest_node(G, point, return_dist=False):

    coords = np.array([[node, data['x'], data['y']] for node, data in G.nodes(data=True)])
    df = pd.DataFrame(coords, columns=['node', 'x', 'y']).set_index('node')
    df['reference_y'] = point[0]
    df['reference_x'] = point[1]

    distances = great_circle_vec(lat1=df['reference_y'],
                                 lng1=df['reference_x'],
                                 lat2=df['x'].astype('float'),
                                 lng2=df['y'].astype('float'))
  
    nearest_node = int(distances.idxmin())
  
    if return_dist:
        return nearest_node, distances.loc[nearest_node]
    else:
        return nearest_node


# In[20]:


#load the training data
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')


#go through the dataset and calculate the shortest path
for index, row in train[24:25].iterrows():

    pick_point = ( row['pickup_longitude'],row['pickup_latitude'])
    drop_point = ( row['dropoff_longitude'],row['dropoff_latitude'])
    
    pick_node = get_nearest_node(G, pick_point)
    drop_node = get_nearest_node(G, drop_point)
   
    try:
        route = nx.shortest_path(G, str(pick_node), str(drop_node))
        #plot the shortest path on the graph
        #fig, ax = ox.plot_graph_route(G, route,fig_height=15, node_size=1)
        print("Shortest Path:")
        print(route)
        
    except:
        print("Some Error")
        #handle error
        pass
    
    #the corresponding node betweenness scores for each edge in the shortest path
    print("Betweenness Centrality for each node on the path")
    node_bet = []
    for edge in route:
        node_bet.append(between[edge])
    print(node_bet)

