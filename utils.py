import pandas as pd
import networkx as nx

def load_network_data(network):
    # load edge list
    edgelist = pd.read_csv('./network_data/{}.txt'.format(network), delimiter=' ', header=None)
    edgelist = edgelist[[0,1]]
    edgelist = edgelist.rename(columns={0: 'source', 1: 'target'})
    edgelist = edgelist.drop_duplicates()
    
    # network object
    g = nx.from_pandas_edgelist(edgelist, source='source', target='target')

    # ectract the largest connected component
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    g = g.subgraph(gcc[0])
    
    return g
