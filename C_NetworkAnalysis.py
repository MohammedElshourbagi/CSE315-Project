import pandas as pd
import networkx as nx
import osmnx as ox
import matplotlib as plt

from main import RoadNetwork_Buses
NETWORK = nx.Graph(RoadNetwork_Buses)

# Adding Urban Reach values for each Node
UrbanReach = pd.read_csv(r"results/UrbanReach.csv")
nx.set_node_attributes(NETWORK, UrbanReach.set_index("NODE").to_dict()["25"], "25")
nx.set_node_attributes(NETWORK, UrbanReach.set_index("NODE").to_dict()["20"], "20")
nx.set_node_attributes(NETWORK, UrbanReach.set_index("NODE").to_dict()["15"], "15")
nx.set_node_attributes(NETWORK, UrbanReach.set_index("NODE").to_dict()["10"], "10")
nx.set_node_attributes(NETWORK, UrbanReach.set_index("NODE").to_dict()["5"], "5")

# CENTRALITY ALGORITHMS
RESULTS = pd.DataFrame(index=list(NETWORK.nodes()))
RESULTS["DegreeCentrality"] = pd.Series(nx.degree_centrality(NETWORK))
RESULTS["EigenVectorCentrality"] = pd.Series(nx.eigenvector_centrality(NETWORK, max_iter=10000, weight="DeltaReach"))
RESULTS["KatzCentrality"] = pd.Series(nx.katz_centrality(NETWORK, max_iter=10000, weight="DeltaReach"))
RESULTS["BetweennessCentrality"] = pd.Series(nx.betweenness_centrality(NETWORK, weight="DeltaReach"))
RESULTS["ClosenessCentrality"] = pd.Series(nx.closeness_centrality(NETWORK, distance="DeltaReach"))
RESULTS["FlowClosenessCentrality"] = pd.Series(nx.current_flow_closeness_centrality(NETWORK, weight="DeltaReach"))
RESULTS["HarmonicCentrality"] = pd.Series(nx.harmonic_centrality(NETWORK, distance="DeltaReach"))
# RESULTS["LaplacianCentrality"] = pd.Series(nx.laplacian_centrality(NETWORK, weight="DeltaReach"))  # TAKES ALOT OF TIME
RESULTS.to_csv(r"Results.csv")

# COMMUNITY ALGORITHMS
"""
communities = sorted(nx.community.louvain_communities(NETWORK, weight="DeltaReach", threshold=0.5), key=len, reverse=True)
for c, v_c in enumerate(communities):
    for v in v_c:
        # Add 1 to save 0 for external edges
        NETWORK.nodes[v]['community'] = c + 1
for v, w, in NETWORK.edges:
    if NETWORK.nodes[v]['community'] == NETWORK.nodes[w]['community']:
        # Internal edge, mark with community
        NETWORK.edges[v, w]['community'] = NETWORK.nodes[v]['community']
    else:
        # External edge, mark as 0
        NETWORK.edges[v, w]['community'] = 0

N_coms = len(communities)                            # Amount of communities
# print(f"The club has {N_coms} communities.")

edges_coms = []                                      # edge list for each community
coms_G = [nx.Graph() for _ in range(N_coms)]         # community graphs
CentralNodesForEachCommunity = {}
for i in range(N_coms):
    edges_coms.append([(u,v,d) for u,v,d in NETWORK.edges(data=True) if d['community'] == i+1])
    coms_G[i].add_edges_from(edges_coms[i])  # add edges

    CentralityInCommunity = nx.current_flow_closeness_centrality(coms_G[i], weight="DeltaReach")
    CentralNodeInCommunity = sorted(CentralityInCommunity.items(), key=lambda x:x[1])[-1]
    CentralNodesForEachCommunity[CentralNodeInCommunity[0]] = CentralNodeInCommunity[1]

print(CentralNodesForEachCommunity)
"""