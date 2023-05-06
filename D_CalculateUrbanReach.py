import pandas as pd
import geopandas as gpd
import shapely.ops
from shapely.geometry import Point, LineString
import networkx as nx
import osmnx as ox
from main import RoadNetwork_Walkable, RoadNetwork_Buses, BuildingCentroids, trip_times

NETWORK = nx.Graph(RoadNetwork_Buses)

# Adding a change of urban reach for each edge
nx.set_edge_attributes(NETWORK, 0, "DeltaReach")
minDelta, maxDelta = 0, 0
for u, v in NETWORK.edges():
    delta = int(NETWORK.nodes()[u]["25"][:-2]) - int(NETWORK.nodes()[v]["25"][:-2])
    minDelta = min(minDelta, delta)
    maxDelta = max(maxDelta, delta)
for u, v in NETWORK.edges():
    delta = int(NETWORK.nodes()[u]["25"][:-2]) - int(NETWORK.nodes()[v]["25"][:-2])
    normalizedDelta = ((delta - maxDelta) / (minDelta - maxDelta))
    NETWORK.edges[u, v].update(DeltaReach=normalizedDelta)

# COMMUNITY + CENTRALITY
communities = sorted(nx.community.louvain_communities(NETWORK, weight="DeltaReach", threshold=0.1), key=len, reverse=True)
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
N_coms = len(communities)
print(f"The Network has {N_coms} communities.")
edges_coms = []  # edge list for each community
coms_G = [nx.Graph() for _ in range(N_coms)]  # community graphs

CentralNodesForEachCommunity = {}
for i in range(N_coms):
    edges_coms.append([(u, v, d) for u, v, d in NETWORK.edges(data=True) if d['community'] == i+1])
    coms_G[i].add_edges_from(edges_coms[i])  # add edges

    CommunityCentrality = nx.group_degree_centrality(NETWORK, coms_G[i].nodes())
    CentralityInCommunity = nx.current_flow_closeness_centrality(coms_G[i], weight="DeltaReach")
    CentralNodeInCommunity = sorted(CentralityInCommunity.items(), key=lambda x: x[1])[-1]
    CentralNodesForEachCommunity[CentralNodeInCommunity[0]] = CentralNodeInCommunity[1]

# Can't use precalculated Urban reach, comes with the problem of over-counting the buildings
def make_iso_polys(G, AtNode, edge_buff=0.001, node_buff=0.001, infill=False):
    isochrone_polys = []
    for trip_time in sorted(trip_times, reverse=True):
        subgraph = nx.ego_graph(G, n=AtNode, radius=trip_time)
        node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
        nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, geometry=node_points)
        nodes_gdf = nodes_gdf.set_index('id')

        edge_lines = []
        for n_fr, n_to in subgraph.edges():
            f = nodes_gdf.loc[n_fr].geometry
            t = nodes_gdf.loc[n_to].geometry
            edge_lookup = G.get_edge_data(n_fr, n_to)[0].get('geometry', LineString([f, t]))
            edge_lines.append(edge_lookup)

        n = nodes_gdf.buffer(node_buff).geometry
        e = gpd.GeoSeries(edge_lines).buffer(edge_buff).geometry
        all_gs = list(n) + list(e)
        new_iso = gpd.GeoSeries(all_gs).unary_union

        isochrone_polys.append(new_iso)
    IsochronesDF = pd.DataFrame(isochrone_polys)
    IsochronesGDF = gpd.GeoDataFrame(IsochronesDF, geometry=IsochronesDF.columns[0], crs='epsg:4326')
    IsochronesGDF["TripTimes"] = trip_times
    return IsochronesGDF

NODES = list(CentralNodesForEachCommunity.keys())
IsochronePolys = gpd.GeoDataFrame(columns=trip_times)
for Node in NODES:
    NODE = RoadNetwork_Buses.nodes()[Node]
    LookAtNode = ox.nearest_nodes(RoadNetwork_Walkable, X=NODE['x'], Y=NODE['y'])
    IsochronePolys.loc[len(IsochronePolys.index)] = list(make_iso_polys(RoadNetwork_Walkable, AtNode=LookAtNode, infill=True).loc[:, 0])

MergedPolys = []
for i in range(len(trip_times)):
    MergedPoly = shapely.MultiPolygon()
    for j in range(len(NODES)-1):
        MergedPoly = shapely.ops.unary_union([MergedPoly, IsochronePolys.iloc[j, i], IsochronePolys.iloc[j + 1, i]])
    MergedPolys.append(MergedPoly)

CombinedPolys = gpd.GeoDataFrame(pd.DataFrame(), geometry=gpd.GeoSeries(MergedPolys), crs='epsg:4326')
CombinedPolys["TripTimes"] = trip_times
CombinedPolys.to_file(r"results/UrbanReachOfStations.gpkg", driver="GPKG")

# Finding the building count within the Isochrone
RESULTS = pd.DataFrame(columns=trip_times)  # Empty Dataframe used to store results
Result = gpd.sjoin(left_df=BuildingCentroids,
                   right_df=CombinedPolys
                   ).groupby("TripTimes")['point_id'].count().rename('Count').reset_index()

RESULTS.loc[0] = pd.Series(list(reversed(Result["Count"].tolist())),
                           index=RESULTS.columns[:len(Result)])

RESULTS.to_csv(r"results/UrbanReachOfStations.csv")

# @Threshold=0.01, there is 42 Communities, has coverage of 98693 / 123643 = 79.8%
# @Threshold=0.1 , there is 111 Communities, has coverage of 99488 / 123643 = 80.5%
