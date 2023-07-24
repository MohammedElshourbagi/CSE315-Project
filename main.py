# Galala University, Faculty of Computer Science and Engineering, Spring Semester 2022-2023
# CSE312: Discrete Mathmatics, Professor: Dr Mohamed Abdel-Aziz
# Student: Mohammed Saeid Elshourbagi, Sophomore Computer Engineer, ID:221100633

# Project on GeoSpatial Graphs, exploring the viability of Public Bus Transportation in New Cairo City
# An attempt to solve the Optimal Bus Placement Problem

# Importing Libraries ==================================================================================================
import shapely
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
ox.settings.log_console = True

# Hyper-parameters =====================================================================================================
BUS_STATIONS = 100
trip_times = [25, 20, 15, 10, 5]  # in minutes
travel_speed = 4.5  # walking speed in km/hour

# Create and save networks generated from OSM ==========================================================================
# Defining area of study as a Shapley Geometry
import pickle
"""
polygonGeometry = shapely.geometry.Polygon()
with open(r"data.NewCairoCity", "wb") as poly_file:
    pickle.dump(polygonGeometry, poly_file, pickle.HIGHEST_PROTOCOL)
"""

with open(r"data/NewCairoCity", "rb") as poly_file:
    NewCairoCity = pickle.load(poly_file)

import os
# Load saved files from Disk
if os.path.isfile(r"data/RoadNetworks/R123.gpkg") is True:
    RoadNetwork_Walkable = ox.load_graphml(r"data/RoadNetworks/R1234.graphml")
    RoadNetwork_Drivable = ox.load_graphml(r"data/RoadNetworks/R0123.graphml")
    RoadNetwork_Buses = ox.load_graphml(r"data/RoadNetworks/R123.graphml")
    Centroids = gpd.read_file(r"data/Centroids.gpkg")

# Initializing Data, Run queries to gather data from OpenStreetMaps
else:
    # Road Networks
    R0123 = ox.graph_from_polygon(NewCairoCity, custom_filter='["highway"~"trunk|primary|secondary|tertiary"]')
    ox.io.save_graph_geopackage(R0123, filepath=r"data/RoadNetworks/R0123.gpkg")
    ox.io.save_graphml(R0123, filepath=r"data/RoadNetworks/R0123.graphml")

    R123 = ox.graph_from_polygon(NewCairoCity, custom_filter='["highway"~"primary|secondary|tertiary"]')
    ox.io.save_graph_geopackage(R123, filepath=r"data/RoadNetworks/R123.gpkg")
    ox.io.save_graphml(R123, filepath=r"data/RoadNetworks/R123.graphml")

    # Walkable Network
    meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
        # add an edge attribute for time in minutes required to traverse each edge
    R1234 = ox.graph_from_polygon(NewCairoCity, network_type='walk')
    for u, v, k, data in R1234.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute

    ox.io.save_graph_geopackage(R1234, filepath=r"data/RoadNetworks/R1234.gpkg")
    ox.io.save_graphml(R1234, filepath=r"data/RoadNetworks/R1234.graphml")

    buildings = ox.geometries_from_polygon(NewCairoCity,
                                           tags={'building': True, 'craft': True, 'shop': True, 'tourism': True}
                                           ).to_crs(crs='epsg:4326')
    # ox.plot_footprints(buildings)

# Creating Isochrone Areas for each node of the Network ================================================================
def make_iso_polys(G, AtNode, edge_buff=0.001, node_buff=0.001, infill=False):
    isochrone_polys = []
    for trip_time in sorted(trip_times, reverse=True):
        subgraph = nx.ego_graph(G, n=AtNode, radius=trip_time)
        node_points = [shapely.geometry.Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
        nodes_gdf = gpd.GeoDataFrame({'id': subgraph.nodes()}, geometry=node_points)
        nodes_gdf = nodes_gdf.set_index('id')

        edge_lines = []
        for n_fr, n_to in subgraph.edges():
            f = nodes_gdf.loc[n_fr].geometry
            t = nodes_gdf.loc[n_to].geometry
            edge_lookup = G.get_edge_data(n_fr, n_to)[0].get('geometry', shapely.geometry.LineString([f, t]))
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

"""
NODES = list(RoadNetwork_Buses.nodes())
RESULTS = pd.DataFrame(columns=trip_times)    # Empty Dataframe used to store results

At = 1
AmountOfNodes = len(NODES)
for Node in NODES:
    print(f"at Node : {Node} | {(At/AmountOfNodes)*100}% complete")
    NODE = RoadNetwork_Buses.nodes()[Node]
    LookAtNode = ox.nearest_nodes(RoadNetwork_Walkable, X=NODE['x'], Y=NODE['y'])

    # Creating the Isochrone Areas
    iso_colors = ox.plot.get_colors(n=len(trip_times), cmap='plasma', start=0, return_hex=True)
    IsochronePolys = make_iso_polys(RoadNetwork_Walkable, AtNode=LookAtNode, infill=True)

    # Finding the building count within the Isochrone
    Result = gpd.sjoin(left_df=Centroids, right_df=IsochronePolys
                       ).groupby("TripTimes")['point_id'].count().rename('Count').reset_index()
    RESULTS.loc[Node] = pd.Series(list(reversed(Result["Count"].tolist())), index=RESULTS.columns[:len(Result)])
    At += 1

RESULTS.to_csv(r"NodesUrbanReach.csv")

# Visualize results using mpl.pyplot
node_colors = {}
for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
    subgraph = nx.ego_graph(RoadNetwork_Walkable, n=LookAtNode, radius=trip_time)
    for node in subgraph.nodes():
        node_colors[node] = color
nc = [node_colors[node] if node in node_colors else 'none' for node in RoadNetwork_Walkable.nodes()]
ns = [15 if node in node_colors else 0 for node in RoadNetwork_Walkable.nodes()]
fig, ax = ox.plot_graph(RoadNetwork_Walkable, 
                        node_color=nc, node_size=ns, node_alpha=0.8, node_zorder=2, 
                        bgcolor='k', edge_linewidth=0.2, edge_color='#999999')
"""

# NETWORK Analysis =====================================================================================================
# Finding the Centrality Coefficients of each Node in the Network
NETWORK = nx.Graph(RoadNetwork_Buses)
"""
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
"""

# Calculating the Urban Reach ==========================================================================================
# defined as the count of buildings that is found within a walking distance [5,10, ..., 25] minute intervals from node.

# Adding a change of urban reach for each edge -------------------------------------------------------------------------
# DeltaUrbanReach using a modified minmax normalization, where larger values are fitted to a smaller numbers
# This is done because of Graph Algorithms prioritizes smaller weights
"""
nx.set_edge_attributes(NETWORK, 0, "DeltaReach")
minDelta, maxDelta = 0, 0
for u, v in NETWORK.edges():  # Min Max values are found first ---------------------------------------------------------
    delta = int(NETWORK.nodes()[u]["25"][:-2]) - int(NETWORK.nodes()[v]["25"][:-2])
    minDelta = min(minDelta, delta)
    maxDelta = max(maxDelta, delta)
for u, v in NETWORK.edges():  # Values are then fitted -----------------------------------------------------------------
    delta = int(NETWORK.nodes()[u]["25"][:-2]) - int(NETWORK.nodes()[v]["25"][:-2])
    normalizedDelta = ((delta - maxDelta) / (minDelta - maxDelta))
    NETWORK.edges[u, v].update(DeltaReach=normalizedDelta)

# BETWEENNESS CENTRALITY on LOUVAIN COMMUNITIES -------------------------------------------------------------------------
communities = sorted(nx.community.louvain_communities(NETWORK, weight="DeltaReach", threshold=0.01), key=len, reverse=True)
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

# Can't use precalculated Urban reach, comes with the problem of over-counting the buildings ---------------------------
NODES = list(CentralNodesForEachCommunity.keys())
IsochronePolys = gpd.GeoDataFrame(columns=trip_times)
for Node in NODES:
    NODE = RoadNetwork_Buses.nodes()[Node]
    LookAtNode = ox.nearest_nodes(RoadNetwork_Walkable, X=NODE['x'], Y=NODE['y'])
    IsochronePolys.loc[len(IsochronePolys.index)] = list(
        make_iso_polys(RoadNetwork_Walkable, AtNode=LookAtNode, infill=True).loc[:, 0]
    )

MergedPolys = []
for i in range(len(trip_times)):
    MergedPoly = shapely.MultiPolygon()
    for j in range(len(NODES)-1):
        MergedPoly = shapely.ops.unary_union([MergedPoly, IsochronePolys.iloc[j, i], IsochronePolys.iloc[j + 1, i]])
    MergedPolys.append(MergedPoly)

CombinedPolys = gpd.GeoDataFrame(pd.DataFrame(), geometry=gpd.GeoSeries(MergedPolys), crs='epsg:4326')
CombinedPolys["TripTimes"] = trip_times
CombinedPolys.to_file(r"results/UrbanReachOfStations.gpkg", driver="GPKG")

# Finding the building count within the Isochrone ----------------------------------------------------------------------
RESULTS = pd.DataFrame(columns=trip_times)  # Empty Dataframe used to store results
Result = gpd.sjoin(left_df=Centroids,
                   right_df=CombinedPolys
                   ).groupby("TripTimes")['point_id'].count().rename('Count').reset_index()

RESULTS.loc[0] = pd.Series(list(reversed(Result["Count"].tolist())),
                           index=RESULTS.columns[:len(Result)])

RESULTS.to_csv(r"results/UrbanReachOfStations.csv")
"""
# ======================================================================================================================
