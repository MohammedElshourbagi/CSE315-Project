import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
import osmnx as ox

from main import RoadNetwork_Walkable, RoadNetwork_Buses, BuildingCentroids, trip_times

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

NODES = list(RoadNetwork_Buses.nodes())
RESULTS = pd.DataFrame(columns=trip_times)    # Empty Dataframe used to store results

At = 1
AmountOfNodes = len(NODES)
for Node in NODES:
    print(f"at Node : {Node} | {(At/AmountOfNodes)*100}% complete")
    NODE = RoadNetwork_Buses.nodes()[Node]
    LookAtNode = ox.nearest_nodes(RoadNetwork_Walkable, X=NODE['x'], Y=NODE['y'])

    # Creating the Isochrone Areas
    iso_colors = ox.plot.get_colors(n=len(trip_times), cmap='plasma', start=0, return_hex=True)  # Assign one color for each isochrone
    IsochronePolys = make_iso_polys(RoadNetwork_Walkable, AtNode=LookAtNode, infill=True)

    # Finding the building count within the Isochrone
    Result = gpd.sjoin(left_df=BuildingCentroids, right_df=IsochronePolys).groupby("TripTimes")['point_id'].count().rename('Count').reset_index()
    RESULTS.loc[Node] = pd.Series(list(reversed(Result["Count"].tolist())), index=RESULTS.columns[:len(Result)])
    At += 1

RESULTS.to_csv(r"NodesUrbanReach.csv")

# Visualize Results
"""
    # Isochrone Area
import matplotlib.pyplot as plt

node_colors = {}
for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
    subgraph = nx.ego_graph(RoadNetwork_Walkable, n=LookAtNode, radius=trip_time)
    for node in subgraph.nodes():
        node_colors[node] = color
nc = [node_colors[node] if node in node_colors else 'none' for node in RoadNetwork_Walkable.nodes()]
ns = [15 if node in node_colors else 0 for node in RoadNetwork_Walkable.nodes()]
fig, ax = ox.plot_graph(RoadNetwork_Walkable, node_color=nc, node_size=ns, node_alpha=0.8, node_zorder=2, bgcolor='k', edge_linewidth=0.2, edge_color='#999999')
"""
