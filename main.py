import matplotlib as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import networkx as nx
import osmnx as ox
ox.settings.log_console = True

# Load Graphs & Boundaries from file
RoadNetwork_Walkable = ox.load_graphml(r"data/RoadNetworks/R1234.graphml")
RoadNetwork_Drivable = ox.load_graphml(r"data/RoadNetworks/R0123.graphml")
RoadNetwork_Buses = ox.load_graphml(r"data/RoadNetworks/R123.graphml")
Boundaries_Suburbs = gpd.read_file(r"data/SubdivisionBoundaries/Suburbs.gpkg")  # DEFUNCT - Might be useful in Analysis
BuildingCentroids = gpd.read_file(r"data/BuildingCentroids.gpkg")

# Hyper-parameters
BUS_STATIONS = 100
trip_times = [25, 20, 15, 10, 5]  # in minutes
travel_speed = 4.5  # walking speed in km/hour

if __name__ == '__main__':
    # do something
    print()
