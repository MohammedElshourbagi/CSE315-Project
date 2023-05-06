import shapely
import osmnx as ox

# Defining Boundary of interest 'New Cairo City'
NewCairoCity = shapely.from_wkt("MultiPolygon (((31.36722200010136774 29.98841439972389011, 31.38342363598793483 29.9948128057707919, 31.3907762110445816 29.99744169968346341, 31.39783648189409604 30.00723615058714344, 31.39830406538438012 30.01008395616501545, 31.40072224600004347 30.01312395600007221, 31.40130046800004493 30.01749960200004352, 31.40408347200002481 30.03855529400004798, 31.40566897800005108 30.05054762700007132, 31.4083419940000681 30.05386406800005261, 31.41919063200003848 30.06732136700003366, 31.43137497215823473 30.07972016386689162, 31.48877615692682852 30.08767716384134161, 31.50285238954576172 30.08509972463840043, 31.50384172686224815 30.08550829577939467, 31.50557306716610739 30.08531373830370725, 31.50595531112929848 30.08486625465714326, 31.50896829295679069 30.08486625465714326, 31.52085862430056906 30.08835983769397515, 31.53429757675286993 30.08555693508850482, 31.54171760662656254 30.08839742924887517, 31.56235878063881373 30.09551776144340351, 31.56433745527179369 30.09567339139307407, 31.57201283364950584 30.10123919644603774, 31.58974160430650002 30.10452967797683854, 31.59164311650761192 30.08445468559226654, 31.5935683304055992 30.08340255571934208, 31.59817871105604681 30.08193393905723312, 31.60017992023947997 30.08053106098699558, 31.61275713998093551 30.05514429959966805, 31.61301679053953606 30.05206368397125871, 31.60146550715160885 30.03487187030270888, 31.60012292377538756 30.02741530944528492, 31.59642448655030833 30.0201994562627057, 31.58221336817174674 30.00970372947335818, 31.53336401412015277 29.94259794978227873, 31.51735750400007419 29.94216699300005757, 31.51660638000004155 29.94109090100005233, 31.50796941224427528 29.94681038452512212, 31.50013171192542671 29.95081231988557136, 31.49623057500698664 29.95142600643259101, 31.48959751799869267 29.95635484354220779, 31.48248103480165483 29.96207237806012458, 31.47709589190847979 29.96600725333798465, 31.47520715703154792 29.96599751374003873, 31.46912498102902234 29.97006858248925809, 31.46782085456637645 29.97075032410890927, 31.46423450679408163 29.97420765601921744, 31.45972627652234621 29.97568794162788208, 31.451271939454152 29.97411026803478507, 31.44722465043214044 29.97412974563930632, 31.42986627751551509 29.97131519219295015, 31.38934165953641653 29.97280862397017742, 31.37886150900004623 29.9723427910000737, 31.37581794400006174 29.9722072590000721, 31.37442301600003702 29.97214512500005412, 31.37439219200007301 29.97361959900007378, 31.37190800500007981 29.97476874800003088, 31.37088184900005672 29.97524343800006363, 31.37002595100005919 29.97575335900006621, 31.36883391200007054 29.97640808600004902, 31.36797978200007719 29.97687721400006922, 31.36762088600005427 29.9770743300000504, 31.36729287700006807 29.97725448200003484, 31.36727607500006343 29.97726020300007121, 31.36564457300005415 29.98663313500003369, 31.36722200010136774 29.98841439972389011)))")

# Initial Step, Gathering all Data Required from OSM then saving in disk
# Road Networks
R0123 = ox.graph_from_polygon(NewCairoCity, custom_filter='["highway"~"trunk|primary|secondary|tertiary"]')
ox.io.save_graph_geopackage(R0123, filepath=r"data/RoadNetworks/R0123.gpkg")
ox.io.save_graphml(R0123, filepath=r"data/RoadNetworks/R0123.graphml")

R123 = ox.graph_from_polygon(NewCairoCity, custom_filter='["highway"~"primary|secondary|tertiary"]')
ox.io.save_graph_geopackage(R123, filepath=r"data/RoadNetworks/R123.gpkg")
ox.io.save_graphml(R123, filepath=r"data/RoadNetworks/R123.graphml")

# Walkable Network
from main import travel_speed
meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
    # add an edge attribute for time in minutes required to traverse each edge
R1234 = ox.graph_from_polygon(NewCairoCity, network_type='walk')
for u, v, k, data in R1234.edges(data=True, keys=True):
    data['time'] = data['length'] / meters_per_minute

ox.io.save_graph_geopackage(R1234, filepath=r"data/RoadNetworks/R1234.gpkg")
ox.io.save_graphml(R1234, filepath=r"data/RoadNetworks/R1234.graphml")

# Defunct - might remove later
"""
    # Gathering all neighbourhood boundaries within New Cairo City
Neighbourhoods = ox.geometries_from_polygon(NewCairoCity, tags={'place': 'neighbourhood'})
# Remove unwanted data gathered from OSM
Neighbourhoods.drop(columns=['nodes', 'name', 'name:ar', 'operator', 'place', 'start_date', 'leisure', 'construction',
                             'website', 'name:fr', 'boundary', 'description', 'alt_name', 'alt_name:ar',
                             'official_name', 'official_name:ar', 'wikidata', 'wikipedia', 'area'], axis=1, inplace=True)
# Neighbourhoods.to_file('NewCairoNeighbourhoods.gpkg', driver="GPKG")  # Save as GeoPackage file

Boundaries_Neighbourhoods = gpd.read_file(r"data/SubdivisionBoundaries/NewCairoNeighbourhoods.gpkg")

"""