import os
import sys
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as sh
from joblib import Parallel
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.features import rasterize
import shapely as sh
from shapely import wkt
from alphashape import alphashape
from progressbar import progressbar as pbar

epsg4326 = CRS.from_epsg(4326)


def flatten_geom(geom, r=[]):
    """
    flattens multipart geo
    geom: a shapely geometry. if multipart, recursion will be used, else it will just be returned           
    r: list of return values for recursion
    
    returns: a list of geometries. if geom is not multipart, returns [geom]
    """
    
    if not isinstance(geom, sh.geometry.base.BaseGeometry):
        raise ValueError("'geom' must be a shapely geometry")

    
    # if not multipart
    if not 'geoms' in dir(geom):
        return [geom]
        
    # loop over all geometries
    for g in geom.geoms:
        r = r + list(flatten_geom(g))

    return r


def concave_hull(geoms, use_pbar=False):
    """
    compute the concave hull of a list of geometries using alphashape
    
    geoms: a shapely geometry or a list of shapely geometries
    
    returns: a shapely geometry with the concave hull
    """
    if not hasattr(geoms, '__iter__'):
        geoms = [geoms]

    if hasattr(geoms, 'geoms'):
        geoms = geoms.geoms
        
    if use_pbar:
        pb = pbar
    else:
        pb = lambda x: x
    
    coords = []
    for g in pb(geoms):
        for p in flatten_geom(g):
            for c in p.boundary.coords:
                coords.append(c)
    return alphashape(coords, 1.0)


def get_boundary(w):
    """
    computes the boundary of a geopandas dataframe by splitting it, computing 
    the concave hulls of each split, and once again the concave hull of these
    
    w: a geopandas dataframe
    
    returns: a shapely geometry
    """
    
    splits = list(range(0, len(w), 1000))
    ashapes = []
    print("computing boundary by splitting", flush=True)
    for i in pbar(range(len(splits))):
        a,b = splits[i], splits[i+1] if i<len(splits)-1 else len(w)

        ashapes.append(concave_hull(w[a:b].geometry.values))

    print ("joining splits", flush=True)
    r = concave_hull(ashapes)
    return r

def get_dataset_definition(dataset_name):
    # define gee image object
    try:
        try:
            cmd = f"from .defs.{dataset_name.replace('-', '')} import DatasetDefinition"
            exec(cmd, globals())
        except Exception as e:
            cmd = f"from .defs.{dataset_name.split('-')[0]} import DatasetDefinition"
            exec(cmd, globals())

        dataset_definition = DatasetDefinition(dataset_name)
        pyfname = dataset_name
    except Exception as e:
        if os.path.isfile(dataset_name):
            pyfname = dataset_name
        elif os.path.isfile(dataset_name+".py"):
            pyfname = dataset_name+".py"
        else:
            raise ValueError(f"dataset {dataset_name} not found. {str(e)}")

        print (f"evaluating python code at {pyfname}")
        dataset_name = open(pyfname).read()
        try:
            exec(dataset_name, globals())
            dataset_definition = DatasetDefinition(dataset_name)
        except Exception as e:
            print ("--------------------------------------")
            print (f"error executing your code at {pyfname}")
            print ("--------------------------------------")
            raise e

    return dataset_definition

class mParallel(Parallel):
    """
    substitutes joblib.Parallel with richer verbose progress information
    """
    def _print(self, msg, msg_args):
        if self.verbose > 10:
            fmsg = '[%s]: %s' % (self, msg % msg_args)
            sys.stdout.write('\r ' + fmsg)
            sys.stdout.flush()



class AOINames:
    """
    a class to hold the allowed AOI names
    """
    def __init__(self):
        pass
    
    def load(self):
        import pkg_resources

        stream = pkg_resources.resource_stream(__name__, 'data/natural_earth.parquet')
        self.data = gpd.read_parquet(stream)

    def get_country_codes(self):
        ccodes = {}
        for _,row in self.data.iterrows():
            code = row.SOV_A3
            if code[-1]=='1':
                code = code[:-1]
            ccodes[row.SOVEREIGNT] = code
        return ccodes

    def get_aoi(self, aoiname):

        if os.path.isfile(aoiname):
            with open(aoiname) as f:
                aoi = wkt.load(f)
            return aoi

        if len(aoiname)==2:
            zaoiname=f'{aoiname}1'
        else:
            zaoiname = aoiname
        search_cols = ['SOVEREIGNT', 'SOV_A3', 'CONTINENT', 'REGION_UN', 'SUBREGION', 'REGION_WB']
        r = None
        for col in search_cols:
            if zaoiname.lower() in self.data[col].str.lower().values:
                r = self.data[self.data[col].str.lower()==zaoiname.lower()]
                break
        if r is None:
            return None
        else:
            return sh.ops.unary_union(r.geometry.values)

aoinames = AOINames()


def expand_dict_column(d, col):
    """
    expands a column with a list of dictionaries 
    into indivual columns for each key
    """
    t = pd.DataFrame(list(d[col].values), index=d.index).fillna(0)
    t.columns = [f'{col}__{i}' for i in t.columns]

    # remove previously existing columns with the same names in d
    d = d[[c for c in d.columns if not c in t.columns]]
    return d.join(t)


def get_binary_mask(geometry, raster_shape):
    """
    creates a binary mask for a shapely geometry
    
    geometry: a shapely geometry
    raster_shape: the shape of the resulting raster
    
    returns: an np array of shape raster_shape with 0's and 1's corresponding
             to the binary mask of geometry.
    """
    raster_shape = raster_shape[:2]

    if 'geoms' in dir(geometry):
        pols = list(geometry.geoms)
    else:
        pols = [geometry]

    # get all coords and normalize to [0,1]
    c = np.r_[[coord for p in pols for coord in p.exterior.coords ]]
    cpols = [(p.exterior.coords - np.min(c, axis=0))/(np.max(c,axis=0) - np.min(c, axis=0)) for p in pols]

    # switch y (lat)
    for p in cpols:
        p[:,1] = 1-p[:,1]

    # scale to raster_shape
    cpols = [p*np.r_[raster_shape[::-1]] for p in cpols]

    # create polygons and rasterize
    cpols = [sh.geometry.Polygon(p) for p in cpols]
    mask = rasterize(cpols, raster_shape, fill=0, default_value=1)
    return mask

def get_region_hash(region):
    """
    region: a shapely geometry
    returns a hash string for region using its coordinates
    """
    s = str(np.r_[region.envelope.boundary.coords].round(5))
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k

def get_regionlist_hash(regionlist):
    """
    returns a hash string for a list of shapely geometries
    """
    s = [get_region_hash(i) for i in regionlist]
    s = " ".join(s)
    k = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**15
    k = str(hex(k))[2:].zfill(13)
    return k


def get_utm_crs(lon, lat):
    """
    returns a UTM CRS in meters with the zone corresponding to lon, lat
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if len(utm_crs_list)==0:
        raise ValueError(f"could not get utm for lon/lat: {lon}, {lat}")
        
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def apply_value_map(array, value_map):
    """
    changes values of 'array' according to map
    value_map: a list, which will produce a map of ordered values in the list to ints 0..n
               a dict with the explicit map
               
    returns: same shape as 'array' but with the values changed
    """
    # if value_map is a list, map ordered values in list to 0..n
    if isinstance(value_map, list):
        if not np.alltrue([isinstance(i, int) for i in value_map]):
            raise ValueError("all mapped values must be int")
        value_map = sorted(value_map)
        
        # add zero if not in map
        if not 0 in value_map:
            value_map = [0] + value_map
            
        value_map = {i:value_map[i] for i in range(len(value_map))}

    # if value map is dict just check all keys and values are ints
    elif isinstance(value_map, dict):
        if not np.alltrue([isinstance(i, int) for i in value_map.keys()]):
            raise ValueError("all keys in map dict must be int")

        if not np.alltrue([isinstance(i, int) for i in value_map.values()]):
            raise ValueError("all values in map dict must be int")

        # add zero if not in map
        if not 0 in value_map.keys() and not 0 in value_map.values():
            value_map[0] = 0

    if 0 in value_map.keys() and value_map[0]==0:
        init_val = 0
    else:
        init_val = list(value_map.keys())[0]

    r = np.ones_like(array)*init_val

    for k,v in value_map.items():
        if v==init_val:
            continue

        r[array==k] = v    
        
    return r

def apply_range_map(array, range_map):
    """
    changes values of array according to interval ranges
    range map: a list of n floats defining a sequence of n+1 intervals
               to create one class per intervar numbered 0,...,n
               
    for instance, if range_map is [5,10,12], 
        - values < 5 will become 0
        - values >=5 and <10 will become 1
        - values >=10 will become 2
    """
    
    range_map = np.r_[range_map]

    if len(range_map.shape)!=1:
        raise ValueError("range_map must have one dimension")

    try:
        range_map = range_map.astype(float)
    except:
        raise ValueError("range_map must be a list of floats")

    if not np.alltrue(range_map[1:]-range_map[:-1]>0):
        raise ValueError("range_map must be a list or ordered floats with no repetitions")

    r = np.zeros_like(array)
    for i in range(0, len(range_map)):
        if i==len(range_map)-1:
            r[array>=range_map[i]] = i+1
        else:
            r[ (array>=range_map[i]) & (array<range_map[i+1]) ] = i+1

    return r

def touch(filename):
    with open(filename,"w") as f:
        pass