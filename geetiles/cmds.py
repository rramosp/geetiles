
import itertools
import ee
import os
from shapely import wkt
import geopandas as gpd
import numpy as np
import shapely as sh
from joblib import delayed
from pyproj import CRS
from progressbar import progressbar as pbar

from . import partitions
from . import utils

epsg4326 = utils.epsg4326


def download(tiles_file, 
             gee_image_pycode, 
             dataset_name, 
             pixels_lonlat, meters_per_pixel, 
             max_downloads,
             shuffle,
             skip_if_exists,
             dtype,
             ee_auth_mode,
             n_processes,
             skip_confirm=False):
    
    # sanity check
    if (pixels_lonlat is None and meters_per_pixel is None) or\
       (pixels_lonlat is not None and meters_per_pixel is not None):
        raise ValueError ("must specify exactly one of 'pixels_lonlat' or 'meters_per_pixel'")

    if pixels_lonlat is not None:
        try:
            pixels_lonlat = eval(pixels_lonlat)
            pixels_lonlat = [int(i) for i in pixels_lonlat]
            if not len(pixels_lonlat)==2:
                raise Exception
        except:
            raise ValueError("'pixels_lonlat' must be a tuple of two ints such as --pixels_lonlat [100,100]")


    print (f"""
using the following download specficication

tiles_file:        {tiles_file}
gee_image_pycode   {gee_image_pycode}
dataset_name       {dataset_name}
pixels_lonlat      {pixels_lonlat}
meters_per_pixel   {meters_per_pixel}
max_downloads      {max_downloads}
shuffle            {shuffle}
skip_if_exists     {skip_if_exists}
dtype              {dtype}
ee_auth_mode       {ee_auth_mode}
n_processes        {n_processes}

        """)
    
    if not skip_confirm:
        while True:
            yesno = input("confirm (y/N): ")        
            yesno = yesno.lower()
            if yesno.strip()=='':
                yesno = 'n'
            if yesno in ['y', 'n', 'yes', 'no']:
                break
        
        if yesno in ['n', 'no']:
            print ("abort!!")
            return
        
    # authenticate on google earth engine
    print ("authenticating to Google Earth Engine")
    if ee_auth_mode is None:
        try:
            print ("trying to use default gee credentials")
            ee.Authenticate(auth_mode = 'appdefault')    
        except:
            print ("could not authenticate with default gee credentials, using auth_method = 'notebook'")
            ee.Authenticate(auth_mode = 'notebook')
        
    else:
        ee.Authenticate(auth_mode = ee_auth_mode)

    ee.Initialize()

    # define gee image object
    if gee_image_pycode == 'sentinel2-rgb-median-2020':
        def maskS2clouds(image):
            qa = image.select('MSK_CLDPRB')
            mask = qa.lt(5)
            return image.updateMask(mask)

        gee_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                        .filterDate('2020-01-01', '2020-12-31')\
                        .map(maskS2clouds)\
                        .select('B4', 'B3', 'B2')\
                        .median()\
                        .visualize(min=0, max=4000)
        
    elif gee_image_pycode == 'esa-world-cover':
        gee_image = ee.ImageCollection("ESA/WorldCover/v100").first()
        
    elif os.path.isfile(gee_image_pycode):
        print (f"evaluating python code at {gee_image_pycode}")
        pyfname = gee_image_pycode
        gee_image_pycode = open(pyfname).read()
        try:
            exec(gee_image_pycode, globals())
            gee_image = get_ee_image()
        except Exception as e:
            print ("--------------------------------------")
            print (f"error executing your code at {pyfname}")
            print ("--------------------------------------")
            raise e
        
    else:
        raise ValueError(f"file {gee_image_pycode} not found")
        
    # download the tiles
    p = partitions.PartitionSet.from_file(tiles_file)

    # save gee_image_codestr
    dest_dir = p.get_downloaded_tiles_dest_dir(dataset_name)
    with open(f"{dest_dir}.gee_image_pycode.py", "w") as f:
        f.write(gee_image_pycode)

    p.download_gee_tiles(gee_image, dataset_name, 
                         meters_per_pixel = meters_per_pixel, 
                         pixels_lonlat = pixels_lonlat,
                         dtype = dtype, 
                         shuffle = shuffle,
                         skip_if_exists = skip_if_exists,
                         enhance_images = None,
                         max_downloads=max_downloads)
    
    print("\ndone.")

def make_random_partitions(aoi_wkt_file, max_rectangle_size_meters, aoi_name, dest_dir, random_variance=0.1, ):
    """
    makes random partitions of the aoi
    """
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
        
    parts = partitions.PartitionSet(aoi_name, region=aoi)
    parts.reset_data().make_random_partitions(max_rectangle_size=max_rectangle_size_meters, random_variance=random_variance)
    print()
    parts.save_as(dest_dir, f'{max_rectangle_size_meters//1000}k')
    
    return parts.data

def make_grid(aoi_wkt_file, chip_size_meters, aoi_name, dest_dir):
    
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
            
    grid = build_grid(aoi=aoi, chip_size_meters=chip_size_meters)
    parts = partitions.PartitionSet(aoi_name, data=grid)
    print()
    parts.save_as(dest_dir, "aschips")
    return parts.data

def build_grid(aoi, chip_size_meters):
    """
    make a grid of squared tiles. The resulting tiles sides have
    constant lat and lon, as required by GEE, otherwise unaligned geometries 
    produce null pixels on the borders, when extracting the geometry from gee.

    aoi: a shapely object with the geometry to cover. must be in degrees (epsg4326)
    chip_size_meters: the length of each chip side in meters.

    returns: a GeoPandas dataframe in epsg4326
    """
    m = chip_size_meters
    
    # make a grid of points using utm crs
    aoi_utm = utils.get_utm_crs(*list(aoi.centroid.coords)[0])
    aoim = gpd.GeoDataFrame({'geometry': [aoi]}, crs=epsg4326).to_crs(aoi_utm).geometry[0]

    rcoords = np.r_[aoim.envelope.boundary.coords]
    minx, miny = rcoords.min(axis=0)
    maxx, maxy = rcoords.max(axis=0)
    rangex = maxx-minx
    rangey = maxy-miny
    gridx = int(rangex//m)
    gridy = int(rangey//m)
    
    def get_polygon(m, gx, gy, minx, miny):

        rlon, rlat = gx*m+minx, gy*m+miny
        point = sh.geometry.Point([rlon, rlat])

        # get point in lon/lat in degrees
        p4326 = gpd.GeoDataFrame({'geometry': [sh.geometry.Point([rlon, rlat])]},
                                          crs = aoi_utm).to_crs(epsg4326)
        clon,clat = list(p4326.geometry.values[0].coords)[0]

        # obtain how many meters per degree lon and lat in this region of the globe.
        # by doing the aritmethic in degrees (not in meters) we ensure tile sides have
        # constant lat and lon, as required by GEE, otherwise unaligned geometries 
        # produce null pixels on the borders.
        lon0,lat0 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]
        lon1,lat1 = list(gpd.GeoSeries([sh.geometry.Point([clon+0.001, clat])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]
        lon2,lat2 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat+0.001])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]

        meters_per_degree_lon = (lon1-lon0) * 1000
        meters_per_degree_lat = (lat2-lat0) * 1000
        delta_degrees_lon =  ((m-1)/2) / meters_per_degree_lon
        delta_degrees_lat =  ((m-1)/2) / meters_per_degree_lat

        part =  sh.geometry.Polygon([[clon-delta_degrees_lon, clat-delta_degrees_lat], 
                                     [clon-delta_degrees_lon, clat+delta_degrees_lat],
                                     [clon+delta_degrees_lon, clat+delta_degrees_lat],
                                     [clon+delta_degrees_lon, clat-delta_degrees_lat],
                                     [clon-delta_degrees_lon, clat-delta_degrees_lat]])    
        return part    
    
    
    # create a polygon at each point
    print (f"inspecting {gridx*gridy} chips", flush=True)

    parts = utils.mParallel(n_jobs=7, verbose=30)(delayed(get_polygon)(m, gx, gy, minx, miny) \
                                            for gx,gy in itertools.product(range(gridx), range(gridy)))
    parts = [i for i in parts if i is not None and aoi.intersects(i)]
    parts = gpd.GeoDataFrame(parts, columns=['geometry'], crs=epsg4326)
    print (f"\naoi covered by {len(parts)} chips")
    return parts


def select_partitions(orig_shapefile, aoi_wkt_file, aoi_name, partition_name, dest_dir):
    """
    selects the geometries in 'orig_shafile' that have some intersention with aoi,
    assigns them an identifier and saves them in a new file.
    """
    print ("reading orig shapefile", flush=True)
    parts = gpd.read_file(orig_shapefile)

    if not  parts.crs == epsg4326:
           raise ValueError("'orig_shapefile' must be in epsg4326, lon/lat degrees "+\
                            f"but found \n{parts.crs}")
    
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
    
    print ("selecting geometries", flush=True)
    parts = [p for p in pbar(parts.geometry) if p.intersects(aoi)]
    # very small intersections probably are cause by numerical approximations
    # on the borders of the aoi
    parts = [p for p in parts if p.intersection(aoi).area>1e-5]
    
    parts = gpd.GeoDataFrame({'geometry': parts}, crs = CRS.from_epsg(4326))
    parts = partitions.PartitionSet(aoi_name, data=parts)
    print ()
    parts.save_as(dest_dir, partition_name)

    return parts
