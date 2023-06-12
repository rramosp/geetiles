
import requests
import rasterio
import shutil
import os
import ee
from retry import retry
from skimage import exposure
import rasterio.mask
import multiprocessing
import numpy as np
import geopandas as gpd
from pyproj import CRS

from . import utils

epsg4326 = utils.epsg4326

_gee_get_tile_progress_period = 100
def _get_tile(i,gee_tile):
    # helper function to download gee tiles
    try:
        gee_tile.get_tile()
    except Exception as e:
        print (f"\n----error----\ntile {gee_tile.identifier}\n------")
        print (e)

    if i%_gee_get_tile_progress_period==0:
        print (f"{i} ", end="", flush=True)

def get_gee_tiles(self, 
                    image_collection, 
                    dest_dir=".", 
                    file_prefix="geetile_", 
                    pixels_lonlat=None, 
                    meters_per_pixel=None,
                    remove_saturated_or_null = False,
                    enhance_images = None,
                    dtype = None,
                    skip_if_exists = False):
    r = []
    for g,i in zip(self.data.geometry.values, self.data.identifier.values):
        r.append(GEETile(image_collection=image_collection,
                        region = g,
                        identifier = i,
                        dest_dir = dest_dir, 
                        file_prefix = file_prefix,
                        pixels_lonlat = pixels_lonlat,
                        meters_per_pixel = meters_per_pixel,
                        remove_saturated_or_null = remove_saturated_or_null,
                        enhance_images = enhance_images,
                        dtype = dtype,
                        skip_if_exists = skip_if_exists)
                )
    return r    


def download_tiles(
                    data,
                    dest_dir,
                    gee_image, 
                    n_processes=10, 
                    pixels_lonlat=None, 
                    meters_per_pixel=None,
                    remove_saturated_or_null = False,
                    enhance_images = None,
                    max_downloads=None, 
                    shuffle=True,
                    skip_if_exists = False,
                    dtype = None):

    """
    downloads in parallel tiles from GEE. See GEETile below for parameters info.
    data: a geopandas dataframe with columns 'geometry' and 'identifier'
    """        
    if not data.crs == epsg4326:
           raise ValueError("'data' must be in epsg4326, lon/lat degrees "+\
                            f"when downloading gee tiles, but found \n{data.crs}")
    
    global _gee_get_tile_progress_period
    
    if not skip_if_exists and os.path.exists(dest_dir):
        raise ValueError(f"destination folder {dest_dir} already exists")
    
    os.makedirs(dest_dir, exist_ok=True)

    gtiles = []
    for g,i in zip(data.geometry.values, data.identifier.values):
        gtiles.append(GEETile(image_collection=gee_image,
                        region = g,
                        identifier = i,
                        dest_dir = dest_dir, 
                        file_prefix = "",
                        pixels_lonlat = pixels_lonlat,
                        meters_per_pixel = meters_per_pixel,
                        remove_saturated_or_null = remove_saturated_or_null,
                        enhance_images = enhance_images,
                        dtype = dtype,
                        skip_if_exists = skip_if_exists)
                )

    if shuffle:
        gtiles = np.random.permutation(gtiles)
    if max_downloads is not None:
        gtiles = gtiles[:max_downloads]

    print (f"downloading {len(gtiles)} tiles. showing progress of parallel downloading.", flush=True)                                    
    _gee_get_tile_progress_period = np.max([len(gtiles)//100,1])
    pool = multiprocessing.Pool(n_processes)
    pool.starmap(_get_tile, enumerate(gtiles))
    pool.close()

class GEETile:
    
    def __init__(self, 
                 region, 
                 image_collection, 
                 dest_dir=".", 
                 file_prefix="geetile_", 
                 meters_per_pixel=None, 
                 pixels_lonlat=None, 
                 identifier=None,
                 remove_saturated_or_null = False,
                 enhance_images = None,
                 dtype = None,
                 skip_if_exists = True):
        """
        region: shapely geometry in epsg 4326 lon/lat
        dest_dir: folder to store downloaded tile from GEE
        file_prefix: to name the tif file with the downloaded tile
        meters_per_pixel: an int, if set, the tile pixel size will be computed to match the requested meters per pixel
        pixels_lonlat: a tuple, if set, the tile will have this exact size in pixels, regardless the physical size.
        image_collection: an instance of ee.ImageCollection
        remove_saturated_or_null: if true, will remove image if saturated or null > 1%
        enhance_images: operation to enhance images
        """
        if not enhance_images in [None, 'none', 'gamma']:
            raise ValueError(f"'enhace_images' value '{enhance_images}' not allowed")

        if sum([(meters_per_pixel is None), (pixels_lonlat is None)])!=1: 
            raise ValueError("must specify exactly one of meters_per_pixel or pixels_lonlat")
            
        self.region = region
        self.meters_per_pixel = meters_per_pixel
        self.image_collection = image_collection
        self.dest_dir = dest_dir
        self.file_prefix = file_prefix
        self.remove_saturated_or_null = remove_saturated_or_null
        self.enhance_images = enhance_images
        self.dtype = dtype
        self.skip_if_exists = skip_if_exists


        if identifier is None:
            self.identifier = utils.get_region_hash(self.region)
        else:
            self.identifier = identifier
                    
        if pixels_lonlat is not None:
            self.pixels_lon, self.pixels_lat = pixels_lonlat



    @retry(tries=10, delay=1, backoff=2)
    def get_tile(self):

        # check if should skip
        ext = 'tif'
        outdir = os.path.abspath(self.dest_dir)
        filename    = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}"

        if self.skip_if_exists and os.path.exists(filename):
            return


        # get appropriate utm crs for this region to measure stuff in meters 
        lon, lat = list(self.region.envelope.boundary.coords)[0]
        utm_crs = utils.get_utm_crs(lon, lat)
        self.region_utm = gpd.GeoDataFrame({'geometry': [self.region]}, crs = CRS.from_epsg(4326)).to_crs(utm_crs).geometry[0]

        # compute image pixel size if meters per pixels where specified
        if self.meters_per_pixel is not None:
            coords = np.r_[self.region_utm.envelope.boundary.coords]
            self.pixels_lon, self.pixels_lat = np.ceil(np.abs(coords[1:] - coords[:-1]).max(axis=0) / self.meters_per_pixel).astype(int)

        # build image request
        dims = f"{self.pixels_lon}x{self.pixels_lat}"

        try:
            rectangle = ee.Geometry.Polygon(list(self.region.boundary.coords)) 
        except:
            # in case multipolygon, or mutipart or other shapely geometries without a boundary
            rectangle = ee.Geometry.Polygon(list(self.region.envelope.boundary.coords)) 

        url = self.image_collection.getDownloadURL(
            {
                'region': rectangle,
                'dimensions': dims,
                'format': 'GEO_TIFF',
                'crs': 'EPSG:4326'
            }
        )

        band_names = self.image_collection.bandNames().getInfo()
    
        # download and save to tiff
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            r.raise_for_status()

        with open(filename, 'wb') as outfile:
            shutil.copyfileobj(r.raw, outfile)

        # reopen tiff to mask out region, set image type and band names
        with rasterio.open(filename) as src:
            out_image, out_transform = rasterio.mask.mask(src, [self.region], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        if self.dtype is not None:
            out_image = out_image.astype(self.dtype)
            out_meta['dtype'] = self.dtype

        with rasterio.open(filename, "w", **out_meta) as dest:
            dest.write(out_image)  
            for i in range(out_image.shape[0]):
                if self.dtype is not None:
                    dest.write_band(i+1, out_image[i].astype(self.dtype)  )
                dest.set_band_description(i+1, band_names[i])

        # open raster again to adjust and check saturation and invalid pixels
        with rasterio.open(filename) as src:
            x = src.read()
            x = np.transpose(x, [1,2,0])
            m = src.read_masks()
            profile = src.profile.copy()


        # enhance image
        if self.enhance_images=='gamma':
            x = exposure.adjust_gamma(x, gamma=.8, gain=1.2)

        # if more than 1% pixels saturated or invalid then remove this file
        if self.remove_saturated_or_null and \
            (np.mean(x>250)>0.01 or np.mean(m==0)>0.01):
            os.remove(filename)
        else:
            # write enhanced image
            with rasterio.open(filename, 'w', **profile) as dest:
                for i in range(src.count):
                    dest.write(x[:,:,i], i+1)      
                    dest.set_band_description(i+1, band_names[i])
