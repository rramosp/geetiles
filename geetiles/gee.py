
import requests
import rasterio
import shutil
import os
import ee
from retry import retry
import rasterio.mask
import multiprocessing
import numpy as np
import geopandas as gpd
from pyproj import CRS
from time import sleep

from . import utils

epsg4326 = utils.epsg4326

_gee_get_tile_progress_period = 100
def _get_tile(i,gee_tile):
    # helper function to download gee tiles
    for _ in range(3):
        try:
            gee_tile.get_tile()
            break
        except Exception as e:
            print (f"\n----error----\ntile {gee_tile.identifier}\n------")
            print (e)
            print ("waiting 2secs and retrying")
            sleep(2)
        

    if i%_gee_get_tile_progress_period==0:
        print (f"{i} ", end="", flush=True)

def get_gee_tiles(self, 
                    dataset_definition, 
                    dest_dir=".", 
                    file_prefix="geetile_", 
                    pixels_lonlat=None, 
                    meters_per_pixel=None,
                    remove_saturated_or_null = False,
                    dtype = None,
                    skip_if_exists = False):
    r = []
    for g,i in zip(self.data.geometry.values, self.data.identifier.values):
        r.append(GEETile(dataset_definition=dataset_definition,
                        tile_geometry = g,
                        identifier = i,
                        dest_dir = dest_dir, 
                        file_prefix = file_prefix,
                        pixels_lonlat = pixels_lonlat,
                        meters_per_pixel = meters_per_pixel,
                        remove_saturated_or_null = remove_saturated_or_null,
                        dtype = dtype,
                        skip_if_exists = skip_if_exists)
                )
    return r    


def download_tiles(
                    data,
                    dest_dir,
                    dataset_definition, 
                    n_processes=10, 
                    pixels_lonlat=None, 
                    meters_per_pixel=None,
                    remove_saturated_or_null = False,
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
        
    os.makedirs(dest_dir, exist_ok=True)

    if dtype is None:
        dtype = dataset_definition.get_dtype()

    gtiles = []
    for g,i in zip(data.geometry.values, data.identifier.values):
        gtiles.append(GEETile(dataset_definition=dataset_definition,
                        tile_geometry = g,
                        identifier = i,
                        dest_dir = dest_dir, 
                        file_prefix = "",
                        pixels_lonlat = pixels_lonlat,
                        meters_per_pixel = meters_per_pixel,
                        remove_saturated_or_null = remove_saturated_or_null,
                        dtype = dtype,
                        skip_if_exists = skip_if_exists)
                )

    if shuffle:
        gtiles = np.random.permutation(gtiles)
    if max_downloads is not None:
        gtiles = gtiles[:max_downloads]

    print (f"downloading {len(gtiles)} tiles. showing progress of {n_processes} parallel download jobs.", flush=True)                                    
    _gee_get_tile_progress_period = np.max([len(gtiles)//100,1])
    pool = multiprocessing.Pool(n_processes)
    pool.starmap(_get_tile, enumerate(gtiles))
    pool.close()

class GEETile:
    
    def __init__(self, 
                 tile_geometry, 
                 dataset_definition,
                 dest_dir=".", 
                 file_prefix="geetile_", 
                 meters_per_pixel=None, 
                 pixels_lonlat=None, 
                 identifier=None,
                 remove_saturated_or_null = False,
                 dtype = None,
                 skip_if_exists = True):
        """
        tile_geometry: shapely geometry in epsg 4326 lon/lat
        dest_dir: folder to store downloaded tile from GEE
        file_prefix: to name the tif file with the downloaded tile
        meters_per_pixel: an int, if set, the tile pixel size will be computed to match the requested meters per pixel
        pixels_lonlat: a tuple, if set, the tile will have this exact size in pixels, regardless the physical size.
        dataset_definition: a DatasetDefinition class
        remove_saturated_or_null: if true, will remove image if saturated or null > 1%
        """

        if sum([(meters_per_pixel is None), (pixels_lonlat is None)])!=1: 
            raise ValueError("must specify exactly one of meters_per_pixel or pixels_lonlat")
            
        self.tile_geometry = tile_geometry
        self.meters_per_pixel = meters_per_pixel
        self.dataset_definition = dataset_definition
        self.dest_dir = dest_dir
        self.file_prefix = file_prefix
        self.remove_saturated_or_null = remove_saturated_or_null
        self.dtype = dtype
        self.skip_if_exists = skip_if_exists


        if identifier is None:
            self.identifier = utils.get_region_hash(self.tile_geometry)
        else:
            self.identifier = identifier
                    
        if pixels_lonlat is not None:
            self.pixels_lon, self.pixels_lat = pixels_lonlat

    def get_filename(self):
        # check if should skip
        ext = 'tif'
        outdir = os.path.abspath(self.dest_dir)
        filename    = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}"
        msk_filename = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}.msk"

        return filename, msk_filename

    def get_tile(self):

        filename, msk_filename = self.get_filename()

        """
        # check if should skip
        ext = 'tif'
        outdir = os.path.abspath(self.dest_dir)
        filename    = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}"
        msk_filename = f"{outdir}/{self.file_prefix}{self.identifier}.{ext}.msk"

        """
        if self.skip_if_exists:
            if os.path.exists(filename):
                return

            if 'must_get_gee_image' in dir(self.dataset_definition) and\
               not self.dataset_definition.must_get_gee_image(filename):
                print ("skipping", filename)
                return

        # get appropriate utm crs for this tile_geometry to measure stuff in meters 
        lon, lat = list(self.tile_geometry.envelope.boundary.coords)[0]
        utm_crs = utils.get_utm_crs(lon, lat)
        self.tile_geometry_utm = gpd.GeoDataFrame({'geometry': [self.tile_geometry]}, crs = CRS.from_epsg(4326)).to_crs(utm_crs).geometry[0]

        # compute image pixel size if meters per pixels where specified
        if self.meters_per_pixel is not None:
            coords = np.r_[self.tile_geometry_utm.envelope.boundary.coords]
            self.pixels_lon, self.pixels_lat = np.ceil(np.abs(coords[1:] - coords[:-1]).max(axis=0) / self.meters_per_pixel).astype(int)

        # build image request
        dims = f"{self.pixels_lon}x{self.pixels_lat}"

        try:
            rectangle = ee.Geometry.Polygon(list(self.tile_geometry.boundary.coords)) 
        except:
            # in case multipolygon, or mutipart or other shapely geometries without a boundary
            rectangle = ee.Geometry.Polygon(list(self.tile_geometry.envelope.boundary.coords)) 

        image_collection = self.dataset_definition.get_gee_image(tile_geometry = self.tile_geometry)

        # if no image collection do nothing
        if image_collection is None:
            return


        try:
            url = image_collection.getDownloadURL(
                {
                    'region': rectangle,
                    'dimensions': dims,
                    'format': 'GEO_TIFF',
                    'crs': 'EPSG:4326'
                }
            )
        except Exception as e:
            # allow dataset to deal with errors
            if 'on_error' in dir(self.dataset_definition):
                self.dataset_definition.on_error(self, e)
                return
            else:
                raise(e)

        band_names = image_collection.bandNames().getInfo()
    
        # download and save to tiff
        r = requests.get(url, stream=True)

        if r.status_code != 200:
            r.raise_for_status()

        with open(filename, 'wb') as outfile:
            shutil.copyfileobj(r.raw, outfile)

        # reopen tiff to mask out tile_geometry, set image type and band names
        with rasterio.open(filename) as src:
            out_image, out_transform = rasterio.mask.mask(src, [self.tile_geometry], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        if self.dtype is not None:
            out_image = out_image.astype(self.dtype)
            out_meta['dtype'] = self.dtype

        if os.path.isfile(filename):
            os.remove(filename)

        with rasterio.open(filename, "w", **out_meta) as dest:
            dest.write(out_image)  
            for i in range(out_image.shape[0]):
                if self.dtype is not None:
                    dest.write_band(i+1, out_image[i]  )
                dest.set_band_description(i+1, band_names[i])

        # give the dataset a chance to do stuff with the downloaded tile
        if 'post_process_tilefile' in dir(self.dataset_definition):
            self.dataset_definition.post_process_tilefile(filename)

        # cleanup
        if os.path.isfile(msk_filename):
            os.remove(msk_filename)
