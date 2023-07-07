import ee
import rasterio
import os
import numpy as np
from datetime import datetime
import shapely as sh
import json
from pathlib import Path
class DatasetDefinition:

    def __init__(self, dataset_name):
        dataset_name_components = dataset_name.split("-")
        if len(dataset_name_components)!=2:
            raise ValueError("incorrect dataset name. must be 'globalfloods-2020' or the year you want")
        
        self.year = dataset_name_components[1]
        self.dataset_name = dataset_name
        try:
            year = int(self.year)
        except Exception as e:
            raise ValueError(f"could not find year in {dataset_name}")
                
    def build(self):
        self.get_floods_metadata()

    def get_dataset_name(self):
        return self.dataset_name
    

    def must_get_gee_image(self, filename):
        """
        return true if needs to call get_gee_image
        """
        if os.path.exists(filename) or os.path.exists(filename+".nodata"):
            return False
        return True

    def get_floods_metadata(self):
        if not 'floods_metadata' in dir(self.__class__):
            print ("reading metadata for all floods")

            gfd = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1');
            
            # there will never bee 10k floods
            floods_metadata = gfd.toList(count=10000).getInfo()
            
            # augment properties with geometry object    
            for i in floods_metadata:
                i['properties']['geometry'] = sh.geometry.Polygon(i['properties']['system:footprint']['coordinates'])

                # creates proporties with dates converted to string
                start_date = datetime.fromtimestamp(ee.Date(i['properties']['system:time_start']).args['value']//1000)
                end_date = datetime.fromtimestamp(ee.Date(i['properties']['system:time_end']).args['value']//1000)
                i['properties']['system:time_start_str'] = start_date.strftime("%Y-%m-%d")
                i['properties']['system:time_end_str'] = end_date.strftime("%Y-%m-%d")
                
            self.__class__.floods_metadata = floods_metadata

        return self.__class__.floods_metadata

    def floods_in_tile(self, tile_geometry):
        """
        filters all floods for this year and this tile
        """

        from_date = f'{self.year}-01-01'
        to_date   = f'{self.year}-12-31'

        tile_floods = []

        all_floods = self.get_floods_metadata()

        for flood in all_floods:
            if flood['properties']['geometry'].intersects(tile_geometry):
                start_date = flood['properties']['system:time_start_str']
                end_date   = flood['properties']['system:time_end_str']
                if start_date>=from_date and end_date<=to_date:
                    tile_floods.append(flood)

        return tile_floods


    def get_gee_image(self, tile_geometry):
        gfd = ee.ImageCollection('GLOBAL_FLOOD_DB/MODIS_EVENTS/V1')
    
        # consider only the floods in this tile
        tile_floods = self.floods_in_tile(tile_geometry)
        if len(tile_floods)==0:
            return None

        bands = ['flooded', 'duration',  'clear_views', 'clear_perc', 'jrc_perm_water']
        aggregated_image = None
   
        # create a single image with the bands of all floods, prepended with the flood id
        for tf in tile_floods:
            
            tid = tf['properties']['id']
            
            img = ee.Image(gfd.filterMetadata('id', 'equals', tid).first())
            img = img.select(bands).rename([f'{tid}_{band}' for band in bands])
            if aggregated_image is None:
                aggregated_image = img.multiply(1) # removes all properties
            else:
                aggregated_image = aggregated_image.addBands(img)

        return aggregated_image
    
    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint16'
    
    def post_process_tilefile(self, filename):
        """
        removes floods with no pixels on this tile and adds flood metadata for the remaining floods
        """
        with rasterio.open(filename) as src:
            x = src.read()
            profile = src.profile.copy()
            band_names = src.descriptions

        # ids of floods with some pixel set to 1
        idxs = {band_names[i]:i for i in range(len(band_names))}
        flooded = [i for i in band_names if i.endswith("_flooded")]
        flooded_ids = [i.split("_")[0] for i in flooded if x[idxs[i]].sum()>0]

        # keep only the bands with those floods
        metadata = self.get_floods_metadata()
        xx = np.r_[[x[i] for i in range(len(band_names)) if sum([band_names[i].startswith(fid) for fid in flooded_ids])>0]]
        xband_names = [band_names[i] for i in range(len(band_names)) if sum([band_names[i].startswith(fid) for fid in flooded_ids])>0]
        props = {str(i['properties']['id']): i['properties'] for i in metadata if str(i['properties']['id']) in flooded_ids}
        sprops = {k:json.dumps({kk:vv for kk,vv in v.items() if not kk in ['geometry']}) for k,v in props.items()}

        # remove file, it will be recreated with the appropriate bands in next step
        os.remove(filename)

        # if we still have anything save it
        if len(xband_names)>0:
            print ("keeping", filename)
            with rasterio.open(filename, 'w', **profile) as dest:
                for i in range(len(xx)):
                    dest.write(xx[i,:,:], i+1)      
                    dest.set_band_description(i+1, xband_names[i])

                dest.update_tags(**sprops)
        else:
            # otherwise signal that it is empty
            Path(filename+".nodata").touch()
            pass
