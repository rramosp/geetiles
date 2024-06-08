import ee
from geetiles import utils
import rasterio
import numpy as np
import os

class DatasetDefinition:

    """ 
    the gee collection applies the transformation 10 log_10 \sigma to the backscattering.
    this results in a range of values of approx [-30,0]

    this class transforms that range into [0,255] so that it can be stored as uint8 without
    loosing too much precision, and saving storage space.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        yearmonth = dataset_name.split("-")[-1]
        if len(yearmonth)!=6 :
            raise ValueError(f"dataset must be year and month, for instance '{dataset_name}-202201' for jan 2022")

        self.year = yearmonth[:4]
        self.month = yearmonth[4:]
        try:
            year = int(self.year)
            month = int(self.month)

            if month<1 or month>12:
                raise ValueError(f"dataset must be year and month, for instance '{dataset_name}-202201' for jan 2022")

        except Exception as e:
            raise ValueError(f"dataset must be year and month, for instance '{dataset_name}-202201' for jan 2022")

    def get_dataset_name(self):
        return self.dataset_name
    

    def must_get_gee_image(self, filename):
        """
        return true if needs to call get_gee_image
        """
        if os.path.exists(filename) or os.path.exists(filename+".nodata"):
            print ("skipping")
            return False
        return True

    def get_gee_image(self, tile_geometry, **kwargs):
        s1grd = None
        year = self.year

        endday = {'01': '31', '02': '28', '03': '31',
                  '04': '30', '05': '31', '06': '30',
                  '07': '31', '08': '31', '09': '30',
                  '10': '31', '11': '30', '12': '31'}

        start_date = f"{self.year}-{self.month}-01"
        end_date   = f"{self.year}-{self.month}-{endday[self.month]}"
        
        geom = ee.Geometry.Polygon(list(tile_geometry.boundary.coords))

        def get_s1_img(mode, direction):
            imgcol = ee.ImageCollection('COPERNICUS/S1_GRD')\
                        .filterDate(start_date, end_date)\
                        .filterBounds(geom) \
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', direction))\
                        .select([mode])

            imgcol_renamed = imgcol.map(lambda image: image.rename(ee.String(f'xxx_{mode}_{direction[:3]}_').cat(image.date().format("YYYY-MM-dd_HHmm"))))

            img_allbands = imgcol_renamed.toBands()
            # remove sentinel granule name
            img_allbands = img_allbands.regexpRename(".*xxx_(.*)", "$1")
            return img_allbands


        vv_asc = get_s1_img("VV", "ASCENDING")
        vv_des = get_s1_img("VV", "DESCENDING")
        vh_asc = get_s1_img("VH", "ASCENDING")
        vh_des = get_s1_img("VH", "DESCENDING")

        r = vv_asc.addBands(vv_des)\
                  .addBands(vh_asc)\
                  .addBands(vh_des)

        return r

    def map_values(self, array):
        return array

    def get_dtype(self):
        return 'float32'

    def post_process_tilefile(self, filename):
        # open raster again to adjust 
        with rasterio.open(filename) as src:
            x = src.read()

            # scale image
            a,b = -30,0
            x = (255*(x-a)/(b-a)).astype(np.uint8)
            
            m = src.read_masks()
            profile = src.profile.copy()
            band_names = src.descriptions

        # write image as uint8
        profile['dtype'] = 'uint8'
        with rasterio.open(filename, 'w', **profile) as dest:
            for i in range(src.count):
                dest.write(x[i,:,:], i+1)      
                dest.write_mask(m) 
                dest.set_band_description(i+1, band_names[i])

    def on_error(self, tile, exception):
        """
        what to do if there is an error while downloading the tile from gee
        tile: a gee.GEETile object
        exception: an Exception object
        """
        if isinstance(exception, ee.ee_exception.EEException):
            filename = tile.get_filename()[0] + ".nodata"
            with open(filename, "w") as f:
                f.write(str(exception))


