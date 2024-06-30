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

        name_split = dataset_name.split("-")
        if len(name_split) != 2:
            raise ValueError(f"dataset must be year, month, for instance 's1grdobs-202201' for jan 2022")

        yearmonth = name_split[1]
        if len(yearmonth)!=6 :
            raise ValueError(f"dataset must be year and month, for instance 's1grdobs-202201' for jan 2022")

        self.year = yearmonth[:4]
        self.month = yearmonth[4:]

        try:
            _year = int(self.year)
            _month = int(self.month)

            if _month<1 or _month>12:
                raise ValueError(f"invalid month {_month}. dataset must be year and month, for instance 's1grdobs-202201' for jan 2022")

        except Exception as e:
            raise ValueError(f"dataset must be year and month, for instance 's1grdobs-202201' for jan 2022")


        # self.band0 = 'VV'
        # self.band1 = 'VH'
        # self.band2 = 'angle'
        #self.bands = ['VV', 'VH', 'HH', 'HV', 'angle']
        self.bands = ['VV', 'VH', 'angle']

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


    def get_s1_img(self, tile_geometry, direction, year, month, day):
        start_date = f"{self.year}-{self.month}-{day:02d}T00:00:00"
        end_date   = f"{self.year}-{self.month}-{day:02d}T23:59:59"
        
        geom = ee.Geometry.Polygon(list(tile_geometry.boundary.coords))

        imgcol = ee.ImageCollection('COPERNICUS/S1_GRD')\
                    .filterDate(start_date, end_date)\
                    .filterBounds(geom) \
                    .filter(ee.Filter.eq('orbitProperties_pass', direction))\
                    .select(self.bands)

        imgcol_renamed = imgcol

        imgcol_renamed = imgcol.map(lambda image: 
                                        image.rename([ee.String(f'xxx_{year}-{month}-{day:02d}_{direction[:3]}_{self.bands[0]}'),
                                                        ee.String(f'xxx_{year}-{month}-{day:02d}_{direction[:3]}_{self.bands[1]}'),
                                                        ee.String(f'xxx_{year}-{month}-{day:02d}_{direction[:3]}_{self.bands[2]}')]))

        return imgcol_renamed.max()

    def get_gee_image(self, tile_geometry, **kwargs):
        s1grd = None
        year = self.year
        endday = {'01': 31, '02': 28, '03': 31,
                  '04': 30, '05': 31, '06': 30,
                  '07': 31, '08': 31, '09': 30,
                  '10': 31, '11': 30, '12': 31}


        def prev_yearmonth(year, month):
            if int(month)==1:
                return f"{int(year)-1:4d}", "12"
            else:
                return year, f"{int(month)-1:02d}"

        def next_yearmonth(year, month):
            if int(month)==12:
                return f"{int(year)+1:4d}", "01"
            else:
                return year, f"{int(month)+1:02d}"                  
        
        ny, nm = next_yearmonth(self.year, self.month)
        py, pm = prev_yearmonth(self.year, self.month)

        imgs =   [self.get_s1_img(tile_geometry, 'ASCENDING', py, pm, day) for day in range(1, endday[pm]+1)]
        imgs +=  [self.get_s1_img(tile_geometry, 'DESCENDING', py, pm, day) for day in range(1, endday[pm]+1)]
        imgs +=  [self.get_s1_img(tile_geometry, 'ASCENDING', self.year, self.month, day) for day in range(1, endday[self.month]+1)]
        imgs +=  [self.get_s1_img(tile_geometry, 'DESCENDING', self.year, self.month, day) for day in range(1, endday[self.month]+1)]
        imgs +=  [self.get_s1_img(tile_geometry, 'ASCENDING', ny, nm, day) for day in range(1, endday[nm]+1)]
        imgs +=  [self.get_s1_img(tile_geometry, 'DESCENDING', ny, nm, day) for day in range(1, endday[nm]+1)]

        collection = ee.ImageCollection.fromImages(imgs)         

        # flatten the bands
        r = collection.toBands()

        # remove prefixes added from toBands
        r = r.regexpRename(".*xxx_(.*)", "$1")

        return r

    def map_values(self, array):
        return array

    def get_dtype(self):
        return 'float32'

    def post_process_tilefile(self, filename):
        # open raster again to adjust 
        return
        with rasterio.open(filename) as src:
            x = src.read()
            band_names = src.descriptions

            # scale image values
            _x = []
            for i in range(len(x)):
                if 'angle' in band_names[i]:
                    # for angle just convert to uint8 
                    _x.append(x[i].astype(np.uint8))
                    
                else:
                    # for amplitude convert [-30,0] to [0,255]
                    a,b = -30,0
                    _xi = (255*(x[i]-a)/(b-a))
                    _xi[_xi<0]   = 0
                    _xi[_xi>255] = 255
                    _x.append(_xi.astype(np.uint8))
            

            x = np.r_[_x].astype(np.uint8)
            m = src.read_masks()
            profile = src.profile.copy()

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


