
import re
import os
import numpy as np
import geopandas as gpd
import shapely as sh
import pandas as pd
from skimage.io import imread

from pyproj import CRS
from joblib import delayed
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from progressbar import progressbar as pbar

from . import gee
from . import utils

epsg4326 = utils.epsg4326


class PartitionSet:
    
    def __init__(self, name, region=None, data=None):
        """
        name: a name for this partition
        region: a shapely shape in epsg 4326 lon/lat coords
        data: a geopandas dataframe with the partition list
        """
        
        assert region is not None or data is not None, "must specify either region or data"
        assert (region is not None) + (data is not None) == 1, "cannot specify both region and data"
        assert not "_" in name, "'name' cannot contain '_'"
        self.name   = name
        self.region = region
        self.region_utm = None
        self.data   = data
        
        if self.data is not None:
            if self.data.crs == CRS.from_epsg(4326):
                # convert to meters crs to measure areas
                lon,lat = np.r_[sh.geometry.GeometryCollection(self.data.geometry.values).envelope.boundary.coords].mean(axis=0)
                datam = self.data.to_crs(utils.get_utm_crs(lon, lat))
                self.data['area_km2'] = [i.area/1e6 for i in datam.geometry]
            else: 
                # if we are not in epsg 4326 assume we have a crs using meters           
                self.data['area_km2'] = [i.area/1e6 for i in self.data.geometry]

            self.data = self.data.to_crs(CRS.from_epsg(4326))
            self.data['identifier'] = [utils.get_region_hash(i) for i in self.data.geometry]
            
        self.loaded_from_file = False

    def compute_region_utm(self):
        """
        compute the region covered by this partitionset by joining all geometries in 
        the corresponding geopandas dataframe
        """

        if self.region_utm is not None:
            return self.region_utm
        
        if self.region is None:
            self.region = utils.get_boundary(self.data)
                        
        # corresponding UTM CRS in meters to this location
        lon, lat = np.r_[self.region.envelope.boundary.coords].mean(axis=0)

        self.utm_crs = utils.get_utm_crs(lon, lat)
        self.epsg4326 = CRS.from_epsg(4326)

        # the region in UTM CRS meters
        self.region_utm = gpd.GeoDataFrame({'geometry': [self.region]}, crs = self.epsg4326).to_crs(self.utm_crs).geometry[0]

    def reset_data(self):
        self.data = None
        self.loaded_from_file = False
        return self
        
    def make_random_partitions(self, max_rectangle_size, random_variance=0.1, n_jobs=5):
        """
        makes random rectangular tiles with max_rectangle_size as maximum side length expressed in meters.
        stores result as a geopandas dataframe in self.data
        """
        assert self.data is None, "cannot make partitions over existing data"
        
        self.compute_region_utm()

        # cut off region, assuming region_utm is expressed in meters
        parts = katana(self.region_utm, threshold=max_rectangle_size, random_variance=random_variance)

        # reproject to epsg 4326
        self.data = gpd.GeoDataFrame({
                                      'geometry': parts, 
                                      'area_km2': [i.area/1e6 for i in parts]
                                      },
                                    crs = self.utm_crs).to_crs(self.epsg4326)

        # align geometries to lonlat
        def f(part):
            try:
                aligned_part = utils.align_to_lonlat(part)
            except Exception as e:
                aligned_part = part
            return aligned_part

        parts = utils.mParallel(n_jobs=n_jobs, verbose=30)(delayed(f)(part) for part in self.data.geometry.values)
        self.data.geometry = parts

        self.data['identifier'] =  [utils.get_region_hash(i) for i in self.data.geometry]
        return self
        
    def make_grid(self, rectangle_size):

        """
        makes a grid of squares for self.region (which must be in epsg 4326 lon/lat)
        rectangle_size: side length in meters of the resulting squares
        stores result as a geopandas dataframe in self.data

        """
        assert self.data is None, "cannot make partitions over existing data"
        self.compute_region_utm()

        coords = np.r_[self.region_utm.envelope.boundary.coords]        
        m = rectangle_size

        minlon, minlat = coords.min(axis=0)
        maxlon, maxlat = coords.max(axis=0)
        parts = []
        for slon in pbar(np.arange(minlon, maxlon, m)):
            for slat in np.arange(minlat, maxlat, m):
                p = sh.geometry.Polygon([[slon, slat], 
                                         [slon, slat+m],
                                         [slon+m, slat+m],
                                         [slon+m, slat],
                                         [slon, slat]])

                if p.intersects(self.region_utm):
                    parts.append(p.intersection(self.region_utm))
                    
        self.data = gpd.GeoDataFrame({
                                      'geometry': parts, 
                                      'area_km2': [i.area/1e6 for i in parts]
                                      },
                                    crs = self.utm_crs).to_crs(self.epsg4326)
        self.data['identifier'] =  [utils.get_region_hash(i) for i in self.data.geometry]

        return self
    
    def get_downloaded_tiles_dest_dir(self, gee_image_name):
        if not 'origin_file' in dir(self):
            raise ValueError("must first save this partitions with 'save_as', or load an existing partitions file with 'from_file'")
        dest_dir = os.path.splitext(self.origin_file)[0]+ "/" + gee_image_name
        return dest_dir
        
    def download_gee_tiles(self,
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
        see gee.download_tiles
        it will use the same folder in which the partitions where saved to or loaded from as geojson.
        """

        dest_dir = self.get_downloaded_tiles_dest_dir(dataset_definition.get_dataset_name())
        print ("saving tiles to", dest_dir, flush=True)
         
        gee.download_tiles( self.data,
                            dest_dir,
                            dataset_definition = dataset_definition,
                            n_processes = n_processes, 
                            pixels_lonlat = pixels_lonlat, 
                            meters_per_pixel = meters_per_pixel,
                            remove_saturated_or_null = remove_saturated_or_null,
                            max_downloads = max_downloads, 
                            shuffle = shuffle,
                            skip_if_exists = skip_if_exists,
                            dtype = dtype)

    def get_partitions(self):
        """
        returns a list the partition objects of this partitionset 
        """
        r = [Partition(partitionset = self, 
                                        identifier = i.identifier, 
                                        geometry = i.geometry, 
                                        crs = self.data.crs) \
            for i in self.data.itertuples()]
        return r

    def save_as(self, dest_dir, partitions_name):
        """
        method used to save partitions that were just created by make_random or make_grid
        """
        if self.data is None:
            raise ValueError("there are no partitions. You must call make_random_partitions or make_grid first")

        if self.loaded_from_file:
            raise ValueError("cannot save partitions previously loaded. You must call reset_data first and create new different partitions")

        if "_" in partitions_name or "partitions" in partitions_name:
            raise ValueError("cannot have '_' or 'partitions' in partitions_name")

        h = utils.get_regionlist_hash(self.data.geometry)
        filename = f"{dest_dir}/{self.name}_partitions_{partitions_name}_{h}.geojson"
        self.data.to_file(filename, driver='GeoJSON')
        self.origin_file = filename
        print (f"saved to {filename}")
        self.partitions_name = partitions_name
        return self      

    def save(self):
        """
        method used to save partitions previously loaded (after adding some column or metadata)
        """
        computed_hash = utils.get_regionlist_hash(self.data.geometry)
        filename_hash = os.path.splitext(os.path.basename(self.origin_file))[0].split("_")[-1]
        if computed_hash != filename_hash:
            raise ValueError("cannot save since geometries changed, use save_as to create a new partition set")
        self.data.to_file(self.origin_file, driver='GeoJSON')
        print (f"saved to {self.origin_file}")



    def expand_proportions(self):
        """
        expands proportions into separate columns for easy visualization in GIS software
        """
        cols_proportions = [i for i in self.data.columns if "_proportions" in i]
        if len(cols_proportions)==0:
            print ("no proportions found in", self.origin_file)
            return
        
        for col in cols_proportions:
            #self.data = self.data[[c for c in self.data.columns if not c.startswith(f"{col}__")]]
            self.data = utils.expand_dict_column(self.data, col)
            
        f,ext = os.path.splitext(self.origin_file)
        expanded_fname = f'{f}_expanded{ext}'
        self.data.to_file(expanded_fname, driver='GeoJSON')
        print ("saved expanded file to", expanded_fname)

    def add_proportions(self, labels_dataset_def, n_jobs=5):
        """
        adds proportions from an image collection with the same geometry (such when this partitionset
        is an rgb image collection and image_collection_name contains segmentation masks)
        """
        def f(identifier, geometry):
            proportions = Partition(partitionset = self, 
                                    identifier = identifier, 
                                    geometry = geometry, 
                                    crs = self.data.crs).compute_proportions_from_raster(
                                                                labels_dataset_def
                                                            )
            return proportions
        
        if n_jobs==1:
            r = [f(i.identifier, i.geometry) for i in pbar(self.data.itertuples(), max_value=len(self.data))]
        else:
            r = utils.mParallel(n_jobs=n_jobs, verbose=30)(delayed(f)(i.identifier, i.geometry) for i in self.data.itertuples())
        self.data[f"{labels_dataset_def.get_dataset_name()}_proportions"] = r
        print()
        self.save()

    def add_foreign_proportions(self, label_dataset_definition, foreign_partitionset):
        """
        add class proportions of the geometries of this partitionset when embedded in a coarser partitionset.
        see Partition.compute_foreign_proportions below
        """
        image_collection_name = label_dataset_definition.get_dataset_name()
        parts = self.get_partitions()
        proportions = []
        foreign_ids = []
        for part in pbar(parts):
            foreign_proportions, foreign_identifier = part.compute_foreign_proportions(image_collection_name, foreign_partitionset)
            #proportions.append({'partition_id': foreign_identifier, 
            #                    'proportions': foreign_proportions})
            proportions.append(foreign_proportions)
            foreign_ids.append(foreign_identifier)
        colname = f"{image_collection_name}_proportions_at_{foreign_partitionset.partitions_name}"
        self.data[colname] = proportions
        foreign_name = foreign_partitionset.origin_file[foreign_partitionset.origin_file.find('partitions_')+11:].split("_")[0]
        #self.data[f'foreignid_{foreign_name}'] = [i['partition_id'] for i in self.data[colname]]
        self.data[f'foreignid_{foreign_name}'] = foreign_ids
        self.save()

    def add_foreign_partition(self, foreign_partitionset):
        """
        add the largest foreign intersection partition for each geometries of this partitionset when embedded in a coarser partitionset.
        see Partition.compute_foreign_partition below
        """
        parts = self.get_partitions()
        foreign_name = foreign_partitionset.origin_file[foreign_partitionset.origin_file.find('partitions_')+11:].split("_")[0]
        print (f"using foreign partition name '{foreign_name}'")
        self.data[f'foreignid_{foreign_name}'] = [part.compute_foreign_partition(foreign_partitionset) for part in pbar(parts)]
        self.save()

    def split(self, nbands, angle, train_pct, test_pct, val_pct, split_col_name='split'):        
        """
        splits the geometries in train, test, val by creating spatial bands
        and assigning each band to train, test or val according to the pcts specified.
        
        nbands: the number of bands
        angle: the angle with which bands are created (in [-pi/2, pi/2])
        train_pct, test_pct, val_pct: the pcts of bands for each kind,
                bands of the same kind are put together and alternating
                as much as possible.
        """
        if angle<-np.pi/2 or angle>np.pi/2:
            raise ValueError("angle must be between -pi/2 and pi/2")
            
        p = self
        coords = np.r_[[np.r_[i.envelope.boundary.coords].mean(axis=0) for i in p.data.geometry]]

        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        crng = cmax - cmin

        if not np.allclose(train_pct + test_pct + val_pct, 1, atol=1e-3):
            raise ValueError("percentages must add up to one")

        min_pct = np.min([i for i in [train_pct, test_pct, val_pct] if i!=0])
        bands_train = int(np.round(train_pct/min_pct,0))
        bands_test  = int(np.round(test_pct/min_pct,0))
        bands_val   = int(np.round(val_pct/min_pct,0))

        if bands_train + bands_test + bands_val > nbands:
            raise ValueError(f"not enough bands for specified percentages. increase nbands to at least {bands_train + bands_test + bands_val}")
        
        if np.abs(angle)<np.pi/4:
            plon, plat = np.abs(angle)/(np.pi/4), 1
        else:
            plon, plat = np.sign(angle), (np.pi/2-np.abs(angle))/(np.pi/4)
        
        ncoords = (coords - cmin)/crng

        if angle<0:
            ncoords = 1-ncoords
        
        # find the factor that matches the desired number of bands
        for k in np.linspace(0.1,50,10000):
            band_id = ((plon*ncoords[:,0] + plat*ncoords[:,1])/(k/nbands)).astype(int)
            band_id = band_id - np.min(band_id)
            if len(np.unique(band_id))==nbands:
                break

        bands_ids = np.sort(np.unique(band_id))

        splits = ['train']*bands_train + ['test']*bands_test + ['val']*bands_val
        splits = (splits * (len(bands_ids)//len(splits) + 1))[:len(bands_ids)]

        band_split_map = {band_id: split for band_id, split in zip(bands_ids, splits)}

        split = [band_split_map[i] for i in band_id]

        self.data[split_col_name] = split

        
    def split_per_partitions(self, nbands, angle, train_pct, test_pct, val_pct, other_partitions_id):
        """
        splits the geometries (as in 'split'), but modifies the result keeping together 
        in the same split all geometries within the same partition.
        
        must have previously run 'add_foreign_proportions'.
        """
        self.split(nbands=nbands, angle=angle, 
                  train_pct=train_pct, 
                  test_pct=test_pct, 
                  val_pct=val_pct, split_col_name='split')
        
        self.data[f'split_{other_partitions_id}'] = self.data.groupby(f'foreignid_{other_partitions_id}')[['split']]\
                                                             .transform(lambda x: pd.Series(x).value_counts().index[0])

        
    def save_splits(self):
        # save the split into a separate file for fast access
        fname = os.path.splitext(self.origin_file)[0] + "_splits.csv"
        splits_df = self.data[[c for c in self.data.columns if ('split' in c and c!='split_nb') or c=='identifier']]
        splits_df.to_csv(fname, index=False)
        self.save()
        print (f"all splits saved to {fname}")

    @classmethod
    def from_file(cls, filename, groups=None):
        data = gpd.read_file(filename)

        if groups is not None:
            if not 'group' in data.columns:
                raise ValueError(f"you specified groups {groups}, but there is no 'group' column in tiles_file")
            original_datalen = len(data)
            lgroups = groups.split(",")
            data = data[data.group.isin(lgroups)]
            print (f"\ndownloading only tiles in groups '{groups}', original data had {original_datalen} tiles, downloading {len(data)} tiles")


        if len(data)==0:
            return None
            
        r = cls("fromfile", data=data)
        r.origin_file = filename
        pname = re.search('_partitions_(.+?)_', filename)
        if pname is None:
            r.partitions_name = None
        else:
            r.partitions_name = pname.group(1)

        r.loaded_from_file = True

        return r          


class Partition:
    
    def __init__(self, partitionset, identifier, geometry, crs):
        self.identifier = identifier
        self.geometry = geometry
        self.crs = crs
        self.partitionset = partitionset
        self.partitionset_dir = os.path.splitext(self.partitionset.origin_file)[0]

    def get_tif(self, image_collection_name):
        basedir = self.partitionset_dir + "/" + image_collection_name
        filename = f"{basedir}/{self.identifier}.tif"
        img = imread(filename)
        return img
    
    def compute_proportions_from_raster(self, labels_dataset_def):

        # retrieve label array from disk
        image_collection_name = labels_dataset_def.get_dataset_name()
        basedir = self.partitionset_dir + "/" + image_collection_name
        filename = f"{basedir}/{self.identifier}.tif"
        img = imread(filename)

        # map image values according to dataset definition
        img = labels_dataset_def.map_values(img)

        # account for proportions only within the geometry 
        # (in case it does not match the rectangular image)
        mask = utils.get_binary_mask(self.geometry, img.shape)
        img = img[mask==1]

        # compute proportions
        r = {k:v for k,v in zip(*np.unique(img, return_counts=True))}        
        total = sum(r.values())
        r = {str(k):v/total for k,v in r.items()}

        return r
    
    def compute_foreign_partition(self, foreign_partition_set):
        """
        returns the id of the foreign partition (geometry) intersecting this one
        """        
        t = foreign_partition_set
        relevant = t.data[[i.intersects(self.geometry) for i in t.data.geometry.values]]
        w = np.r_[[self.geometry.intersection(i).area for i in relevant.geometry]]

        if len(relevant)>0:
            largest_foreign_partition_id = relevant.identifier.values[np.argmax(w)]
        else:
            largest_foreign_partition_id = -1

        return largest_foreign_partition_id

    def compute_foreign_proportions(self, image_collection_name, foreign_partition_set):
        """
        compute class proportions of this geometry when embedded in a coarser partitionset.
        class proportions are computed by (1) obtaining the intersecting partitions on the
        other partitionset, (2) combining the proportions in the intersecting partitions by
        weighting them according to the intersection area with this geometry
        
        returns: a list of proportions, the id of the geometry in "other_partitionset" with greater contribution
        """
        t = foreign_partition_set
        relevant = t.data[[i.intersects(self.geometry) for i in t.data.geometry.values]]

        # weight each higher grained geometry by % of intersection with this geometry
        w = np.r_[[self.geometry.intersection(i).area for i in relevant.geometry]]
        w = w / w.sum()
        foreign_proportions = dict ((pd.DataFrame(list(relevant[f"{image_collection_name}_proportions"].values)) * w.reshape(-1,1) ).sum(axis=0))

        if len(w)>0:
            largest_foreign_partition_id = relevant.identifier.values[np.argmax(w)]
        else:
            largest_foreign_partition_id = -1

        return foreign_proportions, largest_foreign_partition_id

    def compute_proportions_by_interesection(self, other_partitions):
        pass        


def katana(geometry, threshold, count=0, random_variance=0.1):
    """
    splits a polygon recursively into rectangles
    geometry: the geometry to split
    threshold: approximate size of rectangles
    random_variance: 0  - try to make all rectangles of the same size
                     >0 - the greater the number, the more different the rectangle sizes
                     values between 0 and 1 seem more useful
                     
    returns: a list of Polygon or MultyPolygon objects
    """
    
    
    """Split a Polygon into two parts across it's shortest dimension"""
    assert random_variance>=0

    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    random_factor = 2*(1+(np.random.random()-0.5)*random_variance*2)
    
    if max(width, height) <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/random_factor)
        b = box(bounds[0], bounds[1]+height/random_factor, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/random_factor, bounds[3])
        b = box(bounds[0]+width/random_factor, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon)):
                result.extend(katana(e, threshold, count+1, random_variance))
            if isinstance(e, (MultiPolygon)):
                for p in e.geoms:
                    result.extend(katana(p, threshold, count+1, random_variance))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result

# ---- old stuff ----

def flatten_geom(geom):
    
    """
    recursively converts a MultiPolygon into a list of shapely shapes
    geom: a shapely geometry
    returns: a list of geometries 
            (if 'geom' is not a multipart geometry it returns a list containing only geom)
    """
    
    if isinstance(geom, list):
        geoms = geom
    elif 'geoms' in dir(geom):
        geoms = geom.geoms
    else:
        return [geom]
        
    r = []
    for g in geoms:
        r.append(flatten_geom(g))
        
    r = [i for j in r for i in j]

    return r

def change_crs(shapes, to_crs, from_crs=CRS.from_epsg(4326)):
    """
    shapes: a shapely shape or a list of shapely shapes
    from_crs: pyproj CRS object representing the CRS in which geometries are expressed
    to_crs: pyproj CRS object with the target crs 

    returns: a GeometryCollection if 'shapes' is a shapely multi geometry 
             a list of shapes if 'shapes' is a list of shapely shapes
             a shapely shape if 'shapes' is a shapely shape
    """

    if 'geoms' in dir(shapes):
        r = gpd.GeoDataFrame({'geometry': list(shapes.geoms)}, crs=from_crs).to_crs(to_crs)        
    elif isinstance (shapes, list):
        r = gpd.GeoDataFrame({'geometry': shapes}, crs=from_crs).to_crs(to_crs)        
    else:
        r = gpd.GeoDataFrame([shapes], columns=['geometry'], crs=from_crs).to_crs(to_crs)        
        
    r = list(r.to_crs(to_crs).geometry.values)
    
    if 'geoms' in dir(shapes):
        r = sh.geometry.GeometryCollection(r)
    elif isinstance (shapes, list):
        pass
    else:
        r = r[0]

    return r
