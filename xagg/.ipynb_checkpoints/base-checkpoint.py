import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
import xesmf as xe

def normalize(a):
    if (np.all(~np.isnan(a))) & (a.sum()>0):
        return a/a.sum()
    else:
        return a*np.nan

def fix_ds(ds,var_cipher = {'latitude':{'latitude':'lat','longitude':'lon'},
                            'Latitude':{'Latitude':'lat','Longitude':'lon'},
                            'Lat':{'Lat':'lat','Lon':'lon'},
                            'latitude_1':{'latitude_1':'lat','longitude_1':'lon'},
                            'nav_lat':{'nav_lat':'lat','nav_lon':'lon'},
                            'Y':{'Y':'lat','X':'lon'},
                            'y':{'y':'lat','x':'lon'}},
          chg_bnds = True):
    """ Puts the input [ds] into a format compatible with the rest of the package
    
    Concretely, 
        1) grid variables are renamed "lat" and "lon"
        2) the lon dimension is made -180:180 to be consistent with most geographic
           data
    
    
    Keyword arguments:
    ds         -- an input xarray dataset, which may or may not need adjustment to be 
                  compatible with this package
    var_cipher -- a dict of dicts for renaming lat/lon variables to "lat"/"lon". The 
                  form is {[search_str]:{[lat_name]:'lat',[lon_name]:'lon'},...}, 
                  the code looks for [search_str] in the dimensions of the [ds]; 
                  based on that, it renames [lat_name] to 'lat' and [lon_name] to 'lon'.
                  Common names for these variables ('latitude', 'Latitude', 'Lat',
                 'latitude_1','nav_lat','Y') are included out of the box.
    chg_bnds   -- if True (by default), the names of variables with "_bnd" in their names
                  are assumed to be dimension bound variables, and are changed as well if
                  the rest of their name matches 'o' (for lon) or 'a' (for lat. 
                  
    Returns:
    a dataset with lat/lon variables in the format necessary for this package to function      
    """
    
    # List of variables that represent bounds
    if type(ds) is xr.core.dataset.Dataset:
        bnd_vars = [k for k in list(ds.keys()) if 'bnds' in k]
    elif type(ds) is xr.core.dataarray.DataArray:
        bnd_vars = []
    else:
        raise TypeError('[ds] needs to be an xarray structure (Dataset or DataArray).')
    
    # Fix lat/lon variable names (sizes instead of dims to be compatible with both ds, da...)
    if 'lat' not in ds.sizes.keys():
        test_dims = [k for k in var_cipher.keys() if k in ds.sizes.keys()]
        if len(test_dims) == 0:
            raise NameError('No valid lat/lon variables found in the dataset.')
        else:
            # shoudl there be a elif len()>1? If there are multiple things found?
            # Could be an issue in terms of ones where x/y are the coordinates of a 
            # non-rectangular grid with variables lat, lon, which is problematic,
            # or just duplicate dimension names, which is weird. 
            ds = ds.rename(var_cipher[test_dims[0]])
        
        # Now same for the bounds variable, if they exist
        if chg_bnds:
            if len(bnd_vars)>0:
                try:
                    ds = ds.rename({(key+'_bnds'):(value+'_bnds') for (key,value) in var_cipher[test_dims].items()})
                except ValueError:
                    try: 
                        warnings.warn('Assuming dim '+[k for k in bnd_vars if 'o' in k][0]+' is longitude bounds and '+
                                      ' dim '+[k for k in bnd_vars if 'a' in k][0] + ' is latitude bounds.')
                        ds = ds.rename({[k for k in bnd_vars if 'o' in k][0]:'lon_bnds',
                                        [k for k in bnd_vars if 'a' in k][0]:'lat_bnds'})
                    except:
                        warnings.warn('Could not identify which of the following bounds '+
                                      'variables corresponds to lat/lon grid: '+', '.join(bnd_vars)+
                                     '; no bound variables renamed.')
                        

    # Switch longitude to -180:180
    if ds.lon.max()>180:
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        
    # Do the same for a longitude bound, if present (the check for dataset is to 
    # avoid an error for .keys() for a dataarray; it seems like a good assumption 
    # that you're not running a dataarray of just lon_bnds through this). 
    if (type(ds) is xr.core.dataset.Dataset):
        # Three if statements because of what I believe to be a jupyter error
        # (where all three statements are evaluated instead of one at a time)
        if ('lon_bnds' in ds.keys()):
            if (ds.lon_bnds.max()>180):
                ds['lon_bnds'] = (ds['lon_bnds'] + 180) % 360 - 180

    # Sort by lon; this should be robust (and necessary to avoid fix_ds(fix_ds(ds)) 
    # from failing)
    ds = ds.sortby(ds.lon)
        
    # Return fixed ds
    return ds


def process_weights(ds,weights=None,target='ds'):
    """ Process weights - including regridding
    
    If target == 'ds', regrid weights to ds. If target == 'weights',
    regrid ds to weights. 
    
    Also needs an output that will go into gdf_out, with a flag for
    'something was regridded, y'all'
    
    ohhh... wait what if the pixel polygons have the weight in the 
    geodataframe... so this actually goes in the get_pixel_polygons
    
    """
    
    if weights is None:
        # (for robustness against running this without an extra if statement
        # in a wrapper function)
        weights_info = 'nowghts'
    else:
        # Check types
        if type(weights) is not xr.core.dataarray.DataArray:
            raise TypeError('[weights] must be an xarray DataArray.')
        if type(ds) not in [xr.core.dataarray.DataArray,
                                xr.core.dataset.Dataset]:
            raise TypeError('[ds] must be an xarray structure (DataArray or Dataset)')
            
        # Stick weights into the same supported input format as ds
        weights = fix_ds(weights)
        
        # Set regridding info
        weights_info = {'target':target,
                        'ds_grid':{'lat':ds.lat,'lon':ds.lon},
                        'weights_grid':{'lat':weights.lat,'lon':weights.lon}}


        if target is 'ds':
            print('regridding weights to data grid...')
            # Create regridder to the [ds] coordinates
            rgrd = xe.Regridder(weights,ds,'bilinear')
            # Regrid [weights] to [ds] grid
            weights = rgrd(weights)
            
        elif target is 'weights':
            print('regridding data to weights grid...')
            # Create regridder to the [weights] coordinates
            rgrd = xe.Regridder(ds,weights,'bilinear')
            # Regrid [ds] to [weights] grid
            ds = rgrd(ds)
            
        else:
            raise KeyError(target+' is not a supported target for regridding. Choose "weights" or "ds".')
            
        # Add weights to ds
        ds['weights'] = weights
            
    # Return
    return ds,weights_info



def get_bnds(ds,
             edges={'lat':[-90,90],'lon':[-180,180]},
             wrap_around_thresh=5):
    """ Builds vectors of lat/lon bounds if not present in [ds]
    
    Assumes a regular rectangular grid - so each lat/lon bound
    is 0.5 over to the next pixel.
    
    Note:
    Run this after [fix_lons], otherwise the dimension / variable
    names may not be correct
    
    Keyword arguments:
    ds                 -- an xarray dataset that may or may not 
                          contain variables "lat_bnds" and 
                          "lon_bnds"
    wrap_around_thresh -- the minimum distance between the last 
                          pixel edge and the 'edges' of the 
                          coordinate system for which the pixels
                          are 'wrapped around'. For example, given
                          'lon' edges of [-180,180] and a 
                          wrap_around_thresh of 5 (default), if 
                          the calculated edges of pixels match
                          the edge on one side, but not the other
                          (i.e. -180 and 179.4) and this gap 
                          (180-179.4) is less than 5, the -180 
                          edge is changed to 179.4 to allow the pixel
                          to 'wrap around' the edge of the coordinate
                          system.
          
    Returns
    the same dataset as inputted, unchanged if "lat/lon_bnds" 
    already existed, or with new variables "lat_bnds" and "lon_bnds"
    if not.
    """
    if ds.lon.max()>180:
        raise ValueError('Longitude seems to be in the 0:360 format.'+
                         ' -180:180 format required.')
        # Future versions should be able to work with 0:360 as well...
        # honestly, it *may* already work by just changing edges['lon']
        # to [0,360], but it's not tested yet. 
        
    if ('lat' not in ds.keys()) | ('lon' not in ds.keys()):
        raise KeyError('"lat"/"lon" not found in [ds]. Make sure the '+
                       'geographic dimensions follow this naming convention.')
    
    
    if 'lat_bnds' in ds.keys():
        return ds
    else:
        print('lat/lon bounds not found in dataset; they will be created.')
        # Build lat / lon bound 
        for var in ['lat','lon']:
            bnds_tmp = xr.DataArray(data=np.zeros((ds.dims[var],2))*np.nan,
                                    dims=[var,'bnds'],
                                    coords=[ds[var],np.arange(0,2)])

            # Assign all non-edge bounds as just half of the distance from the center
            # of each pixel to the center of the next pixel
            bnds_tmp[1:,:] = xr.concat([ds[var]-0.5*ds[var].diff(var),
                                          ds[var]+0.5*ds[var].diff(var)],dim='bnd').transpose(var,'bnd')

            # Fill in last missing band before edge cases (the inner band of the 
            # first pixel, which is just equal to the next edge)
            bnds_tmp[0,1] = bnds_tmp[1,0]

            # Now deal with edge cases; basically either use the diff from the last
            # interval between pixels, or max out at 90. 
            if ds[var].diff(var)[0]>0:
                bnds_tmp[0,0] = np.max([edges[var][0],ds[var][0].values-0.5*(ds[var][1]-ds[var][0]).values])
                bnds_tmp[-1,1] = np.min([edges[var][1],ds[var][-1].values+0.5*(ds[var][-1]-ds[var][-2]).values])
            else:
                bnds_tmp[0,0] = np.min([edges[var][1],ds[var][0].values+0.5*(ds[var][1]-ds[var][0]).values])
                bnds_tmp[-1,1] = np.max([edges[var][0],ds[var][-1].values-0.5*(ds[var][-1]-ds[var][-2]).values])

            # Fix crossing-over-360 issues in the lon
            if var is 'lon':
                # To be robust to partial grids; setting the rolled over edges equal to
                # each other if one of the edges is the -180/180 and the other one isn't, 
                # but 'close enough' (within 5 degrees + warning)
                if (bnds_tmp[0,0] in edges[var]) & (bnds_tmp[-1,1] not in edges[var]):
                    # Make sure that the other edge is within the wrap-around threshold
                    # (to avoid wrapping around if a grid is only -180:45 for example)
                    if np.min(np.abs(bnds_tmp[-1,1].values-edges[var])) <= wrap_around_thresh:
                        if np.min(np.abs(bnds_tmp[-1,1].values-edges[var])) > ds[var].diff(var).max():
                            warnings.warn('Wrapping around '+[var]+' value of '+bnds_tmp[-1,1].values+', '+
                                          'because it is closer to a coordinate edge ('+
                                          ', '.join([str(n) for n in edges[var]])+') than the '+
                                          '[wrap_around_thresh] ('+str(wrap_around_thresh)+'); '+
                                          'however, it is farther away from that edge than the '+
                                          'maximum pixel width in the '+var+' direction. If this is '+
                                          'intended, no further action is necessary. Otherwise, reduce '+
                                          'the [wrap_around_thresh].')
                        bnds_tmp[0,0] = bnds_tmp[-1,1]
                elif (bnds_tmp[0,0] not in edges[var]) & (bnds_tmp[-1,1] in edges[var]):
                    if np.min(np.abs(bnds_tmp[0,0].values-edges[var])) <= wrap_around_thresh:
                        if np.min(np.abs(bnds_tmp[0,0].values-edges[var])) > ds[var].diff(var).max():
                            warnings.warn('Wrapping around '+[var]+' value of '+bnds_tmp[0,0].values+', '+
                                          'because it is closer to a coordinate edge ('+
                                          ', '.join([str(n) for n in edges[var]])+') than the '+
                                          '[wrap_around_thresh] ('+str(wrap_around_thresh)+'); '+
                                          'however, it is farther away from that edge than the '+
                                          'maximum pixel width in the '+var+' direction. If this is '+
                                          'intended, no further action is necessary. Otherwise, adjust '+
                                          'the [wrap_around_thresh].')
                    bnds_tmp[-1,1] = bnds_tmp[0,0]
            # Add to ds
            ds[var+'_bnds'] = bnds_tmp
            del bnds_tmp
        
    # Return
    return ds    

        
def create_raster_polygons(ds,
                           mask=None,subset_bbox=None,
                           weights=None,weights_target='ds'):
    """ Create polygons for each pixel in a raster
    
    Keyword arguments:
    ds -- an xarray dataset with the variables 
          'lat_bnds' and 'lon_bnds', which are both
          lat/lon x 2 arrays giving the min and 
          max values of lat and lon for each pixel
          given by lat/lon
    subset_bbox -- by default None; if a geopandas
                   geodataframe is entered, the bounding
                   box around the geometries in the gdf 
                   are used to mask the grid, to reduce
                   the number of pixel polygons created
    mask -- ## THIS IS WHERE MASKS CAN BE ADDED - 
    # I.E. AN OCEAN MASK. OR MAYBE EVEN ALLOW 
    # SHAPEFILES TO BE ADDED AND CALCULATED
    # THE MASKED PIXELS ARE JUST IGNORED, AND NOT 
    # ADDED. will make identifying them harder
    # in the first bit of aggregate, but could 
    # make processing faster if you have a ton
    # of ocean pixels or something... 
          
    Returns:
    a geopandas geodataframe containing a 'geometry' 
    giving the pixel boundaries for each 'lat' / 'lon' 
    pair
                  
    Note: 
    'lat_bnds' and 'lon_bnds' can be created through the
    'get_bnds' function if they are not already included
    in the input raster file. 
    
    Note:
    Currently this code only supports regular 
    rectangular grids (so where every pixel side is
    a straight line in lat/lon space). Future versions
    may include support for irregular grids. 
    """
    
    # Standardize inputs
    ds = fix_ds(ds)
    ds = get_bnds(ds)
    #breakpoint()
    # Subset by shapefile bounding box, if desired
    if subset_bbox is not None:
        if type(subset_bbox) is gpd.geodataframe.GeoDataFrame:
            # Using the biggest difference in lat/lon to make sure that the pixels are subset
            # in a way that the bounding box is fully filled out
            bbox_thresh = np.max([ds.lat.diff('lat').max(),ds.lon.diff('lon').max()])+0.1
            ds = ds.sel(lon=slice(subset_bbox.total_bounds[0]-bbox_thresh,subset_bbox.total_bounds[2]+bbox_thresh),
                        lat=slice(subset_bbox.total_bounds[1]-bbox_thresh,subset_bbox.total_bounds[3]+bbox_thresh))
        else:
            warnings.warn('[subset_bbox] is not a geodataframe; no mask by polygon bounding box used.')
            
    # Process weights
    ds,winf = process_weights(ds,weights,target=weights_target)
            
    #breakpoint()
    # Mask
    if mask is not None:
        warnings.warn('Masking by grid not yet supported. Stay tuned...')
        
    # Create dataset which has a lat/lon bound value for each individual pixel, 
    # broadcasted out over each lat/lon pair
    (ds_bnds,) = (xr.broadcast(ds.isel({d:0 for d in [k for k in ds.dims.keys() if k not in ['lat','lon','bnds']]}).
                              drop_vars([v for v in ds.keys() if v not in ['lat_bnds','lon_bnds']])))
    # Stack so it's just pixels and bounds
    ds_bnds = ds_bnds.stack(loc=('lat','lon'))
    
    # In order:
    # (lon0,lat0),(lon0,lat1),(lon1,lat1),(lon1,lat1), but as a single array; to be 
    # put in the right format for Polygon in the next step
    pix_poly_coords = np.transpose(np.vstack([ds_bnds.lon_bnds.isel(bnds=0).values,ds_bnds.lat_bnds.isel(bnds=0).values,
                                                ds_bnds.lon_bnds.isel(bnds=0).values,ds_bnds.lat_bnds.isel(bnds=1).values,
                                                ds_bnds.lon_bnds.isel(bnds=1).values,ds_bnds.lat_bnds.isel(bnds=1).values,
                                                ds_bnds.lon_bnds.isel(bnds=1).values,ds_bnds.lat_bnds.isel(bnds=0).values]))
    
    # Reshape so each location has a 4 x 2 (vertex vs coordinate) array, 
    # and convert each of those vertices to tuples. This means every element
    # of pix_poly_coords is the input to shapely.geometry.Polygon of one pixel
    pix_poly_coords = tuple(map(tuple,np.reshape(pix_poly_coords,(np.shape(pix_poly_coords)[0],4,2))))
    
    # Create empty geodataframe
    gdf_pixels = gpd.GeoDataFrame()
    gdf_pixels['lat'] = [None]*ds_bnds.dims['loc']
    gdf_pixels['lon'] = [None]*ds_bnds.dims['loc']
    gdf_pixels['geometry'] = [None]*ds_bnds.dims['loc']
    if weights is not None:
        # Stack weights so they are linearly indexed like the ds (and fill
        # NAs with 0s)
        weights = ds.weights.stack(loc=('lat','lon')).fillna(0)
        # Preallocate weights column
        gdf_pixels['weights'] = [None]*ds_bnds.dims['loc']
    
    # Now populate with a polygon for every pixel, and the lat/lon coordinates
    # of that pixel (Try if preallocating it with the right dimensions above 
    # makes it faster, because it's pretty slow rn (NB: it doesn't really))
    for loc_idx in np.arange(0,ds_bnds.dims['loc']):
        gdf_pixels.loc[loc_idx,'lat'] = ds_bnds.lat.isel(loc=loc_idx).values
        gdf_pixels.loc[loc_idx,'lon'] = ds_bnds.lon.isel(loc=loc_idx).values
        gdf_pixels.loc[loc_idx,'geometry'] = Polygon(pix_poly_coords[loc_idx])
        if weights is not None:
            gdf_pixels.loc[loc_idx,'weights'] = weights.isel(loc=loc_idx).values
        
    # Add a "pixel idx" to make indexing better later
    gdf_pixels['pix_idx'] = gdf_pixels.index.values
    
    # Save the source grid for further reference
    source_grid = {'lat':ds_bnds.lat,'lon':ds_bnds.lon}
    
    pix_agg = {'gdf_pixels':gdf_pixels,'source_grid':source_grid}
    
    # Return the created geodataframe
    return pix_agg


def get_pixel_overlaps(gdf_in,pix_agg):
    """ Get, for each polygon, the pixels that overlap and their area of overlap
    
    Finds, for each polygon in gdf_in, which pixels intersect it, and by how much. 
    
    Note: 
    Uses WGS84 to calculate relative areas
    
    Keyword arguments:
    gdf_in     -- a GeoPandas GeoDataFrame giving the polygons over which 
                  the variables should be aggregated. Can be just a read
                  shapefile (with the added column of "poly_idx", which 
                  is just the index as a column).
    pix_agg    -- the output of [create_raster_polygons]; a dict containing:
                    'gdf_pixels': a GeoPandas GeoDataFrame giving for each row 
                                  the columns "lat" and "lon" (with coordinates) 
                                  and a polygon giving the boundary of the pixel 
                                  given by lat/lon
                    'source_grid':[da.lat,da.lon] of the grid used to create
                                   the pixel polygons
                  
    Returns:
    A dictionary containing: 
    ['agg']: a dataframe containing all the fields of [gdf_in] (except
             geometry) and the additional columns 
                coords:   the lat/lon coordiates of all pixels that overlap
                          the polygon of that row
                pix_idxs: the linear indices of those pixels within the 
                          gdf_pixels grid
                rel_area: the relative area of each of the overlaps between
                          the pixels and the polygon (summing to 1 - e.g. 
                          if the polygon is exactly the size and location of
                          two pixels, their rel_areas would be 0.5 each)
    ['source_grid']: a dictionary with keys 'lat' and 'lon' giving the 
                     original lat/lon grid whose overlaps with the polygons
                     was calculated
    ['geometry']: just the polygons from [gdf_in]                                        
    """
    
    
    # Add an index for each polygon as a column to make indexing easier
    if 'poly_idx' not in gdf_in.columns:
        gdf_in['poly_idx'] = gdf_in.index.values

    # Get GeoDataFrame of the overlaps between every pixel and the polygons
    overlaps = gpd.overlay(gdf_in,pix_agg['gdf_pixels'],how='intersection')
    
    # Now, group by poly_idx (each polygon in the shapefile)
    #(check if poly_idx exists in gdf_in first?)
    ov_groups = overlaps.groupby('poly_idx')
    
    # Calculate relative area of each overlap (so how much of the total 
    # area of each polygon is taken up by each pixel), the pixels 
    # corresponding to those areas, and the lat/lon coordinates of 
    # those pixels
    overlap_info = ov_groups.agg(rel_area=pd.NamedAgg(column='geometry',aggfunc=lambda ds: [ds.area/ds.area.sum()]),
                                  pix_idxs=pd.NamedAgg(column='pix_idx',aggfunc=lambda ds: [idx for idx in ds]),
                                  lat=pd.NamedAgg(column='lat',aggfunc=lambda ds: [x for x in ds]),
                                  lon=pd.NamedAgg(column='lon',aggfunc=lambda ds: [x for x in ds]))
    
    # Zip lat, lon columns into a list of (lat,lon) coordinates
    # (separate from above because as of 12/20, named aggs with 
    # multiple columns is still an open issue in the pandas github)
    overlap_info['coords'] = overlap_info.apply(lambda row: list(zip(row['lat'],row['lon'])),axis=1)
    overlap_info = overlap_info.drop(columns=['lat','lon'])
    
    # Reset index to make poly_idx a column for merging with gdf_in
    overlap_info = overlap_info.reset_index()
    
    # Merge in pixel overlaps to the input polygon geodataframe
    gdf_in = pd.merge(gdf_in,overlap_info,'inner')
    
    # Drop 'geometry' eventually, just for size/clarity
    gdf_out = {'agg':gdf_in.drop('geometry',axis=1),
               'source_grid':pix_agg['source_grid'],
               'geometry':gdf_in.geometry}
    
    if 'weights' in pix_agg['gdf_pixels'].columns:
        gdf_out['weights'] = pix_agg['gdf_pixels'].weights
    
    # Really, gdf_out should be its own class, with poly_idx, rel_area, 
    # pix_idxs, and coords required; and the rest of the information the 
    # stuff you keep
    
    return gdf_out


def pixel_overlaps(ds,gdf_in,
                   weights=None,weights_target='ds'):
    """ Wrapper function for determining overlaps between grid and polygon
    
    For a geodataframe [gdf_in], takes an xarray structure [ds] (Dataset or 
    DataArray) and for each polygon in [gdf_in] provides a list of pixels 
    given by the [ds] grid which overlap that polygon, in addition to their
    relative area of overlap with the polygon. 
    
    The output is then ready to be fed into [aggregate], which aggregates
    the variables in [ds] to the polygons in [gdf_in] using area- (and 
    optionally other) weights. 
    
    Keyword arguments:
    ds             -- an xarray Dataset or DataArray containing at least
                      grid variables ("lat"/"lon", though several other names
                      are supported; see docs for [fix_ds]) and at least 
                      one variable on that grid
    gdf_in         -- a geopandas GeoDataFrame containing polygons (and 
                      any other fields, for example fields from shapefiles)
    weights        -- (by default, None) if additional weights are desired,
                      (for example, weighting pixels by population in addition
                      to by area overlap), [weights] is an xarray DataArray 
                      containing that information. It does *not* have to 
                      be on the same grid as [ds] - grids will be homogonized
                      (see below). 
    weights_target -- if 'ds', then weights are regridded to the grid in [ds];
                      if 'weights', then the [ds] variables are regridded to
                      the grid in 'weights' (LATTER NOT FULLY SUPPORTED YET)
   
    (the wrapper also assumes [subset_bbox = True] in [create_raster_polygons])
    
    Output:
    [gdf_out], which gives the mapping of pixels to polygon aggregation; to be
    input into [aggregate]. 
    
    """
    
    # Create a polygon for each pixel
    print('creating polygons for each pixel...')
    pix_agg = create_raster_polygons(ds,subset_bbox=gdf_in,weights=weights)
    
    # Get overlaps between these pixel polygons and the gdf_in polygons
    print('calculating overlaps between pixels and output polygons...')
    gdf_out = get_pixel_overlaps(gdf_in,pix_agg)
    
    return gdf_out


def subset_find(ds0,ds1):
    """ Finds the grid of ds1 in ds0, and subsets ds0 to the grid in ds1
    
    Keyword arguments:
    ds0 -- an xarray Dataset to be subset based on the grid of ds1; must 
           contain grid variables "lat" or "lon" (could add a fix_ds call)
    ds1 -- either an xarray structrue (Dataset, DataArray) with "lat" "lon"
           variables, or a dictionary with DataArrays ['lat'] and ['lon'].
           IMPORTANT: DS1 HAS TO BE BROADCAST - i.e. one value of lat, lon 
           each coordinate, with lat and lon vectors of equal length. 
        
    
    """
    
    if 'loc' not in ds0.sizes.keys():
        ds0 = ds0.stack(loc = ('lat','lon'))
        was_stacked = True
    else:
        was_stacked = False
    #if 'loc' not in ds1.sizes.keys():
    #    ds1 = ds1.stack(loc = ('lat','lon'))
    
    # Need a test to make sure the grid is the same. So maybe the gdf_out class 
    # has the lat/lon grid included - and then we can skip the lat/lon column
    # and just keep the pix_idxs
    if ((len(ds0.lat) is not len(ds1['lat'])) or (len(ds0.lon) is not len(ds1['lon'])) or
         not (np.allclose(ds1['lat'],ds0.lat)) or not (np.allclose(ds1['lon'],ds0.lon))):
        print('adjusting grid... (this may happen because only a subset of pixels '+
              'were used for aggregation for efficiency - i.e. [subset_bbox=True] in '+
             '[create_raster_polygons])') #(this also happens because ds and ds_bnds above was already subset)
        # Zip up lat,lon pairs to allow comparison
        latlons = list(zip(ds0.lat.values,ds0.lon.values))
        latlons0 = list(zip(ds1['lat'].values,ds1['lon'].values))
        
        # Find indices of the used grid for aggregation in the input grid
        loc_idxs = [latlons.index(i) for i in latlons0]
        
        if np.allclose(len(loc_idxs),len(latlons0)):
            print('grid adjustment successful')
            # Subset by those indices
            ds0 = ds0.isel(loc=loc_idxs)
        else:
            raise ValueError('Was not able to match grids!')
        
    if was_stacked:
        # THIS MAY NOT WORK IN ALL CASES
        ds0 = ds0.unstack()
        
    return ds0


def aggregate(ds,gdf_out):
    """ Aggregate raster variable(s) to polygon(s)
    
    Aggregates (N-D) raster variables in [ds] to the polygons
    in [gfd_out] - in other words, gives the weighted average
    of the values in [ds] based on each pixel's relative area
    overlap with the polygons. 
    
    The values will be additionally weighted if a [weight] was
    inputted into [create_raster_polygons]
    
    The code checks whether the input lat/lon grid in [ds] is 
    equivalent to the linearly indexed grid in [gdf_out], or 
    if it can be cropped to that grid. 
    
    
    Keyword arguments:
    ds      -- an xarray dataset containing one or more
               variables with dimensions lat, lon (and possibly
               more). The dataset's geographic grid has to 
               include the lat/lon coordinates used in 
               determining the pixel overlaps in 
               [get_pixel_overlaps] (and saved in 
               gdf_out['source_grid'])
               
    gdf_out -- the output to [get_pixel_overlaps]; a 
               dict containing ['agg'] - a dataframe, with one 
               row per polygon, and the columns pix_idxs 
               and rel_area, giving the linear indices and 
               the relative area of each pixel over the polygon,
               respectively; and ['source_grid'] describing 
               the lat/lon grid on which the aggregating parameters
               were calculated (and on which the linear indices 
               are based)
               
    Returns:
    [gdf_out], with added columns for each variable in [ds] that's
    aggregated, giving the aggregated values (in a list) over each
    row's polygon
    
    """

    # Stack 
    ds = ds.stack(loc=('lat','lon'))
    
    # Adjust grid of [ds] if necessary to match 
    ds = subset_find(ds,gdf_out['source_grid'])
    
    # Set weights; or replace with ones if no additional weight information
    if 'weights' in gdf_out.keys():
        weights = np.array([float(k) for k in gdf_out['weights']])
    else:
        weights = np.ones((len(gdf_out['source_grid']['lat']),1))
        
    for var in ds.var():
        # Process for every variable that has locational information, but isn't a 
        # bound variable
        if ('bnds' not in ds[var].dims) & ('loc' in ds[var].dims):
            print('aggregating '+var+'...')
            # Create the column for the relevant variable
            gdf_out['agg'][var] = None

            # Get weighted average of variable based on pixel overlap + other weights
            for poly_idx in gdf_out['agg'].poly_idx:
                # Get average value of variable over the polygon; weighted by 
                # how much of the pixel area is in the polygon, and by (optionally)
                # a separate gridded weight
                gdf_out['agg'].loc[poly_idx,var] = [[((ds[var].isel(loc=gdf_out['agg'].iloc[poly_idx,:].pix_idxs)*
                                                       normalize(gdf_out['agg'].iloc[poly_idx,:].rel_area*
                                                                 np.transpose(weights[gdf_out['agg'].iloc[poly_idx,:].pix_idxs]))).
                                                      sum('loc')).values]]
                
    # Return
    print('all variables aggregated to polygons!')
    return gdf_out


def prep_for_nc(gdf_out,ds,loc_dim='poly_idx'):
    """ Preps aggregated data for output as a netcdf
    
    Concretely, aggregated data is placed in a new xarray dataset 
    with dimensions of location (the different polygons in gdf_out)
    and any other dimension(s) in the original input raster data. 
    All fields from the input polygons are kept as variables with 
    dimension of location.
    
    Keyword arguments:
    gdf_out -- the output from [.....]
    ds      -- an xarray dataset containing the variables aggregated
               to gdf_out
    loc_dim -- the name of the location dimension; by definition 
               'poly_idx'. Values of that dimension are currently
               only an integer index (with further information given
               by the field variables). Future versions may allow, 
               if loc_dim is set to the name of a field in the input
               polygons, to replace the dimension with the values of
               that field (however, this may cause issues when 
               exporting to netcdf?)
               
    Returns:
    an xarray dataset containing the aggregated variables in addition
    to the original fields from the location polygons. Dimensions are
    a location dimension (counting down the polygons - this is the 
    dimension of all the field contents) and any other non-location
    dimensions contained in the variables before being aggregated
    """
    # To output as netcdf, first put data back into an xarray dataset
    
    # Create xarray dataset with the aggregation polygons (poly_idx) 
    # there. 
    ds_out = xr.Dataset(coords={'poly_idx':(['poly_idx'],gdf_out['agg'].poly_idx.values)})
    
    # Add other polygon attributes
    for var in [c for c in gdf_out['agg'].columns if c not in ['poly_idx','rel_area','pix_idxs','coords']]:
        if var not in ds.var():
            # For auxiliary variables (from the shapefile), just copy them wholesale into the dataset
            ds_out[var] = xr.DataArray(data=gdf_out['agg'][var],coords=[gdf_out['agg'].poly_idx],dims=['poly_idx'])
        else:
            # For data variables (from the input grid), create empty array
            ds_out[var] = xr.DataArray(data=np.zeros((len(gdf_out['agg']),
                                                         *[ds[var].sizes[k] for k in ds[var].sizes.keys() if k not in ['lat','lon','loc']]))*np.nan,
                                       dims=['poly_idx',*[k for k in ds[var].sizes.keys() if k not in ['lat','lon','loc']]],
                                       coords=[[k for k in gdf_out['agg'].poly_idx],*[ds[var][k].values for k in ds[var].sizes.keys() if k not in ['lat','lon','loc']]])
        
            # Now insert aggregated values 
            for poly_idx in gdf_out['agg'].poly_idx:
                ds_out[var].loc[{'poly_idx':poly_idx}] = gdf_out['agg'].loc[poly_idx,var][0]
    
    # Add non-geographic coordinates for the variables to be aggregated
    for crd in [k for k in ds.sizes.keys() if (k not in ['lat','lon','loc','bnds'])]:
        ds_out[crd] = xr.DataArray(dims=[crd],data=ds[crd].values,coords=[ds[crd].values])
        
        
    # Rename poly_idx if desired
    if loc_dim is not 'poly_idx':
        ds_out = ds_out.rename({'poly_idx':loc_dim})
        
    # Return ds_out
    return ds_out


def prep_for_csv(gdf_out,ds):
    """ Preps aggregated data for output as a netcdf
    
    Concretely, aggregated data is placed in a new pandas dataframe
    and expanded wide - each aggregated variable is placed in new 
    columns; one column per coordinate in each dimension that isn't
    the location (poolygon). So, for example, a lat x lon x time
    variable "tas", aggregated to location x time, would be reshaped 
    long to columns "tas0", "tas1", "tas2",... for timestep 0, 1, etc.
    
    Note: 
    Currently no support for variables with more than one extra dimension
    beyond their location dimensions. Potential options: a multi-index
    column name, so [var]0-0, [var]0-1, etc...
    
    Keyword arguments:
    gdf_out -- the output from [.....]
    ds      -- an xarray dataset containing the variables aggregated
               to gdf_out
               
    Returns:
    a pandas dataframe containing all the fields from the original 
    location polygons + columns containing the values of the aggregated
    variables at each location. This can then easily be exported as a 
    csv directly (using .to_csv) or to shapefiles by first turning into
    a geodataframe. 
    
    """
    # For output into csv, work with existing geopandas data frame
    csv_out = gdf_out['agg'].drop(columns=['rel_area','pix_idxs','coords','poly_idx'])
    
    # Now expand the aggregated variable into multiple columns
    for var in [c for c in gdf_out['agg'].columns if ((c not in ['poly_idx','rel_area','pix_idxs','coords']) & (c in ds.var()))]:
        # NOT YET IMPLEMENTED: dynamic column naming - so if it recognizes 
        # it as a date, then instead of doing var0, var1, var2,... it does
        # varYYYYMMDD etc.
        # These are the coordinates of the variable in the original raster
        dimsteps = [ds[var][d].values for d in ds[var].sizes.keys() if d not in ['lat','lon','loc']]
        # ALSO SHOULD check to see if the variables are multi-D - if they are
        # there are two options: 
        # - a multi-index column title (var0-0, var0-1)
        # - or an error saying csv output is not supported for this
    
        # Reshape the variable wide and name the columns [var]0, [var]1,...
        expanded_var = (pd.DataFrame(pd.DataFrame(csv_out[var].to_list())[0].to_list(),
                                     columns=[var+str(idx) for idx in np.arange(0,len(csv_out[var][0][0]))]))
        # Append to existing series
        csv_out = pd.concat([csv_out.drop(columns=(var)),
                             expanded_var],
                            axis=1)
        del expanded_var
        
    # Return 
    return csv_out


def output_data(gdf_out,ds,output_format,output_fn,loc_dim='poly_idx'):
    """ Wrapper for prep_for_* functions
    
    
    Keyword arguments:
    gdf_out       -- 
    ds            --
    output_format -- supported: 'netcdf', 'csv', or 'shp'
    output_fn     -- the output filename, as a string   
    loc_dim       -- by default "poly_idx"; the name of the
                     dimension with location indices; used
                     only by prep_for_nc (see that function 
                     for more info)
                     
    Returns
    the variable that gets saved, so depending on the 
    [output_format]:
        "netcdf" - the xarray dataset on which [.to_netcdf] 
                   was called
        "csv"    - the pandas dataframe on which [.to_csv] 
                   was called
        "shp"    - the geopandas geodataframe on which 
                   [.to_file] was called
     
    """

    if output_format is 'netcdf':

        ds_out = prep_for_nc(gdf_out,ds,loc_dim=loc_dim)

        # Save as netcdf
        if not output_fn.endswith('.nc'):
            output_fn = output_fn+'.nc'
        ds_out.to_netcdf(output_fn)
        print(output_fn+' saved!')

        # Return
        return ds_out

    elif output_format is 'csv':

        csv_out = prep_for_csv(gdf_out,ds)

        # Save as csv
        if not output_fn.endswith('.csv'):
            output_fn = output_fn+'.csv'
        csv_out.to_csv(output_fn)
        print(output_fn+' saved!')

        # Return 
        return csv_out

    elif output_format is 'shp':
        # This uses the same processing as the to_csv option above, since 
        # shapefiles function similarly (each field can only have one value
        # for each polygon, etc.)
        csv_out = prep_for_csv(gdf_out,ds)

        # gdf_in.geometry should be replaced with gdf_out['geometry']
        # which should be kept when gdf_out is created... 
        # Turn back into GeoDataFrame
        shp_out = gpd.GeoDataFrame(csv_out,geometry=gdf_in.geometry)

        # Export that GeoDataFrame to shapefile
        if not output_fn.endswith('.shp'):
            output_fn = output_fn+'.shp'
        shp_out.to_file(output_fn)
        print(output_fn+' saved!')

        # Return
        return shp_out

    else: 
        raise KeyError(output_format+' is not a supported output format.')