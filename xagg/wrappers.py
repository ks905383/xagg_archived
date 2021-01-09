import warnings

from . core import (create_raster_polygons,get_pixel_overlaps)


def pixel_overlaps(ds,gdf_in,
                   weights=None,weights_target='ds',
                   subset_bbox = True):
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
    if subset_bbox:
        pix_agg = create_raster_polygons(ds,subset_bbox=gdf_in,weights=weights)
    else:
        pix_agg = create_raster_polygons(ds,subset_bbox=None,weights=weights)
    
    # Get overlaps between these pixel polygons and the gdf_in polygons
    print('calculating overlaps between pixels and output polygons...')
    wm_out = get_pixel_overlaps(gdf_in,pix_agg)
    print('success!')
    
    return wm_out


