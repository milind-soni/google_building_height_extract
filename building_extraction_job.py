@fused.udf
def prepare_thane_tiles():
    """Generate list of tiles covering Thane region."""
    import pandas as pd
    from shapely import box
    
    # Define Thane bounds (approximately covering Thane city)
    THANE_BOUNDS = {
        'north': 19.33,  # Northern boundary near Ghodbunder Road
        'south': 19.15,  # Southern boundary near Mulund
        'east': 73.05,   # Eastern boundary beyond Thane city
        'west': 72.93    # Western boundary near Thane creek
    }
    
    def get_tiles(bounds, tile_size=0.02):  # Smaller tile size for more granular processing
        """Split bounding box into tiles."""
        minx, miny, maxx, maxy = bounds
        
        tiles = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                tile_minx = x
                tile_miny = y
                tile_maxx = min(x + tile_size, maxx)
                tile_maxy = min(y + tile_size, maxy)
                
                tiles.append((float(tile_minx), float(tile_miny), 
                            float(tile_maxx), float(tile_maxy)))
                y += tile_size
            x += tile_size
        
        return tiles
    
    # Generate tiles
    tiles = get_tiles([
        THANE_BOUNDS['west'], 
        THANE_BOUNDS['south'],
        THANE_BOUNDS['east'],
        THANE_BOUNDS['north']
    ])
    
    print(f"Generated {len(tiles)} tiles")
    return pd.DataFrame(tiles, columns=['minx', 'miny', 'maxx', 'maxy'])

@fused.udf
def process_thane_buildings(tile_info):
    """Process buildings for a single tile."""
    import math
    import os
    import ee
    import json 
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely import box
    import io
    import requests
    from datetime import datetime
    utils = fused.load("https://github.com/fusedio/udfs/tree/be3bc93/public/common/").utils

    def generate_service_account_info():
        service_account_info = {
          "type": "service_account",
          "project_id": "ee-milindsoni",
          "universe_domain": "googleapis.com"
        }
        return service_account_info

    OUTPUT_PATH = "s3://fused-users/milind/thane_buildings_with_timestamps/"
    
    def save_tile_data(gdf, tile_bounds):
        """Save processed tile data with all timestamps to S3."""
        try:
            # Convert to pandas DataFrame with WKT geometry
            df = pd.DataFrame(gdf.drop(columns=['geometry']))
            df['geometry'] = gdf.geometry.apply(lambda x: x.wkt)
            
            # Add tile metadata
            minx, miny, maxx, maxy = tile_bounds
            df['tile_bounds'] = f"{minx},{miny},{maxx},{maxy}"
            df['processed_at'] = datetime.now().isoformat()
            
            # Generate filename
            tile_id = f"{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}"
            filename = f"buildings_tile_{tile_id}.parquet"
            full_path = os.path.join(OUTPUT_PATH, filename)
            
            # Save to S3
            df.to_parquet(full_path, index=False)
            print(f"Saved tile data with multiple timestamps to {full_path}")
            return True
        except Exception as e:
            print(f"Error saving tile data: {e}")
            print(f"Attempted path: {full_path}")
            return False

    try:
        # Extract tile bounds
        tile_bounds = (
            float(tile_info['minx']), 
            float(tile_info['miny']),
            float(tile_info['maxx']), 
            float(tile_info['maxy'])
        )
        
        # Create tile geometry
        tile_polygon = box(*tile_bounds)
        tile_gdf = gpd.GeoDataFrame(geometry=[tile_polygon], crs="EPSG:4326")
        
        print(f"Processing tile with bounds: {tile_bounds}")
        
        # Get buildings for tile
        overture_buildings = fused.utils.Overture_Maps_Example.get_overture(
            bbox=tile_gdf, 
            min_zoom=10
        )
        
        if overture_buildings is None or len(overture_buildings) < 1:
            print(f"No buildings found in tile {tile_bounds}")
            return None
            
        print(f"Found {len(overture_buildings)} buildings")
        
        # Calculate base area in UTM
        overture_utm = overture_buildings.to_crs(overture_buildings.estimate_utm_crs())
        overture_buildings['area_m2'] = overture_utm.geometry.area.round(2)
        
        # Initialize Earth Engine
        service_account_info = generate_service_account_info()
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'], 
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            credentials=credentials
        )
        
        # Create EE geometry
        minx, miny, maxx, maxy = tile_bounds
        coords = [[
            [minx, miny],
            [minx, maxy],
            [maxx, maxy],
            [maxx, miny],
            [minx, miny]
        ]]
        tile_bounds_ee = ee.Geometry.Polygon(coords)
        
        # Get all available timestamps
        buildings_collection = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')
        bounded_collection = buildings_collection.filterBounds(tile_bounds_ee)
        timestamps = bounded_collection.aggregate_array('system:time_start').distinct().getInfo()
        
        if not timestamps:
            print(f"No timestamps available for tile {tile_bounds}")
            return None
            
        print(f"Found {len(timestamps)} timestamps for tile")
        
        # Initialize base buildings DataFrame
        base_buildings = pd.DataFrame({
            'id': overture_buildings['id'],
            'geometry': overture_buildings.geometry,
            'area_m2': overture_buildings['area_m2']
        })
        
        # Process each timestamp
        for current_timestamp in timestamps:
            timestamp_str = datetime.fromtimestamp(current_timestamp/1000).strftime('%Y%m')
            print(f"Processing timestamp: {timestamp_str}")
            
            buildings = bounded_collection.filter(ee.Filter.eq('system:time_start', current_timestamp))
            mosaic = buildings.mosaic()
            
            # Get height and presence data
            presence_url = mosaic.select('building_presence').getThumbURL({
                'region': tile_bounds_ee,
                'dimensions': '512x512',
                'format': 'NPY',
                'min': 0,
                'max': 1
            })
            
            height_url = mosaic.select('building_height').getThumbURL({
                'region': tile_bounds_ee,
                'dimensions': '512x512',
                'format': 'NPY',
                'min': 0,
                'max': 100
            })
            
            presence_array = np.load(io.BytesIO(requests.get(presence_url).content)).astype(np.float64)
            height_array = np.load(io.BytesIO(requests.get(height_url).content)).astype(np.float64)
            
            # Process building metrics
            y_coords = np.linspace(maxy, miny, height_array.shape[0])
            x_coords = np.linspace(minx, maxx, height_array.shape[1])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            mask = presence_array.flatten() >= 0.6
            points_df = pd.DataFrame({
                'lat': Y.flatten()[mask],
                'lng': X.flatten()[mask],
                'height_val': height_array.flatten()[mask]
            })
            
            from shapely.geometry import Point
            points_gdf = gpd.GeoDataFrame(
                points_df,
                geometry=[Point(x, y) for x, y in zip(points_df['lng'], points_df['lat'])],
                crs="EPSG:4326"
            )
            
            # Reset index for points_gdf to avoid conflicts
            points_gdf = points_gdf.reset_index(drop=True)
            
            # Spatial join with nearest points
            joined = gpd.sjoin_nearest(
                overture_buildings.reset_index(drop=True),
                points_gdf[['geometry', 'height_val']],
                how='left',
                max_distance=0.0001  # Approx 10m at equator
            )
            
            if len(joined) > 0:
                result = joined.groupby('id').agg({
                    'height_val': ['mean', 'max', 'count']
                }).reset_index()
                
                result.columns = ['id', 
                                f'height_mean_{timestamp_str}',
                                f'height_max_{timestamp_str}',
                                f'points_count_{timestamp_str}']
                
                # Round metrics
                result[f'height_mean_{timestamp_str}'] = result[f'height_mean_{timestamp_str}'].round(2)
                result[f'height_max_{timestamp_str}'] = result[f'height_max_{timestamp_str}'].round(2)
                
                # Add confidence and volume
                result[f'height_confidence_{timestamp_str}'] = (
                    result[f'points_count_{timestamp_str}'] / result[f'points_count_{timestamp_str}'].max()
                ).round(3)
                
                # Merge with base buildings
                base_buildings = base_buildings.merge(result, on='id', how='left')
                
                # Calculate volume for this timestamp
                base_buildings[f'volume_m3_{timestamp_str}'] = (
                    base_buildings['area_m2'] * base_buildings[f'height_mean_{timestamp_str}']
                ).round(2)
        
        if len(base_buildings) > 0:
            # Convert to GeoDataFrame
            result_gdf = gpd.GeoDataFrame(base_buildings, geometry='geometry', crs="EPSG:4326")
            
            # Save tile data with all timestamps
            save_success = save_tile_data(result_gdf, tile_bounds)
            if not save_success:
                print(f"Failed to save tile data for bounds {tile_bounds}")
            
            return len(result_gdf)
        
        return 0
        
    except Exception as e:
        print(f"Error processing tile: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def run_thane_batch_processing():
    """Run the complete batch processing workflow"""
    # Get list of tiles to process
    tiles_df = fused.run(prepare_thane_tiles)
    print(f"Generated {len(tiles_df)} tiles to process")
    
    # Run batch job
    job = process_thane_buildings(arg_list=tiles_df.to_dict('records')).run_remote(disk_size_gb=450)
    return job

# Run the processing
job = run_thane_batch_processing()
print(job)

# Monitor progress
fused.api.job_tail_logs(job.job_id)
