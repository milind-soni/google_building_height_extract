@fused.udf
def udf(
    buildings_path: str = 's3://fused-users/milind/meow_ceew/thane_buildings_all.csv',
    boundary_path: str = 's3://fused-users/milind/meow_ceew/thane.geojson',
    output_path: str = 's3://fused-users/milind/meow_ceew/final_thane.csv'):
    
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt
    import fsspec
    
    try:
        # Read buildings
        print("Reading buildings...")
        buildings_df = pd.read_csv(buildings_path)
        buildings_gdf = gpd.GeoDataFrame(
            buildings_df,
            geometry=buildings_df['geometry'].apply(wkt.loads),
            crs="EPSG:4326"
        )
        
        # Read boundary
        print("Reading Thane boundary...")
        with fsspec.open(boundary_path) as f:
            boundary_gdf = gpd.read_file(f)
        
        # Clip buildings
        print("Clipping buildings...")
        clipped = gpd.clip(buildings_gdf, boundary_gdf)
        
        # Save result
        print("Saving results...")
        save_df = pd.DataFrame(clipped.drop(columns=['geometry']))
        save_df['geometry'] = clipped.geometry.apply(lambda x: x.wkt)
        save_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(clipped)} buildings to {output_path}")
        return len(clipped)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
