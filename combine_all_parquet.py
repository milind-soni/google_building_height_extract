@fused.udf
def udf():
    """
    Combines parquet files from the source path into a single CSV file in the target path.
    """
    import pandas as pd
    import s3fs
    import os
    
    INPUT_PATH = "s3://fused-users/milind/thane_buildings_with_timestamps/"
    OUTPUT_PATH = "s3://fused-users/milind/meow_ceew/thane_buildings_all.csv"
    
    try:
        # Initialize S3 filesystem
        fs = s3fs.S3FileSystem()
        
        # List all parquet files in the input directory
        print(f"Searching for parquet files in: {INPUT_PATH}")
        parquet_files = fs.glob(INPUT_PATH + "*.parquet")
        
        if not parquet_files:
            print("No parquet files found in the source path")
            return None
            
        print(f"Found {len(parquet_files)} parquet files")
        
        # Read and combine all parquet files
        dfs = []
        for i, file in enumerate(parquet_files, 1):
            print(f"Processing file {i}/{len(parquet_files)}: {file}")
            try:
                # Use s3fs to open the file
                with fs.open(file, 'rb') as f:
                    df = pd.read_parquet(f)
                dfs.append(df)
                print(f"Successfully read file with {len(df)} rows")
            except Exception as e:
                print(f"Error reading file {file}: {str(e)}")
                continue
        
        if not dfs:
            print("No data could be read from parquet files")
            return None
        
        # Combine all dataframes
        print("Combining dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined shape: {combined_df.shape}")
        
        # Save to CSV in the target location
        print(f"Saving combined CSV to: {OUTPUT_PATH}")
        with fs.open(OUTPUT_PATH, 'w') as f:
            combined_df.to_csv(f, index=False)
        
        print(f"Successfully combined {len(parquet_files)} files")
        print(f"Final dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        # Print column names for verification
        print("\nColumns in combined dataset:")
        print(combined_df.columns.tolist())
        
        return len(combined_df)
        
    except Exception as e:
        print(f"Error combining files: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
