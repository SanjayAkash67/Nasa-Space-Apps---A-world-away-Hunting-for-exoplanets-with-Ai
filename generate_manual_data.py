import lightkurve as lk
import pandas as pd
import numpy as np
import warnings

# This print statement will confirm the script is executing.
print("--- SCRIPT IS RUNNING ---")

warnings.filterwarnings('ignore')

def generate_files_for_star(target_id, mission):
    """
    Downloads data for a star and creates the two CSV files
    needed for the manual analysis tool.
    """
    print(f"--- Starting data generation for {target_id} ---")
    
    try:
        print(f"1/3: Searching for {target_id} in {mission} database...")
        search_result = lk.search_lightcurve(target_id, mission=mission)
        if not search_result:
            print(f"Error: No data found for {target_id}")
            return

        print("2/3: Downloading light curve data...")
        lc = search_result[0:5].download_all().stitch().remove_nans().normalize().remove_outliers()
    except Exception as e:
        print(f"Error during download: {e}")
        return

    # SAVE THE LIGHTCURVE.CSV FILE
    lightcurve_df_raw = lc.to_pandas().reset_index()
    time_col = 'time'
    # TESS data sometimes uses a different time column name
    if 'time' not in lightcurve_df_raw.columns and 'time_bin_start' in lightcurve_df_raw.columns:
        time_col = 'time_bin_start'
    
    if time_col in lightcurve_df_raw.columns and 'flux' in lightcurve_df_raw.columns:
        lightcurve_df_to_save = lightcurve_df_raw[[time_col, 'flux']]
        lightcurve_df_to_save = lightcurve_df_to_save.rename(columns={time_col: 'time'})
        lightcurve_df_to_save.to_csv("lightcurve.csv", index=False)
        print("\n✅ Successfully created lightcurve.csv")
    else:
        print("Error: Could not find required 'time' and 'flux' columns.")
        return

    # RUN BLS TO GET KEY FEATURES
    print("3/3: Analyzing data to extract key features...")
    bls = lc.to_periodogram(method='bls')
    period = bls.period_at_max_power.value
    duration_hours = (bls.duration_at_max_power.to('h')).value
    depth = bls.depth_at_max_power.value
    if not np.isscalar(depth):
        depth = depth[0]

    # CREATE THE FEATURES.CSV FILE
    feature_columns = [
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period', 
        'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 
        'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 
        'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 'koi_teq_err1', 'koi_teq_err2', 'koi_insol', 
        'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 
        'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 
        'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag'
    ]
    
    feature_data = {col: 0 for col in feature_columns}
    feature_data['koi_period'] = period
    feature_data['koi_duration'] = duration_hours
    feature_data['koi_depth'] = depth * 1_000_000

    features_df = pd.DataFrame([feature_data])
    features_df.to_csv("features.csv", index=False)
    print("✅ Successfully created features.csv")
    print("\n--- Data generation complete! You can now use these two files in the app. ---")

# This is the main execution block. It will run when you execute the script.
if __name__ == "__main__":
    # Choose the star you want to generate data for
    # Example 1:
    # MISSION = "Kepler"
    # TARGET_ID = "KIC 6541920" # Kepler-11

    # Example 2:
    MISSION = "TESS"
    TARGET_ID = "TIC 150428135" # TOI-700
    
    generate_files_for_star(TARGET_ID, MISSION)