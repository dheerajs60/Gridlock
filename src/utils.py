import pandas as pd
import numpy as np

def decode_geohash(geohash):
    """
    Decodes a geohash string into latitude and longitude.
    """
    if not isinstance(geohash, str) or len(geohash) == 0:
        return np.nan, np.nan
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    
    for char in geohash:
        if char not in base32:
            return np.nan, np.nan
        val = base32.index(char)
        for i in range(4, -1, -1):
            bit = (val >> i) & 1
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit == 1:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit == 1:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

def parse_time_features(df):
    """
    Extracts time features from the timestamp column.
    timestamp format is 'H:M' (e.g. '2:15')
    """
    df = df.copy()
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # Cyclic features
    df['sin_time'] = np.sin(2 * np.pi * df['time_minutes'] / 1440.0)
    df['cos_time'] = np.cos(2 * np.pi * df['time_minutes'] / 1440.0)
    
    return df

def impute_missing_values(df, train_df=None):
    """
    Imputes missing values for RoadType, Weather, and Temperature.
    If train_df is provided, uses it to compute imputation mappings.
    """
    df = df.copy()
    ref_df = train_df if train_df is not None else df
    
    # Modes
    road_mode = ref_df['RoadType'].mode()[0] if not ref_df['RoadType'].mode().empty else 'Residential'
    weather_mode = ref_df['Weather'].mode()[0] if not ref_df['Weather'].mode().empty else 'Sunny'
    
    df['RoadType'] = df['RoadType'].fillna(road_mode)
    df['Weather'] = df['Weather'].fillna(weather_mode)
    
    # Temperature mapping: Group by hour and Weather to get mean temperature
    temp_map = ref_df.groupby(['hour', 'Weather'])['Temperature'].mean().to_dict()
    overall_mean = ref_df['Temperature'].mean()
    if np.isnan(overall_mean):
        overall_mean = 20.0
        
    def fill_temp(row):
        if not np.isnan(row['Temperature']):
            return row['Temperature']
        key = (row['hour'], row['Weather'])
        if key in temp_map and not np.isnan(temp_map[key]):
            return temp_map[key]
        return overall_mean
        
    df['Temperature'] = df.apply(fill_temp, axis=1)
    return df
