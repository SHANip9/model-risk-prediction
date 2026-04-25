import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    scores_path = project_root / 'data' / 'processed' / 'zone_risk_scores_latest.csv'
    output_path = project_root / 'data' / 'processed' / 'zone_risk_heatmap_latest.csv'
    
    # Load time-series scores
    df_scores = pd.read_csv(scores_path)
    df_scores['week_start'] = pd.to_datetime(df_scores['week_start'])
    
    # Static zone geo map (from FIXED)
    zone_geo = {
        'zone_1': {'display_name': 'Dharavi', 'city': 'Mumbai', 'lat': 19.0422, 'lon': 72.8553},
        'zone_2': {'display_name': 'Kurla West', 'city': 'Mumbai', 'lat': 19.0728, 'lon': 72.8826},
        'zone_3': {'display_name': 'Andheri East', 'city': 'Mumbai', 'lat': 19.1136, 'lon': 72.8697},
        'zone_4': {'display_name': 'Bandra Kurla', 'city': 'Mumbai', 'lat': 19.0596, 'lon': 72.8656},
        'zone_5': {'display_name': 'Thane West', 'city': 'Thane', 'lat': 19.1852, 'lon': 72.971}
    }
    
    # Mumbai center for distance
    center_lat, center_lon = 19.0761, 72.8777
    
    # Aggregate per zone using all data points
    heatmap_data = []
    for zone_id, geo in zone_geo.items():
        zone_df = df_scores[df_scores['zone_id'] == zone_id]
        if zone_df.empty:
            continue
        
        latest = zone_df.loc[zone_df['week_start'].idxmax()]
        
        agg = {
            'zone_id': zone_id,
            'display_name': geo['display_name'],
            'city': geo['city'],
            'latitude': geo['lat'],
            'longitude': geo['lon'],
            'risk_score_0_100': latest['risk_score_0_100'],
            'risk_bin': latest['risk_bin'],
            'week_start': latest['week_start'].strftime('%Y-%m-%d'),
            'avg_risk': zone_df['risk_score_0_100'].mean(),
            'min_risk': zone_df['risk_score_0_100'].min(),
            'max_risk': zone_df['risk_score_0_100'].max(),
            'total_weeks': len(zone_df),
            'distance_km': haversine(geo['lat'], geo['lon'], center_lat, center_lon),
            'in_radius': True if haversine(geo['lat'], geo['lon'], center_lat, center_lon) < 16 else False,
            'scored_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-7] + '+00:00'
        }
        heatmap_data.append(agg)
    
    df_heatmap = pd.DataFrame(heatmap_data)
    df_heatmap = df_heatmap.sort_values('zone_id').reset_index(drop=True)
    
    # Write CSV matching FIXED format/order
    df_heatmap.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df_heatmap)} zones.")

if __name__ == "__main__":
    main()

