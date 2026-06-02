"""
add_advanced_clutch.py (Corregido)
Añade clutch_efg_pct y clutch_net_rating usando umbral configurable (por defecto 5 puntos).
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/all_playoffs_2020_2025.csv"
INPUT_CSV = "data/processed/complete_playoffs_features_enhanced.csv"
OUTPUT_CSV = "data/processed/complete_playoffs_features_final.csv"
CLUTCH_THRESHOLD = 5   # Puedes cambiarlo a 10 si quieres más muestra

def load_raw():
    df = pd.read_csv(RAW_DATA_PATH)
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def compute_clutch_advanced(df_season, threshold=CLUTCH_THRESHOLD):
    # Calcular diferencia de puntos por partido de forma robusta
    game_diffs = []
    for gid, group in df_season.groupby('GAME_ID'):
        if len(group) == 2:
            pts = group['PTS'].values
            diff = abs(pts[0] - pts[1])
            game_diffs.append({'GAME_ID': gid, 'point_diff': diff})
    game_diffs_df = pd.DataFrame(game_diffs)
    df_with_diff = df_season.merge(game_diffs_df, on='GAME_ID', how='left')
    clutch = df_with_diff[df_with_diff['point_diff'] <= threshold].copy()
    
    if clutch.empty:
        all_teams = df_season['TEAM_ID'].unique()
        season_year = int(df_season['SEASON'].iloc[0].split('-')[0]) + 1
        return pd.DataFrame({
            'TEAM_ID': all_teams,
            'season': season_year,
            'clutch_efg_pct': 0,
            'clutch_net_rating': 0
        })
    
    clutch['efg_pct_game'] = (clutch['FGM'] + 0.5 * clutch['FG3M']) / clutch['FGA']
    clutch_stats = clutch.groupby('TEAM_ID').agg(
        clutch_efg_pct=('efg_pct_game', 'mean'),
        clutch_net_rating=('PLUS_MINUS', 'mean')
    ).reset_index()
    season_year = int(df_season['SEASON'].iloc[0].split('-')[0]) + 1
    clutch_stats['season'] = season_year
    return clutch_stats

def main():
    print("Cargando raw data...")
    df_raw = load_raw()
    print("Cargando dataset enhanced existente...")
    df_enhanced = pd.read_csv(INPUT_CSV)
    
    seasons = df_raw['SEASON'].unique()
    all_clutch = []
    for season in seasons:
        print(f"Procesando {season}...")
        df_season = df_raw[df_raw['SEASON'] == season].copy()
        clutch_adv = compute_clutch_advanced(df_season)
        all_clutch.append(clutch_adv)
    
    df_clutch = pd.concat(all_clutch, ignore_index=True)
    df_final = df_enhanced.merge(df_clutch, on=['TEAM_ID', 'season'], how='left')
    df_final['clutch_efg_pct'] = df_final['clutch_efg_pct'].fillna(0)
    df_final['clutch_net_rating'] = df_final['clutch_net_rating'].fillna(0)
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset final guardado en {OUTPUT_CSV}")
    print(f"Filas: {len(df_final)}, Columnas: {len(df_final.columns)}")
    print("Nuevas columnas añadidas: clutch_efg_pct, clutch_net_rating")
    print(f"Umbral usado: {CLUTCH_THRESHOLD} puntos")

if __name__ == "__main__":
    main()