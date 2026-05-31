"""
build_features.py (Versión definitiva sin errores de tipo)
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/all_playoffs_2020_2025.csv"
PROCESSED_DIR = "data/processed"
CHAMPIONS = {2021: "MIL", 2022: "GSW", 2023: "DEN", 2024: "BOS", 2025: None}
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Mapeo de IDs a nombres y abreviaturas
TEAM_ID_TO_NAME = {
    1610612737: 'Atlanta Hawks', 1610612738: 'Boston Celtics', 1610612751: 'Brooklyn Nets',
    1610612766: 'Charlotte Hornets', 1610612741: 'Chicago Bulls', 1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks', 1610612743: 'Denver Nuggets', 1610612765: 'Detroit Pistons',
    1610612744: 'Golden State Warriors', 1610612745: 'Houston Rockets', 1610612754: 'Indiana Pacers',
    1610612746: 'LA Clippers', 1610612747: 'Los Angeles Lakers', 1610612763: 'Memphis Grizzlies',
    1610612748: 'Miami Heat', 1610612749: 'Milwaukee Bucks', 1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans', 1610612752: 'New York Knicks', 1610612760: 'Oklahoma City Thunder',
    1610612753: 'Orlando Magic', 1610612755: 'Philadelphia 76ers', 1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers', 1610612758: 'Sacramento Kings', 1610612759: 'San Antonio Spurs',
    1610612761: 'Toronto Raptors', 1610612762: 'Utah Jazz', 1610612764: 'Washington Wizards'
}
NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def calculate_efficiency_metrics(df_season):
    team_stats = df_season.groupby(['TEAM_ID', 'TEAM_NAME']).agg(
        games=('GAME_ID', 'count'),
        pts=('PTS', 'mean'),
        fgm=('FGM', 'mean'),
        fga=('FGA', 'mean'),
        fg3m=('FG3M', 'mean'),
        fg3a=('FG3A', 'mean'),
        ftm=('FTM', 'mean'),
        fta=('FTA', 'mean'),
        oreb=('OREB', 'mean'),
        tov=('TOV', 'mean'),
        reb=('REB', 'mean')
    ).reset_index()
    pm = df_season.groupby('TEAM_ID')['PLUS_MINUS'].mean().reset_index()
    team_stats = team_stats.merge(pm, on='TEAM_ID', how='left')
    team_stats.rename(columns={'PLUS_MINUS': 'net_rating'}, inplace=True)
    team_stats['efg_pct'] = (team_stats['fgm'] + 0.5 * team_stats['fg3m']) / team_stats['fga']
    team_stats['tov_pct'] = team_stats['tov'] / (team_stats['fga'] + 0.44 * team_stats['fta'] + team_stats['tov'])
    team_stats['ft_rate'] = team_stats['fta'] / team_stats['fga']
    return team_stats

def calculate_clutch_and_context(df_season):
    game_diffs = df_season.groupby('GAME_ID')['PTS'].diff().abs().dropna().reset_index()
    game_diffs.columns = ['GAME_ID', 'point_diff']
    df_with_diff = df_season.merge(game_diffs, on='GAME_ID', how='left')
    clutch = df_with_diff[df_with_diff['point_diff'] <= 5]
    clutch_stats = clutch.groupby('TEAM_ID').apply(
        lambda x: pd.Series({
            'clutch_games': len(x),
            'clutch_win_pct': (x['PLUS_MINUS'] > 0).mean() if len(x) > 0 else 0
        })
    ).reset_index()
    home_away = df_season.groupby(['TEAM_ID', 'MATCHUP'])['PLUS_MINUS'].mean().reset_index()
    home = home_away[home_away['MATCHUP'].str.contains('vs.')].set_index('TEAM_ID')['PLUS_MINUS']
    away = home_away[home_away['MATCHUP'].str.contains('@')].set_index('TEAM_ID')['PLUS_MINUS']
    home_adv = (home - away).reset_index()
    home_adv.columns = ['TEAM_ID', 'home_court_advantage']
    return clutch_stats, home_adv

def calculate_defensive_metrics(df_season):
    opp = df_season.copy()
    opp['OPP_PTS'] = opp.groupby('GAME_ID')['PTS'].transform('sum') - opp['PTS']
    opp['OPP_FGM'] = opp.groupby('GAME_ID')['FGM'].transform('sum') - opp['FGM']
    opp['OPP_FGA'] = opp.groupby('GAME_ID')['FGA'].transform('sum') - opp['FGA']
    opp['OPP_FG3M'] = opp.groupby('GAME_ID')['FG3M'].transform('sum') - opp['FG3M']
    opp['OPP_FG3A'] = opp.groupby('GAME_ID')['FG3A'].transform('sum') - opp['FG3A']
    opp['OPP_REB'] = opp.groupby('GAME_ID')['REB'].transform('sum') - opp['REB']
    def_agg = opp.groupby('TEAM_ID').agg(
        opp_pts=('OPP_PTS', 'mean'),
        opp_fgm_mean=('OPP_FGM', 'mean'),
        opp_fga_mean=('OPP_FGA', 'mean'),
        opp_fg3m_mean=('OPP_FG3M', 'mean'),
        opp_fg3a_mean=('OPP_FG3A', 'mean'),
        opp_reb_mean=('OPP_REB', 'mean')
    ).reset_index()
    def_agg['opp_fg_pct'] = def_agg['opp_fgm_mean'] / def_agg['opp_fga_mean'].replace(0, np.nan)
    def_agg['opp_fg3_pct'] = def_agg['opp_fg3m_mean'] / def_agg['opp_fg3a_mean'].replace(0, np.nan)
    defensive = def_agg[['TEAM_ID', 'opp_pts', 'opp_fg_pct', 'opp_fg3_pct', 'opp_reb_mean']]
    def poss(row):
        return row['FGA'] - row['OREB'] + row['TOV'] + 0.44 * row['FTA']
    df_season['poss_team'] = df_season.apply(poss, axis=1)
    game_pace = df_season.groupby('GAME_ID')['poss_team'].sum() / 2
    df_season = df_season.merge(game_pace.rename('pace'), on='GAME_ID')
    team_pace = df_season.groupby('TEAM_ID')['pace'].mean().reset_index()
    team_pace.columns = ['TEAM_ID', 'pace']
    return defensive, team_pace

def calculate_momentum(df_historical):
    df_sorted = df_historical.sort_values(['TEAM_ID', 'GAME_DATE'])
    df_sorted['is_win'] = (df_sorted['PLUS_MINUS'] > 0).astype(int)
    last10 = df_sorted.groupby('TEAM_ID').tail(10)
    momentum = last10.groupby('TEAM_ID')['is_win'].mean().reset_index()
    momentum.columns = ['TEAM_ID', 'momentum']
    return momentum

def calculate_rebound_differential(team_stats, defensive):
    # team_stats tiene 'reb', defensive tiene 'opp_reb_mean'
    merged = team_stats[['TEAM_ID', 'reb']].merge(defensive[['TEAM_ID', 'opp_reb_mean']], on='TEAM_ID', how='left')
    merged['reb_diff'] = merged['reb'] - merged['opp_reb_mean']
    return merged[['TEAM_ID', 'reb_diff']]

def calculate_days_rest(df_season):
    df_sorted = df_season.sort_values(['TEAM_ID', 'GAME_DATE'])
    df_sorted['prev_date'] = df_sorted.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df_sorted['rest_days'] = (df_sorted['GAME_DATE'] - df_sorted['prev_date']).dt.days
    rest = df_sorted.dropna(subset=['rest_days'])
    avg_rest = rest.groupby('TEAM_ID')['rest_days'].mean().reset_index()
    avg_rest.columns = ['TEAM_ID', 'days_rest_avg']
    return avg_rest

def calculate_opponent_net_rating(df_season):
    team_avg_nr = df_season.groupby('TEAM_ID')['PLUS_MINUS'].mean().to_dict()
    opp_list = []
    for gid, group in df_season.groupby('GAME_ID'):
        if len(group) == 2:
            teams = group['TEAM_ID'].values
            nr0 = team_avg_nr.get(teams[0], 0)
            nr1 = team_avg_nr.get(teams[1], 0)
            opp_list.append({'TEAM_ID': teams[0], 'opp_nr': nr1})
            opp_list.append({'TEAM_ID': teams[1], 'opp_nr': nr0})
    opp_df = pd.DataFrame(opp_list)
    avg_opp_nr = opp_df.groupby('TEAM_ID')['opp_nr'].mean().reset_index()
    avg_opp_nr.columns = ['TEAM_ID', 'opp_net_rating_avg']
    return avg_opp_nr

def calculate_elimination_games(df_season):
    # Crear identificador de serie basado en dos equipos por GAME_ID
    series = {}
    for gid, group in df_season.groupby('GAME_ID'):
        teams = sorted(group['TEAM_ID'].unique())
        if len(teams) == 2:
            key = tuple(teams)
            for _, row in group.iterrows():
                series.setdefault(key, []).append({
                    'GAME_ID': row['GAME_ID'],
                    'TEAM_ID': row['TEAM_ID'],
                    'GAME_DATE': row['GAME_DATE'],
                    'win': row['PLUS_MINUS'] > 0
                })
    elim_data = []
    for (t1, t2), games in series.items():
        games_sorted = sorted(games, key=lambda x: x['GAME_DATE'])
        losses = {t1: 0, t2: 0}
        for game in games_sorted:
            team = game['TEAM_ID']
            # Es eliminación si el equipo ya tiene 3 derrotas antes del partido
            elim = (losses[team] == 3)
            if not game['win']:
                losses[team] += 1
            elim_data.append({'TEAM_ID': team, 'elimination_game': elim, 'win': game['win']})
    elim_df = pd.DataFrame(elim_data)
    if elim_df.empty:
        # Si no hay datos, devolver DataFrame vacío con columnas esperadas
        return pd.DataFrame(columns=['TEAM_ID', 'elimination_games', 'elimination_win_pct'])
    elim_stats = elim_df[elim_df['elimination_game']].groupby('TEAM_ID').agg(
        elimination_games=('elimination_game', 'count'),
        elimination_win_pct=('win', 'mean')
    ).reset_index()
    # Asegurar que todos los equipos estén presentes
    all_teams = df_season['TEAM_ID'].unique()
    elim_stats = elim_stats.set_index('TEAM_ID').reindex(all_teams).reset_index()
    elim_stats['elimination_games'] = elim_stats['elimination_games'].fillna(0)
    elim_stats['elimination_win_pct'] = elim_stats['elimination_win_pct'].fillna(0)
    elim_stats.rename(columns={'index': 'TEAM_ID'}, inplace=True)
    return elim_stats

def main():
    print("Cargando datos...")
    df_raw = load_data()
    seasons = df_raw['SEASON'].unique()
    all_features = []

    for season in seasons:
        print(f"Procesando {season}...")
        df_season = df_raw[df_raw['SEASON'] == season].copy()
        
        team_stats = calculate_efficiency_metrics(df_season)
        clutch, home_adv = calculate_clutch_and_context(df_season)
        defensive, pace = calculate_defensive_metrics(df_season)
        momentum = calculate_momentum(df_raw)
        reb_diff = calculate_rebound_differential(team_stats, defensive)
        days_rest = calculate_days_rest(df_season)
        opp_nr = calculate_opponent_net_rating(df_season)
        elim = calculate_elimination_games(df_season)
        
        # Combinar todo con merges sucesivos
        feat = team_stats.merge(clutch, on='TEAM_ID', how='left')
        feat = feat.merge(home_adv, on='TEAM_ID', how='left')
        feat = feat.merge(defensive, on='TEAM_ID', how='left')
        feat = feat.merge(pace, on='TEAM_ID', how='left')
        feat = feat.merge(momentum, on='TEAM_ID', how='left')
        feat = feat.merge(reb_diff, on='TEAM_ID', how='left')
        feat = feat.merge(days_rest, on='TEAM_ID', how='left')
        feat = feat.merge(opp_nr, on='TEAM_ID', how='left')
        feat = feat.merge(elim, on='TEAM_ID', how='left')
        
        # Rellenar solo columnas numéricas (excluir 'TEAM_NAME' y 'TEAM_ID')
        numeric_cols = feat.select_dtypes(include=[np.number]).columns
        fill_values = {
            'clutch_games': 0, 'clutch_win_pct': 0, 'home_court_advantage': 0,
            'opp_fg_pct': 0, 'opp_fg3_pct': 0, 'momentum': 0,
            'reb_diff': 0, 'days_rest_avg': 2.0, 'opp_net_rating_avg': 0,
            'elimination_games': 0, 'elimination_win_pct': 0
        }
        for col, val in fill_values.items():
            if col in feat.columns:
                feat[col] = feat[col].fillna(val)
        
        # Convertir temporada
        feat['season'] = int(season.split('-')[0]) + 1
        all_features.append(feat)
    
    final = pd.concat(all_features, ignore_index=True)
    
    # Añadir nombre y abreviatura si no existen
    if 'TEAM_NAME' not in final.columns:
        final['TEAM_NAME'] = final['TEAM_ID'].map(TEAM_ID_TO_NAME)
    final['team_abbr'] = final['TEAM_NAME'].map(NAME_TO_ABBR)
    
    # Variable objetivo is_champion
    final['is_champion'] = 0
    for season, champ_abbr in CHAMPIONS.items():
        if champ_abbr is None:
            continue
        mask = (final['season'] == season) & (final['team_abbr'] == champ_abbr)
        final.loc[mask, 'is_champion'] = 1
    
    # Columnas de salida
    out_cols = ['TEAM_ID', 'team_abbr', 'season', 'games', 'pts', 'efg_pct', 'tov_pct',
                'ft_rate', 'net_rating', 'clutch_games', 'clutch_win_pct',
                'home_court_advantage', 'opp_pts', 'opp_fg_pct', 'opp_fg3_pct',
                'pace', 'momentum', 'reb_diff', 'days_rest_avg', 'opp_net_rating_avg',
                'elimination_games', 'elimination_win_pct', 'is_champion']
    
    # Asegurar que todas las columnas existan (si falta alguna, la creamos con ceros)
    for col in out_cols:
        if col not in final.columns:
            final[col] = 0
    
    final = final[out_cols]
    
    output_path = os.path.join(PROCESSED_DIR, 'complete_playoffs_features_enhanced.csv')
    final.to_csv(output_path, index=False)
    print(f"\nDataset mejorado guardado en: {output_path}")
    print(f"Filas: {len(final)}, Columnas: {len(final.columns)}")
    print(final.head())

if __name__ == "__main__":
    main()