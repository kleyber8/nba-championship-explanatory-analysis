import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

# Configuración: crea la carpeta 'data/raw/' si no existe
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Lista de temporadas en el formato que espera la NBA API
SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

def fetch_playoff_games(season):
    """Descarga los datos de todos los partidos de playoffs para una temporada dada."""
    print(f"Descargando datos de playoffs para la temporada {season}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Playoffs'
        )
        # 'leaguegamefinder' devuelve una lista de DataFrames. El primero contiene los datos.
        games_df = gamefinder.get_data_frames()[0]

        # Añadimos una columna con la temporada para identificar el dato fácilmente
        games_df['SEASON'] = season
        print(f"  -> {len(games_df)} partidos encontrados.")
        return games_df
    except Exception as e:
        print(f"  -> Error descargando {season}: {e}")
        return pd.DataFrame()

def main():
    all_playoffs_data = pd.DataFrame()
    for season in SEASONS:
        df_season = fetch_playoff_games(season)
        if not df_season.empty:
            all_playoffs_data = pd.concat([all_playoffs_data, df_season], ignore_index=True)
        # Pequeña pausa para no saturar la API
        time.sleep(1)

    # Guardamos todos los datos en un único archivo CSV en la carpeta 'data/raw/'
    output_path = os.path.join(RAW_DATA_DIR, "all_playoffs_2020_2025.csv")
    all_playoffs_data.to_csv(output_path, index=False)
    print(f"\nDatos guardados exitosamente en: {output_path}")

if __name__ == "__main__":
    main()