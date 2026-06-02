"""
eda_preview.py (corregido)
Análisis exploratorio preliminar del dataset final.
Genera estadísticas y visualizaciones en la carpeta reports/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
DATA_PATH = "data/processed/complete_playoffs_features_final.csv"
OUTPUT_DIR = "reports"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Cargar datos
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas\n")

# 2. Información general
print("=== Información general ===")
print(df.info())
print("\n=== Primeras filas ===")
print(df.head())

# 3. Estadísticas descriptivas (variables numéricas)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n=== Estadísticas descriptivas (numéricas) ===")
print(df[num_cols].describe())

# 4. Valores nulos
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print("\n=== Valores nulos por columna ===")
    print(null_counts[null_counts > 0])
else:
    print("\nNo hay valores nulos en el dataset.")

# 5. Distribución de la variable objetivo
print("\n=== Distribución de 'is_champion' ===")
champ_counts = df['is_champion'].value_counts()
print(champ_counts)
print(f"Proporción de campeones: {champ_counts[1]/len(df):.3f}")

# 6. Boxplots comparativos: campeón vs no campeón para variables seleccionadas
key_metrics = ['net_rating', 'efg_pct', 'clutch_win_pct', 'reb_diff', 
               'opp_net_rating_avg', 'clutch_efg_pct', 'clutch_net_rating', 'momentum']
existing_metrics = [m for m in key_metrics if m in df.columns]

if existing_metrics:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, metric in enumerate(existing_metrics):
        # Corregido: usar hue en lugar de palette directamente
        sns.boxplot(data=df, x='is_champion', y=metric, hue='is_champion', 
                    legend=False, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{metric} por campeonato')
        axes[i].set_xlabel('Campeón (1) / No campeón (0)')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'boxplots_champion_vs_rest.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBoxplots guardados en {FIGURES_DIR}/boxplots_champion_vs_rest.png")

# 7. Matriz de correlación de las variables numéricas (excluyendo columnas no relevantes)
exclude_from_corr = ['TEAM_ID', 'season', 'is_champion']
corr_cols = [c for c in num_cols if c not in exclude_from_corr]
if len(corr_cols) > 1:
    plt.figure(figsize=(16, 12))
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de correlación (variables numéricas)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matriz de correlación guardada en {FIGURES_DIR}/correlation_matrix.png")
    
    # Correlaciones con la variable objetivo
    if 'is_champion' in corr_matrix.columns:
        corr_with_target = corr_matrix['is_champion'].sort_values(ascending=False)
        print("\n=== Correlación de cada variable con 'is_champion' ===")
        print(corr_with_target)
else:
    print("No hay suficientes columnas numéricas para la matriz de correlación.")

# 8. Histogramas de distribución (sin KDE para evitar errores de matriz singular)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, metric in enumerate(existing_metrics[:8]):
    sns.histplot(data=df, x=metric, hue='is_champion', kde=False, ax=axes[i], 
                 palette='Set2', alpha=0.6, bins=20)
    axes[i].set_title(f'Distribución de {metric}')
    axes[i].set_xlabel(metric)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'histograms_key_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Histogramas guardados en {FIGURES_DIR}/histograms_key_metrics.png")

# 9. Resumen de estadísticas agrupadas por campeón
print("\n=== Medias de las métricas clave por campeón ===")
grouped = df.groupby('is_champion')[existing_metrics].mean().round(3)
print(grouped)

# 10. Guardar un informe de texto
report_lines = []
report_lines.append("=== REPORTE EDA PRELIMINAR ===\n")
report_lines.append(f"Dimensiones: {df.shape}\n")
report_lines.append(f"Proporción de campeones: {champ_counts[1]/len(df):.3f}\n")
report_lines.append("=== Medias por campeón ===\n")
report_lines.append(grouped.to_string())
with open(os.path.join(OUTPUT_DIR, 'eda_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\nResumen de texto guardado en {OUTPUT_DIR}/eda_summary.txt")
print("EDA finalizado. Revisa la carpeta 'reports/' para ver los resultados.")