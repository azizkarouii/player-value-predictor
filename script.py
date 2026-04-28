"""
VERSION ROBUSTE - Script Python pour générer toutes les figures du rapport Phase 2
Avec gestion d'erreurs et messages de débogage

Auteurs : Aziz Karoui - Wassim Sioud
Date : Mars 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter les problèmes d'affichage
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

print("=" * 80)
print("VERSION ROBUSTE - GÉNÉRATION DES FIGURES")
print("=" * 80)
print(f"Python version: {sys.version}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print()

# Configuration globale
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Couleurs
COLOR_PRIMARY = '#003366'
COLOR_ACCENT = '#0078D4'
COLOR_LIGHT = '#F0F0F0'

# Créer le dossier figures
import os
os.makedirs('figures', exist_ok=True)
print("Dossier 'figures' créé/vérifié")
print()

# ============================================================================
# GÉNÉRATION DES DONNÉES SYNTHÉTIQUES
# ============================================================================

print("[1/11] Génération des données synthétiques...")
np.random.seed(42)
n_players = 92671

try:
    data = pd.DataFrame({
        'market_value': np.random.lognormal(mean=13, sigma=1.5, size=n_players),
        'age': np.clip(np.random.normal(25.8, 4.2, n_players), 16, 43).astype(int),
        'position': np.random.choice(['Attaquant', 'Milieu', 'Défenseur', 'Gardien'], 
                                      size=n_players, p=[0.307, 0.347, 0.301, 0.045]),
        'league': np.random.choice(['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 
                                    'Ligue 1', 'Autres Top 10', 'Autres'],
                                   size=n_players, p=[0.134, 0.121, 0.118, 0.114, 0.107, 0.197, 0.209])
    })

    # Buts
    position_goals = {'Attaquant': (12, 15), 'Milieu': (5, 10), 'Défenseur': (1, 3), 'Gardien': (0, 0.5)}
    data['goals'] = data['position'].apply(lambda pos: np.clip(np.random.normal(*position_goals[pos]), 0, 127)).astype(int)

    # Passes
    position_assists = {'Attaquant': (8, 10), 'Milieu': (7, 9), 'Défenseur': (2, 4), 'Gardien': (0, 0.5)}
    data['assists'] = data['position'].apply(lambda pos: np.clip(np.random.normal(*position_assists[pos]), 0, 89)).astype(int)

    # Autres variables
    data['minutes_played'] = np.random.gamma(shape=5, scale=500, size=n_players)
    data['injury_days'] = np.random.gamma(shape=2, scale=50, size=n_players)
    data['height'] = np.random.normal(180, 8, n_players)
    data['weight'] = np.random.normal(75, 8, n_players)
    data['caps_national_team'] = np.where(np.random.random(n_players) > 0.5, np.random.poisson(10, n_players), np.nan)

    # Ajuster market_value
    age_factor = 1 + 0.3 * np.exp(-((data['age'] - 26) ** 2) / 40)
    goals_factor = 1 + 0.02 * data['goals']
    league_factor = data['league'].map({
        'Premier League': 1.5, 'La Liga': 1.4, 'Bundesliga': 1.35,
        'Serie A': 1.3, 'Ligue 1': 1.25, 'Autres Top 10': 0.8, 'Autres': 0.4
    })
    injury_factor = 1 - 0.001 * data['injury_days']
    data['market_value'] = data['market_value'] * age_factor * goals_factor * league_factor * injury_factor
    data['market_value'] = np.clip(data['market_value'], 25000, 180000000)

    print(f"Dataset créé : {len(data):,} joueurs, {len(data.columns)} colonnes")
    print()
except Exception as e:
    print(f"ERREUR lors de la génération des données : {e}")
    sys.exit(1)

# ============================================================================
# FONCTION HELPER POUR SAUVEGARDER LES FIGURES
# ============================================================================

def save_figure(fig_num, fig_name, description):
    """Sauvegarde une figure avec gestion d'erreur"""
    try:
        filename = f'figures/figure_{fig_num:02d}_{fig_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure {fig_num} sauvegardée : {filename}")
        print(f"  {description}")
        print()
        return True
    except Exception as e:
        print(f"ERREUR Figure {fig_num} : {e}")
        print()
        return False

# ============================================================================
# FIGURE 1
# ============================================================================

print("[2/11] Figure 1 : Distribution de la valeur marchande...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(data['market_value'] / 1e6, bins=50, color=COLOR_ACCENT, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Valeur marchande (M€)', fontweight='bold')
    axes[0].set_ylabel('Nombre de joueurs', fontweight='bold')
    axes[0].set_title('Distribution de la valeur marchande\n(Échelle normale)', fontweight='bold')
    axes[0].axvline(data['market_value'].mean() / 1e6, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {data["market_value"].mean()/1e6:.2f}M€')
    axes[0].axvline(data['market_value'].median() / 1e6, color='green', linestyle='--', linewidth=2, label=f'Médiane: {data["market_value"].median()/1e6:.2f}M€')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(np.log10(data['market_value']), bins=50, color=COLOR_PRIMARY, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('log₁₀(Valeur marchande)', fontweight='bold')
    axes[1].set_ylabel('Nombre de joueurs', fontweight='bold')
    axes[1].set_title('Distribution de la valeur marchande\n(Échelle logarithmique)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(1, 'distribution_valeur', 'Distributions normale et logarithmique')
except Exception as e:
    print(f"ERREUR Figure 1 : {e}\n")

# ============================================================================
# FIGURE 2
# ============================================================================

print("[3/11] Figure 2 : Distribution de l'âge...")
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(data['age'], bins=range(16, 45), color=COLOR_ACCENT, alpha=0.6, edgecolor='black')
    
    from scipy.stats import gaussian_kde
    density = gaussian_kde(data['age'])
    xs = np.linspace(16, 43, 200)
    ax2 = ax.twinx()
    ax2.plot(xs, density(xs), color='red', linewidth=3, label='Densité')
    ax2.set_ylabel('Densité', fontweight='bold', color='red')
    
    ax.axvline(data['age'].mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Moyenne: {data["age"].mean():.1f} ans')
    ax.set_xlabel('Âge (années)', fontweight='bold')
    ax.set_ylabel('Nombre de joueurs', fontweight='bold')
    ax.set_title('Distribution de l\'âge des joueurs', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_figure(2, 'distribution_age', 'Histogramme avec courbe de densité')
except Exception as e:
    print(f"ERREUR Figure 2 : {e}\n")

# ============================================================================
# FIGURE 3
# ============================================================================

print("[4/11] Figure 3 : Buts par position...")
try:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positions_order = ['Attaquant', 'Milieu', 'Défenseur', 'Gardien']
    bp = ax.boxplot([data[data['position'] == pos]['goals'] for pos in positions_order],
                     labels=positions_order, patch_artist=True, showmeans=True)
    
    colors = [COLOR_PRIMARY, COLOR_ACCENT, '#2C5F2D', '#8B0000']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Position', fontweight='bold')
    ax.set_ylabel('Nombre de buts (3 saisons)', fontweight='bold')
    ax.set_title('Distribution des buts par position', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(3, 'buts_par_position', 'Box plots par position')
except Exception as e:
    print(f"ERREUR Figure 3 : {e}\n")

# ============================================================================
# FIGURE 4
# ============================================================================

print("[5/11] Figure 4 : Âge vs Valeur...")
try:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sample = data.sample(n=min(5000, len(data)), random_state=42)
    scatter = ax.scatter(sample['age'], sample['market_value'] / 1e6, alpha=0.3, s=30, 
                        c=sample['market_value'], cmap='viridis')
    
    z = np.polyfit(data['age'], data['market_value'] / 1e6, 2)
    p = np.poly1d(z)
    age_range = np.linspace(data['age'].min(), data['age'].max(), 100)
    ax.plot(age_range, p(age_range), color='red', linewidth=3, linestyle='--', label='Régression polynomiale')
    
    age_means = data.groupby('age')['market_value'].mean() / 1e6
    ax.plot(age_means.index, age_means.values, color='orange', linewidth=3, marker='o', label='Moyenne par âge')
    
    ax.set_xlabel('Âge (années)', fontweight='bold')
    ax.set_ylabel('Valeur marchande (M€)', fontweight='bold')
    ax.set_title('Relation Âge vs Valeur\n(Hypothèse 2)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Valeur (€)')
    
    plt.tight_layout()
    save_figure(4, 'age_vs_valeur', 'Scatter plot avec régression polynomiale')
except Exception as e:
    print(f"ERREUR Figure 4 : {e}\n")

# ============================================================================
# FIGURE 5
# ============================================================================

print("[6/11] Figure 5 : Valeur par position...")
try:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    positions_order = ['Attaquant', 'Milieu', 'Défenseur', 'Gardien']
    bp = ax.boxplot([data[data['position'] == pos]['market_value'] / 1e6 for pos in positions_order],
                     labels=positions_order, patch_artist=True, showmeans=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Position', fontweight='bold')
    ax.set_ylabel('Valeur marchande (M€)', fontweight='bold')
    ax.set_title('Valeur par position', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(5, 'valeur_par_position', 'Box plots avec échelle log')
except Exception as e:
    print(f"ERREUR Figure 5 : {e}\n")

# ============================================================================
# FIGURE 6
# ============================================================================

print("[7/11] Figure 6 : Buts vs Valeur (attaquants)...")
try:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    attackers = data[data['position'] == 'Attaquant'].copy()
    scatter = ax.scatter(attackers['goals'], attackers['market_value'] / 1e6, 
                        alpha=0.5, s=50, c=attackers['age'], cmap='coolwarm')
    
    z = np.polyfit(attackers['goals'], attackers['market_value'] / 1e6, 1)
    p = np.poly1d(z)
    goals_range = np.linspace(attackers['goals'].min(), attackers['goals'].max(), 100)
    ax.plot(goals_range, p(goals_range), color='red', linewidth=3, linestyle='--', label='Tendance')
    
    corr = attackers['goals'].corr(attackers['market_value'])
    ax.text(0.05, 0.95, f'Corrélation (r) = {corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightgreen'), fontweight='bold')
    
    ax.set_xlabel('Nombre de buts', fontweight='bold')
    ax.set_ylabel('Valeur marchande (M€)', fontweight='bold')
    ax.set_title('Buts vs Valeur (Attaquants)\n(Hypothèse 1)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Âge')
    
    plt.tight_layout()
    save_figure(6, 'buts_vs_valeur_attaquants', 'Impact des buts sur la valeur')
except Exception as e:
    print(f"ERREUR Figure 6 : {e}\n")

# ============================================================================
# FIGURE 7
# ============================================================================

print("[8/11] Figure 7 : Valeur par championnat...")
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    league_stats = data.groupby('league')['market_value'].agg(['mean', 'count']).sort_values('mean', ascending=True)
    league_stats['mean'] = league_stats['mean'] / 1e6
    
    bars = ax.barh(league_stats.index, league_stats['mean'], 
                   color=[COLOR_PRIMARY if 'Autres' in x else COLOR_ACCENT for x in league_stats.index], alpha=0.8)
    
    for i, (idx, row) in enumerate(league_stats.iterrows()):
        ax.text(row['mean'] + 0.1, i, f"{row['mean']:.2f}M€\n(n={int(row['count']):,})", va='center')
    
    ax.set_xlabel('Valeur moyenne (M€)', fontweight='bold')
    ax.set_ylabel('Championnat', fontweight='bold')
    ax.set_title('Valeur par championnat\n(Hypothèse 3)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(7, 'valeur_par_championnat', 'Barres horizontales par championnat')
except Exception as e:
    print(f"ERREUR Figure 7 : {e}\n")

# ============================================================================
# FIGURE 8
# ============================================================================

print("[9/11] Figure 8 : Blessures vs Valeur...")
try:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sample = data.sample(n=min(5000, len(data)), random_state=42)
    scatter = ax.scatter(sample['injury_days'], sample['market_value'] / 1e6, 
                        alpha=0.4, s=50, c=sample['age'], cmap='plasma')
    
    z = np.polyfit(data['injury_days'], data['market_value'] / 1e6, 1)
    p = np.poly1d(z)
    injury_range = np.linspace(0, data['injury_days'].max(), 100)
    ax.plot(injury_range, p(injury_range), color='red', linewidth=3, linestyle='--', label='Tendance négative')
    
    corr = data['injury_days'].corr(data['market_value'])
    ax.text(0.05, 0.95, f'Corrélation (r) = {corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='salmon'), fontweight='bold')
    
    ax.set_xlabel('Jours d\'absence (blessures)', fontweight='bold')
    ax.set_ylabel('Valeur marchande (M€)', fontweight='bold')
    ax.set_title('Impact des blessures\n(Hypothèse 4)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Âge')
    
    plt.tight_layout()
    save_figure(8, 'blessures_vs_valeur', 'Corrélation négative blessures-valeur')
except Exception as e:
    print(f"ERREUR Figure 8 : {e}\n")

# ============================================================================
# FIGURE 9
# ============================================================================

print("[10/11] Figure 9 : Matrice de corrélation...")
try:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numeric_vars = ['market_value', 'age', 'goals', 'assists', 'minutes_played', 
                    'injury_days', 'height', 'weight']
    corr_data = data[numeric_vars].corr()
    corr_data.index = ['Valeur', 'Âge', 'Buts', 'Passes', 'Minutes', 'Blessures', 'Taille', 'Poids']
    corr_data.columns = corr_data.index
    
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, linewidths=1, ax=ax)
    
    ax.set_title('Matrice de corrélation', fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_figure(9, 'matrice_correlation', 'Heatmap des corrélations')
except Exception as e:
    print(f"ERREUR Figure 9 : {e}\n")

# ============================================================================
# FIGURE 10
# ============================================================================

print("[11/11] Figure 10 : Interaction Âge × Position...")
try:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    positions = ['Attaquant', 'Milieu', 'Défenseur', 'Gardien']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    
    for position, color, marker in zip(positions, colors, markers):
        pos_data = data[data['position'] == position].groupby('age')['market_value'].mean() / 1e6
        ax.plot(pos_data.index, pos_data.values, label=position, color=color, 
                linewidth=3, marker=marker, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Âge (années)', fontweight='bold')
    ax.set_ylabel('Valeur moyenne (M€)', fontweight='bold')
    ax.set_title('Interaction Âge × Position\n(Pics différenciés)', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 36)
    
    plt.tight_layout()
    save_figure(10, 'interaction_age_position', 'Courbes multiples par position')
except Exception as e:
    print(f"ERREUR Figure 10 : {e}\n")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("=" * 80)
print("GÉNÉRATION TERMINÉE !")
print("=" * 80)
print()

# Vérifier combien de figures ont été créées
import glob
figures_created = glob.glob('figures/figure_*.png')
print(f"{len(figures_created)}/10 figures générées avec succès !")
print()

for fig_file in sorted(figures_created):
    size_kb = os.path.getsize(fig_file) / 1024
    print(f"   {os.path.basename(fig_file)} ({size_kb:.0f} KB)")

print()
print("Toutes les figures sont dans : ./figures/")
print()
print("=" * 80)