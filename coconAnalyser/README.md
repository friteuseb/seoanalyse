# Cocon Semantique Analyzer

Un outil d'analyse de cocons sémantiques qui permet d'évaluer la structure et la qualité des liens internes d'un site web.

## Fonctionnalités

- Analyse de la structure des liens internes
- Détection des clusters thématiques
- Calcul des scores de cohérence sémantique
- Identification des pages stratégiques
- Génération de rapports détaillés
- Calcul du PageRank interne
- Analyse des flux thématiques entre clusters

## Installation

```bash
git clone https://github.com/votre-username/cocon-semantique-analyzer.git
cd cocon-semantique-analyzer
pip install -r requirements.txt
```

## Utilisation

```bash
python analyze_cocoon_enhanced.py <crawl_id>
```

Le script générera un rapport détaillé incluant :
- Métriques clés du cocon
- Analyse des clusters thématiques
- Pages stratégiques
- Points d'amélioration
- Score global
- Flux thématique
- Cohérence sémantique

## Métriques et Calculs

### PageRank Interne
Le PageRank est calculé selon l'algorithme original de Google avec :
- Un facteur d'amortissement (damping factor) de 0.85
- Une distribution de probabilité uniforme initiale
- Une convergence basée sur un seuil de 1e-6

La formule utilisée est :
```
PR(A) = (1-d)/N + d * sum(PR(Bi)/C(Bi))
```
où :
- d est le facteur d'amortissement (0.85)
- N est le nombre total de pages
- Bi sont les pages qui pointent vers A
- C(Bi) est le nombre de liens sortants de la page Bi

### Score de Cohérence des Clusters
La cohérence d'un cluster est calculée selon plusieurs facteurs :
1. Densité interne : ratio entre les liens internes réels et le nombre maximum possible de liens internes
   ```
   densité = liens_internes / (n * (n-1))
   ```
   où n est le nombre de pages dans le cluster

2. Score de cohérence : ratio entre les liens internes et externes
   ```
   cohérence = liens_internes / (liens_internes + liens_externes)
   ```

### Score Global de Qualité
Le score global (0-100) est calculé selon la formule :
```
base_score = (min(avg_internal_links / 5, 1) * 40) +  # Liens internes
             ((1 - connectivity_ratio) * 30) +         # Connectivité
             (max(0, 1 - avg_depth/5) * 30)           # Profondeur
```

Des pénalités sont appliquées :
- Pages orphelines : -50% × (pages_orphelines/total_pages)
- Pages trop profondes (>3 clics) : -50% × (pages_profondes/total_pages)

### Classification des Clusters
Les clusters sont évalués selon leur cohésion :
- Excellent : cohésion > 1
- Correct : 0.5 ≤ cohésion ≤ 1
- À renforcer : cohésion < 0.5

### Flux Thématique
Le flux thématique analyse les liens entre clusters pour mesurer :
1. La force des connexions inter-clusters
2. L'équilibre des liens entrants/sortants
3. La centralité de chaque cluster dans la structure globale

## Structure du projet

- `cocon_analyzer.py`: Analyseur de base
- `enhanced_analyzer.py`: Version enrichie avec analyses supplémentaires
- `analyze_cocoon_enhanced.py`: Script principal

## Dépendances

- NetworkX
- Redis
- NumPy

## Interprétation des Résultats

### Scores de Page
- PageRank > 0.1 : Page très centrale
- PageRank 0.05-0.1 : Page importante
- PageRank < 0.05 : Page secondaire

### Qualité du Cocon
- 90-100 : Excellent cocon sémantique
- 70-89 : Bon cocon avec optimisations possibles
- 50-69 : Améliorations nécessaires
- <50 : Structure à retravailler en profondeur

### Connectivité
- Ratio entrée/sortie proche de 1 : Page équilibrée
- Ratio > 2 : Page autorité (beaucoup de liens entrants)
- Ratio < 0.5 : Page de distribution (beaucoup de liens sortants)
