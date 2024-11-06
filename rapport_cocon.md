# documentation.md
"""
# Guide d'Analyse du Cocon Sémantique

## Métriques Clés

### PageRank (0-1)
- Algorithme mesurant l'importance des pages selon les liens entrants/sortants
- > 0.02 : Page très importante
- 0.01-0.02 : Page moyennement importante 
- < 0.01 : Page peu centrale

### Liens
- Entrants : Pages pointant vers la page analysée
- Sortants : Liens vers d'autres pages
- Ratio optimal entrée/sortie : proche de 1

### Profondeur
- Idéale : ≤ 3 clics depuis la racine
- Impact SEO négatif au-delà de 4 clics

### Clusters
- Cohésion : Mesure des liens entre pages d'un même thème
- > 1 : Excellent maillage thématique
- < 0.5 : Thématique à renforcer

## Score Global
Score sur 100 calculé selon :
- Maillage (40%) : Distribution des liens
- Connectivité (30%) : Cohésion des sections
- Architecture (30%) : Profondeur et accessibilité
"""
