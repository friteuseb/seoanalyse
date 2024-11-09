# Analyse Comparative de Crawls : Documentation Technique

## Vue d'ensemble

Le script permet d'analyser et de comparer deux crawls d'un site web pour évaluer l'impact des modifications de maillage interne. Il utilise plusieurs métriques regroupées en catégories distinctes pour fournir une analyse complète des changements structurels.

## Métriques Analysées

### 1. Métriques Structurelles
```python
structural_metrics = {
    'average_clustering': float,    # Cohésion des groupes de pages
    'reciprocity': float,          # Taux de liens bidirectionnels
    'density': float,              # Ratio liens existants/possibles
    'bilingual_links': int,        # Nombre de liens entre versions linguistiques
    'thematic_links': int,         # Liens entre pages de même thème
    'cross_thematic_links': int    # Liens entre thèmes différents
}
```

### 2. Métriques d'Accessibilité
```python
accessibility_metrics = {
    'mean_depth': float,           # Profondeur moyenne des pages
    'depth_variance': float,       # Variance de la profondeur
    'max_depth': float,           # Profondeur maximale
    'pagerank_entropy': float,     # Distribution de l'autorité
    'pages_within_3_clicks': float # Ratio de pages accessibles en 3 clics
}
```

### 3. Métriques de Clusters
```python
cluster_metrics = {
    'number_of_clusters': int,     # Nombre de groupes thématiques
    'average_cluster_size': float, # Taille moyenne des clusters
    'cluster_density': float,      # Cohésion interne des clusters
    'inter_cluster_links': int     # Liens entre clusters
}
```

## Fonctionnement

### 1. Collecte des Données
- Connexion à Redis pour récupérer les crawls
- Extraction des métadonnées de chaque page
- Construction du graphe de liens

### 2. Calcul des Métriques
```python
def calculate_scientific_metrics(self):
    """Calcule toutes les métriques pour un crawl"""
    return {
        'structural_metrics': self._calculate_structural_metrics(),
        'semantic_metrics': self._calculate_semantic_metrics(),
        'accessibility_metrics': self._calculate_accessibility_metrics(),
        'cluster_metrics': self._calculate_cluster_metrics()
    }
```

### 3. Comparaison des Métriques
```python
def format_change(self, before, after):
    """Calcule et formate les changements entre deux valeurs"""
    if before == 0:
        return f"{after:.2f} (nouveau)"
    
    change = ((after - before) / before) * 100
    # Formatage avec indicateurs visuels
```

### 4. Génération du Rapport
```python
def generate_scientific_report(self, metrics1, metrics2):
    """Génère un rapport comparatif détaillé"""
    # Structure du rapport :
    # 1. Évolution globale
    # 2. Qualité du maillage
    # 3. Accessibilité
    # 4. Structure des clusters
    # 5. Synthèse et recommandations
```

## Utilisation du Script

### 1. Prérequis
```bash
pip install redis networkx numpy scipy tabulate
```

### 2. Format des Données dans Redis
```json
{
    "crawl_id:doc:page_id": {
        "url": "string",
        "internal_links_out": ["url1", "url2"],
        "cluster": "int",
        "content_length": "int",
        "crawl_date": "datetime"
    }
}
```

### 3. Exécution
```bash
python compare_crawls.py
```

### 4. Sélection des Crawls
- Liste des crawls disponibles dans Redis
- Sélection par numéro des deux crawls à comparer

## Interprétation des Résultats

### 1. Indicateurs de Changement
- `↑↑` : Amélioration majeure (>100%)
- `↑` : Amélioration significative (30-100%)
- `→` : Changement modéré (<30%)
- `↓` : Dégradation significative (>30% négatif)

### 2. Recommandations Automatiques
- Basées sur des seuils prédéfinis
- Adaptées aux métriques dégradées
- Suggestions d'amélioration contextuelles

## Extension du Script

### 1. Ajout de Nouvelles Métriques
```python
def _calculate_custom_metric(self):
    """Template pour ajouter une nouvelle métrique"""
    try:
        # Calculs personnalisés
        return result
    except Exception as e:
        print(f"Erreur : {str(e)}")
        return 0
```

### 2. Personnalisation des Seuils
```python
class Config:
    """Configuration des seuils d'analyse"""
    DENSITY_THRESHOLD = 0.15
    ACCESSIBILITY_THRESHOLD = 0.9
    CLUSTER_DENSITY_THRESHOLD = 0.3
    SIGNIFICANT_CHANGE = 30
```

### 3. Export des Résultats
- Format texte par défaut
- Possibilité d'export en JSON/CSV
- Génération de graphiques avec matplotlib

## Bonnes Pratiques

1. **Régularité des Analyses**
   - Analyser avant/après chaque modification majeure
   - Suivre l'évolution dans le temps

2. **Validation des Données**
   - Vérifier la cohérence des crawls
   - S'assurer de la comparabilité des versions

3. **Interprétation Contextuelle**
   - Considérer les objectifs spécifiques
   - Tenir compte des contraintes techniques

4. **Documentation des Changements**
   - Noter les modifications effectuées
   - Corréler avec les variations de métriques