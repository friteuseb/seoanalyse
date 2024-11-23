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





# CIR Compare 
# Tableau des Métriques d'Analyse du Cocon Sémantique

## 1. Métriques de Structure

| Métrique | Description | Formule | Valeur Optimale | Interprétation |
|----------|-------------|---------|-----------------|----------------|
| Densité du graphe | Ratio liens existants/possibles | liens / (pages * (pages-1)) | 0.1-0.3 | Mesure l'équilibre entre navigation et hiérarchie |
| Profondeur moyenne | Moyenne des chemins depuis racine | avg(shortest_paths) | 1.5-2.5 | Évalue l'efficacité de la structure hiérarchique |
| Accessibilité à 3 clics | % pages proches de la racine | (pages_≤3_clics / total) * 100 | >80% | Indique la facilité de navigation |

## 2. Métriques de Distribution

| Métrique | Description | Formule | Valeur Optimale | Interprétation |
|----------|-------------|---------|-----------------|----------------|
| Entropie PageRank | Distribution de l'autorité | -Σ(PR_i * log2(PR_i)) | >3.0 | Mesure l'équilibre de la distribution du PageRank |
| Balance thématique | Ratio intra/inter clusters | liens_intra / liens_inter | 1.0-2.0 | Évalue l'équilibre entre cohésion et connectivité |

## 3. Scores Composites

| Score | Composantes | Pondération | Objectif |
|-------|-------------|-------------|-----------|
| Score Global | Moyenne pondérée | 0.3 * densité + 0.3 * accessibilité + 0.2 * entropie + 0.2 * balance | >70% | 
| Score Technique | Métriques structurelles | Selon impact SEO | >80% |
| Score UX | Métriques navigation | Selon comportement utilisateur | >75% |


# Métriques d'Analyse des Cocons Sémantiques Émergents

## 1. Mesures de Cohérence

| Métrique | Description | Optimum | Interprétation |
|----------|-------------|---------|----------------|
| Similarité Intra-Cluster | Cohésion sémantique moyenne des pages liées | > 0.7 | Mesure la pertinence naturelle des connexions |
| Équilibre Inter-Clusters | Distribution des liens entre groupes thématiques | 0.3-0.5 | Force des ponts thématiques |
| Émergence Structurelle | Degré de formation naturelle des clusters | > 0.6 | Opposition à une structure forcée |

## 2. Métriques de Flux

| Métrique | Description | Optimum | Interprétation |
|----------|-------------|---------|----------------|
| Efficacité du Transfert | % de PageRank conservé lors des transmissions | > 0.8 | Mesure les pertes de jus de lien |
| Distribution d'Autorité | Entropie de la distribution du PageRank | > 3.0 | Équilibre vs concentration |
| Points d'Accumulation | Identification des hubs naturels d'autorité | N/A | Cartographie des points forts |

## 3. Accessibilité Naturelle

| Métrique | Description | Optimum | Interprétation |
|----------|-------------|---------|----------------|
| Densité Émergente | Ratio liens existants/possibles | 0.1-0.3 | Équilibre naturel du maillage |
| Profondeur Effective | Distribution des chemins d'accès | < 3.5 | Facilité de navigation |
| Résilience | Robustesse aux suppressions de liens | > 0.6 | Stabilité de la structure |