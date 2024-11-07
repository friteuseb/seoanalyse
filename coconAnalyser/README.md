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

## Structure du projet

- `cocon_analyzer.py`: Analyseur de base
- `enhanced_analyzer.py`: Version enrichie avec analyses supplémentaires
- `analyze_cocoon_enhanced.py`: Script principal

## Dépendances

- NetworkX
- Redis
- NumPy
