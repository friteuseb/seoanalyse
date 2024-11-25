# Analyse Comparative des Modèles d'Embedding

## Métriques Détaillées
|    | Modèle        |   Threshold |   Liens totaux | Pages Orphelines      | Densité du maillage   | Accessibilité à 3 clics   |   Distribution PageRank |   Efficacité du transfert |   Ratio thématique |   Score Global |
|---:|:--------------|------------:|---------------:|:----------------------|:----------------------|:--------------------------|------------------------:|--------------------------:|-------------------:|---------------:|
|  3 | ada2-t60      |          60 |            180 | 0 (0.0%) [-100.0%] ✅ | 0.47 (+1231.6%) ⚠️    | 100.0% (+666.7%) ⚠️       |                    4.19 |                      0.94 |               0.6  |           66   |
|  2 | camembert-t60 |          60 |            159 | 0 (0.0%) [-100.0%] ✅ | 0.42 (+1076.2%) ⚠️    | 90.0% (+590.0%) ⚠️        |                    4.08 |                      0.93 |               0.56 |           64.2 |
|  1 | ada3-t60      |          60 |             28 | 3 (15.8%) [-72.7%] ✅ | 0.07 (+107.1%) ⚠️     | 20.0% (+53.3%) ⚠️         |                    4.2  |                      0.85 |               0.96 |           30.3 |
|  4 | minilm-t20    |          20 |             20 | 6 (31.6%) [-45.5%] ✅ | 0.05 (+48.0%) ⚠️      | 15.0% (+15.0%) ⚠️         |                    4.06 |                      0.7  |               1    |           24.9 |
|  0 | cocon-t0      |           0 |             18 | 11 (50.0%) [+0.0%] ✅ | 0.04 (+0.0%) ⚠️       | 13.0% (+0.0%) ⚠️          |                    4.1  |                      0.52 |               0.88 |           22.5 |

## Matrice de Comparaison des Liens
|               | cocon-base        | cocon-t0          | ada3-t60           | camembert-t60        | ada2-t60             | minilm-t20         |
|:--------------|:------------------|:------------------|:-------------------|:---------------------|:---------------------|:-------------------|
| cocon-base    | 100.0% (18/18/18) | 100.0% (18/18/18) | 77.8% (14/18/28)   | 94.4% (17/18/159)    | 94.4% (17/18/180)    | 66.7% (12/18/20)   |
| cocon-t0      | 100.0% (18/18/18) | 100.0% (18/18/18) | 77.8% (14/18/28)   | 94.4% (17/18/159)    | 94.4% (17/18/180)    | 66.7% (12/18/20)   |
| ada3-t60      | 77.8% (14/28/18)  | 77.8% (14/28/18)  | 100.0% (28/28/28)  | 100.0% (28/28/159)   | 100.0% (28/28/180)   | 100.0% (20/28/20)  |
| camembert-t60 | 94.4% (17/159/18) | 94.4% (17/159/18) | 100.0% (28/159/28) | 100.0% (159/159/159) | 70.4% (112/159/180)  | 100.0% (20/159/20) |
| ada2-t60      | 94.4% (17/180/18) | 94.4% (17/180/18) | 100.0% (28/180/28) | 70.4% (112/180/159)  | 100.0% (180/180/180) | 100.0% (20/180/20) |
| minilm-t20    | 66.7% (12/20/18)  | 66.7% (12/20/18)  | 100.0% (20/20/28)  | 100.0% (20/20/159)   | 100.0% (20/20/180)   | 100.0% (20/20/20)  |

## Guide d'Interprétation

=== GUIDE D'INTERPRÉTATION DES MÉTRIQUES ===


## Pages Orphelines
Description: Pages qui n'ont aucun lien entrant (sauf la page d'accueil)
Interprétation:
• Format: nombre (pourcentage) [variation%] status
• ✅ : Réduction ou stabilité du nombre de pages orphelines
• ⚠️ : Augmentation du nombre de pages orphelines

Seuils recommandés:
• Optimal: < 5% du total des pages
• Acceptable: 5-10% du total des pages
• Problématique: > 10% du total des pages

## Métriques Fondamentales

### Densité du maillage
Description: Ratio entre liens existants et possibles
Plage optimale: 0.1-0.3
Interprétation:
- < 0.1 : Maillage insuffisant
- 0.1-0.3 : Navigation optimale
- > 0.3 : Dilution du PageRank

### Accessibilité à 3 clics
Description: Pourcentage de pages accessibles en 3 clics
Plage optimale: > 80%
Interprétation:
- < 60% : Navigation complexe
- > 80% : Bonne accessibilité
- = 100% : Structure très plate

## Flux de PageRank

### Distribution PageRank
Description: Entropie de la distribution du PageRank
Plage optimale: > 3.0
Interprétation:
- < 2.0 : Concentration excessive
- > 3.0 : Bonne distribution
- > 4.0 : Distribution très uniforme

### Efficacité du transfert
Description: Conservation du PageRank dans les transmissions
Plage optimale: > 0.7
Interprétation:
- < 0.5 : Pertes importantes
- > 0.7 : Bon transfert
- > 0.9 : Transfert optimal

## Cohérence Sémantique

### Ratio thématique
Description: Pertinence sémantique des liens
Plage optimale: > 0.6
Interprétation:
- < 0.4 : Faible cohérence
- > 0.6 : Bonne cohérence
- > 0.8 : Très forte cohérence

=== DÉTAILS DES PAGES ORPHELINES ===

cocon-t0 :
• 11 (50.0%) [+0.0%] ✅
Exemples d'URLs orphelines :
- 404
- the-essentials-of-chess-history
- all-about-chess-variants
- exploring-chess-strategy
- our-articles

ada3-t60 :
• 3 (15.8%) [-72.7%] ✅
Exemples d'URLs orphelines :
- all-about-chess-variants
- the-essentials-of-tennis-history-and-legends
- discover-tennis-techniques

camembert-t60 :
• 0 (0.0%) [-100.0%] ✅

ada2-t60 :
• 0 (0.0%) [-100.0%] ✅

minilm-t20 :
• 6 (31.6%) [-45.5%] ✅
Exemples d'URLs orphelines :
- the-essentials-of-chess-psychology
- the-essentials-of-chess-history
- discover-tennis-equipment
- discover-tennis-techniques
- all-about-chess-variants


## Détail des Pages Orphelines

### ada2-t60
- 0 (0.0%) [-100.0%] ✅

### camembert-t60
- 0 (0.0%) [-100.0%] ✅

### ada3-t60
- 3 (15.8%) [-72.7%] ✅

### minilm-t20
- 6 (31.6%) [-45.5%] ✅

### cocon-t0
- 11 (50.0%) [+0.0%] ✅
