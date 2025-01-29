# SEO Analyse

Ce projet est une application d'analyse sémantique et de visualisation de liens internes d'un site web. Il utilise des crawlers pour extraire le contenu des pages web, analyse les données sémantiques et visualise les résultats sous forme de graphes interactifs.

## Fonctionnalités

1. **Crawl de sites web** : Extraction du contenu des pages web.
2. **Analyse sémantique** : Détermination des thématiques dominantes à partir du contenu.
3. **Visualisation des liens internes** : Affichage des liens internes sous forme de graphes interactifs.
4. **Clusterisation** : Regroupement des pages en clusters basés sur le contenu sémantique.
5. **Assignation de couleurs de cluster** : Harmonisation des couleurs pour une meilleure visualisation.
6. **Filtrage des pages** : Exclusion de pages spécifiques basée sur des patterns d'URLs.

## Fonctionnement de l'application

L'application de Visualisation Sémantique fonctionne en plusieurs étapes interconnectées, comme illustré dans le schéma ci-dessous :

![Schéma fonctionnel de l'application](schema_fonctionnel.png)

Ce schéma montre le flux de travail de l'application :

1. Le script principal `main.py` orchestre l'ensemble du processus.
2. `01_crawl.py` crawle le site web et stocke les données dans Redis.
3. `03_crawl_internal_links.py` extrait les liens internes des pages crawlées.
4. `02_analyse.py` effectue l'analyse sémantique des données.
5. Les résultats sont utilisés pour générer des graphes de visualisation.
6. Ces graphes sont présentés via une interface web interactive.

Redis joue un rôle central dans ce processus, servant de stockage intermédiaire pour les données à chaque étape. Cette architecture permet une analyse efficace et une visualisation dynamique des résultats.

## Prérequis

- Python 3.7 ou supérieur
- Redis
- Les bibliothèques Python suivantes :
  - `redis`
  - `trafilatura`
  - `beautifulsoup4`
  - `sentence-transformers`
  - `sklearn`
  - `matplotlib`
  - `mpld3`
  - `nltk`

## Installation

Créez un environnement virtuel : 
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

1. Clonez le dépôt GitHub :
   ```sh
   git clone https://github.com/friteuseb/seoanalyse.git
   cd seoanalyse
   ```

2. Installez les dépendances :
   ```sh
   pip install -r requirements.txt
   ```

3. Assurez-vous que Redis est installé et en cours d'exécution.

4. Téléchargez les stopwords français pour NLTK :
   ```sh
   python -m nltk.downloader stopwords
   ```

## Utilisation

### Options de lancement

Il existe plusieurs façons d'utiliser l'application en fonction de vos besoins :

1. **Analyse basique** :
```sh
python3 main.py https://example.com ".content"
```

2. **Exclusion de pages par pattern d'URL** :
```sh
python3 main.py https://example.com ".content" -e pattern1 pattern2 pattern3
```
Par exemple, pour exclure les pages en anglais et les profils :
```sh
python3 main.py https://www.cnrs.fr ".main-column" -e en person personne
```

3. **Désactivation de la clusterisation** :
```sh
python3 main.py https://example.com ".content" --no-cluster
```

4. **Combinaison des options** :
```sh
python3 main.py https://example.com ".content" -e pattern1 pattern2 --no-cluster
```

### Exclusion de zones dans le DOM

Vous pouvez utiliser les sélecteurs CSS pour exclure des zones spécifiques :

1. **Exclusion de classes ou IDs** :
```sh
python3 main.py https://example.com "#content:not(.menu):not(.footer)"
```
ou
```sh
python3 main.py https://example.com ".content:not(#menu):not(.sidebar)"
```

2. **Utilisation des attributs role** :
```sh
python3 main.py https://example.com "[role='main']"
python3 main.py https://example.com "main:not(nav):not(footer)"
```

### Visualisation des résultats

Les résultats sont sauvegardés dans Redis et peuvent être visualisés via des graphiques interactifs générés par D3.js et HTML.

## Structure du projet

- `01_crawl.py` : Script de crawling des sites web.
- `02_analyse.py` : Script d'analyse sémantique et de clusterisation.
- `03_crawl_internal_links.py` : Script d'analyse des liens internes et mise à jour des fichiers JSON.
- `main.py` : Script principal pour lancer toutes les étapes de l'analyse.
- `script.js` : Script JavaScript pour la visualisation des graphes interactifs.
- `style.css` : Fichier CSS pour le style de la page de visualisation.
- `requirements.txt` : Liste des dépendances Python.

## Contribution

Les contributions sont les bienvenues ! Veuillez soumettre des issues et des pull requests pour toute amélioration ou bug.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.