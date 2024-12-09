import redis
import numpy as np
import json
import nltk
import sys
import logging
import subprocess
import torch
import warnings
import pandas as pd 
from transformers import CamembertModel, CamembertTokenizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords

# Ignorer les avertissements spécifiques
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_redis_port():
    try:
        port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
        return int(port.strip())
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

# Connexion à Redis
r = redis.Redis(host='localhost', port=get_redis_port(), db=0)


def get_documents_from_redis(crawl_id):
    documents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        doc_id = key.decode('utf-8')
        url = doc_data[b'url'].decode('utf-8')
        content = doc_data[b'content'].decode('utf-8')
        internal_links_out = json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
        documents.append({
            "doc_id": doc_id,
            "url": url,
            "content": content,
            "internal_links_out": internal_links_out
        })
    if not documents:
        logging.warning(f"No documents found for crawl ID {crawl_id}")
    return documents

def save_results_to_redis(crawl_id, documents, clusters, labels):
    """Sauvegarde des résultats dans Redis avec gestion des outliers."""
    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        cluster_id = int(clusters[i])
        label = labels.get(cluster_id, "Non classé")
        r.hset(doc_id, mapping={
            "cluster": cluster_id,
            "label": label
        })

def create_display_label(url):
    """Crée un label d'affichage lisible à partir d'une URL."""
    # Supprimer le protocole
    clean_url = url.replace('https://', '').replace('http://', '')
    
    # Supprimer le trailing slash
    clean_url = clean_url.rstrip('/')
    
    # Diviser l'URL en parties
    parts = clean_url.split('/')
    
    if len(parts) <= 1:
        # C'est juste un domaine
        return clean_url
    
    # Retourner la dernière partie non-vide de l'URL
    for part in reversed(parts):
        if part:
            return part
    
    return url  # Fallback au cas où

def save_graph_to_redis(crawl_id, documents, clusters, labels):
    nodes = []
    links = []
    url_to_index = {}
    urls_set = set()  # Pour suivre toutes les URLs connues

    # Debug des documents
    logging.info(f"Début du traitement des documents...")
    
    # Premièrement, collectons toutes les URLs valides
    for doc in documents:
        url = doc["url"].rstrip('/')  # Normaliser les URLs en retirant le slash final
        urls_set.add(url)
        if url.endswith('/'):
            urls_set.add(url[:-1])  # Ajouter aussi la version sans slash
        else:
            urls_set.add(url + '/')  # Ajouter aussi la version avec slash

    logging.info(f"URLs valides collectées: {len(urls_set)}")

    # Création des nœuds et collecte des liens
    for i, doc in enumerate(documents):
        url = doc["url"].rstrip('/')
        url_to_index[url] = i
        doc_id = doc["doc_id"]
        
        # Récupération et parsing des liens internes
        try:
            doc_data = r.hgetall(doc_id)
            internal_links_raw = doc_data.get(b'internal_links_out', b'[]').decode('utf-8')
            internal_links = json.loads(internal_links_raw)
            
            logging.info(f"\nAnalyse de {url}:")
            logging.info(f"Nombre de liens trouvés: {len(internal_links)}")
            
            if internal_links:
                for link in internal_links:
                    normalized_link = link.strip().rstrip('/')
                    # Vérifier si le lien existe dans notre ensemble d'URLs valides
                    if normalized_link in urls_set:
                        links.append({
                            "source": url,
                            "target": normalized_link,
                            "value": 1
                        })
                    else:
                        logging.info(f"Lien ignoré: {link} -> n'existe pas dans les URLs valides")
                        
        except json.JSONDecodeError as e:
            logging.error(f"Erreur de décodage JSON pour {url}: {e}")
            continue
        except Exception as e:
            logging.error(f"Erreur lors du traitement des liens pour {url}: {e}")
            continue

        display_label = create_display_label(url)
        nodes.append({
            "id": url,
            "label": display_label,
            "group": int(clusters[i]),
            "title": labels[clusters[i]],
            "internal_links_count": len(internal_links) if 'internal_links' in locals() else 0
        })

    # Filtrer et compter les liens
    valid_links = []
    link_count = {}  # Pour compter les liens par URL

    for link in links:
        source = link["source"].rstrip('/')
        target = link["target"].rstrip('/')
        
        if source in urls_set and target in urls_set:
            valid_links.append(link)
            # Compter les liens
            if source not in link_count:
                link_count[source] = {"out": 0, "in": 0}
            if target not in link_count:
                link_count[target] = {"out": 0, "in": 0}
            link_count[source]["out"] += 1
            link_count[target]["in"] += 1

    # Afficher les statistiques détaillées
    logging.info("\n📊 Statistiques très détaillées du graphe:")
    logging.info(f"Nombre de nœuds: {len(nodes)}")
    logging.info(f"Nombre total de liens trouvés: {len(links)}")
    logging.info(f"Nombre de liens valides après filtrage: {len(valid_links)}")
    
    # Afficher les URLs avec le plus de liens entrants/sortants
    sorted_by_in = sorted(link_count.items(), key=lambda x: x[1]["in"], reverse=True)[:5]
    sorted_by_out = sorted(link_count.items(), key=lambda x: x[1]["out"], reverse=True)[:5]
    
    logging.info("\nTop 5 des pages avec le plus de liens entrants:")
    for url, counts in sorted_by_in:
        logging.info(f"{url}: {counts['in']} liens entrants")
    
    logging.info("\nTop 5 des pages avec le plus de liens sortants:")
    for url, counts in sorted_by_out:
        logging.info(f"{url}: {counts['out']} liens sortants")

    graph_data = {
        "nodes": nodes,
        "links": valid_links
    }

    # Sauvegarde dans Redis avec logging détaillé
    try:
        # S'assurer que le crawl_id est bien formaté
        base_crawl_id = crawl_id.split(':')[0] if ':' in crawl_id else crawl_id
        
        # Construire les clés
        simple_key = f"{base_crawl_id}_simple_graph"
        clustered_key = f"{base_crawl_id}_clustered_graph"
        
        logging.info(f"Tentative de sauvegarde des graphes...")
        logging.info(f"Clé simple: {simple_key}")
        logging.info(f"Clé clustered: {clustered_key}")
        
        # Sérialiser les données une seule fois
        graph_json = json.dumps(graph_data, default=convert_to_serializable)
        
        # Sauvegarder les deux versions
        r.set(simple_key, graph_json)
        r.set(clustered_key, graph_json)
        
        # Vérifier immédiatement la sauvegarde
        if r.exists(simple_key) and r.exists(clustered_key):
            logging.info("✅ Graphes sauvegardés avec succès dans Redis")
            logging.info(f"Taille des données : {len(graph_json)} bytes")
            
            # Liste toutes les clés liées à ce crawl pour debug
            keys = list(r.scan_iter(f"{base_crawl_id}*"))
            logging.info("Clés Redis associées à ce crawl :")
            for key in keys:
                logging.info(f"- {key.decode('utf-8')}")
        else:
            logging.error("❌ Échec de la vérification post-sauvegarde")
            
    except Exception as e:
        logging.error(f"❌ Erreur lors de la sauvegarde dans Redis: {str(e)}")
        logging.error(f"Crawl ID: {crawl_id}")
        logging.error(f"Base crawl ID: {base_crawl_id}")
        raise

def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_ddev_url():
    """Récupère l'URL DDEV du projet."""
    try:
        cmd = "ddev describe -j | jq -r '.raw.name'"
        project_name = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        return f"https://{project_name}.ddev.site"
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de l'URL DDEV: {e}")
        return "http://localhost"



class SemanticAnalyzer:
    def __init__(self):
        # Charger CamemBERT
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
            self.model = CamembertModel.from_pretrained('camembert-base')
        self.model.eval()
        
        # Préparer les stop words comme une liste
        try:
            nltk.download('stopwords', quiet=True)
            self.french_stop_words = list(stopwords.words('french'))
        except Exception as e:
            logging.warning(f"Erreur lors du chargement des stop words: {e}")
            self.french_stop_words = None

        
    def compute_embeddings(self, texts, batch_size=8):
        """Génère des embeddings par lots pour optimiser la mémoire."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.numpy())
                
        return np.vstack(all_embeddings)
    

    def cluster_documents(self, embeddings):
        """Clustering optimisé avec DBSCAN."""
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        best_silhouette = -1
        best_labels = None
        best_eps = None
        best_outliers_ratio = 1.0
        
        # Ajuster les paramètres pour avoir plus de clusters
        distances = np.linalg.norm(embeddings_scaled[:, None] - embeddings_scaled, axis=2)
        eps_range = np.percentile(distances, [5, 10, 15, 20, 25, 30])
        
        logging.info("Début de la recherche des meilleurs paramètres de clustering...")
        
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(embeddings_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            outliers_count = np.sum(labels == -1)
            outliers_ratio = outliers_count / len(labels)
            
            logging.info(f"""
            Test avec eps={eps:.3f}:
            - Nombre de clusters: {n_clusters}
            - Outliers: {outliers_count}/{len(labels)} ({outliers_ratio*100:.1f}%)
            """)
            
            if n_clusters >= 2:  # On veut au moins 2 clusters
                mask = labels != -1
                if np.sum(mask) > 1:
                    score = silhouette_score(embeddings_scaled[mask], labels[mask])
                    logging.info(f"- Score silhouette: {score:.3f}")
                    
                    # Favoriser les solutions avec plus de clusters et moins d'outliers
                    current_score = score * (1 - outliers_ratio) * (n_clusters / 10)  
                    
                    if current_score > best_silhouette:
                        best_silhouette = current_score
                        best_labels = labels
                        best_eps = eps
                        best_outliers_ratio = outliers_ratio
        
        if best_labels is None:
            logging.warning("Pas de clustering optimal trouvé, utilisation de paramètres par défaut")
            # Utiliser des paramètres plus stricts pour forcer plus de clusters
            dbscan = DBSCAN(eps=0.3, min_samples=2)
            best_labels = dbscan.fit_predict(embeddings_scaled)
            best_eps = 0.3
            best_outliers_ratio = np.sum(best_labels == -1) / len(best_labels)

        # Résumé final
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        cluster_counts = {i: np.sum(best_labels == i) for i in set(best_labels)}
        cluster_info = "\n".join([f"Cluster {k}: {v} documents" for k, v in cluster_counts.items()])
        
        logging.info(f"""
        Meilleurs paramètres trouvés:
        - Epsilon: {best_eps:.3f}
        - Score silhouette ajusté: {best_silhouette:.3f}
        - Nombre de clusters: {n_clusters}
        - Ratio d'outliers: {best_outliers_ratio*100:.1f}%
        
        Distribution des clusters:
        {cluster_info}
        """)
        
        return best_labels

    def extract_topics(self, texts, labels):
        """Extraction des thèmes avec gestion appropriée des stop words."""
        try:
            # Configuration du vectorizer avec stop_words comme liste ou None
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words=self.french_stop_words if self.french_stop_words else None
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            topics = {}
            unique_labels = set(labels[labels != -1])
            
            for label in unique_labels:
                cluster_docs = tfidf_matrix[labels == label]
                avg_tfidf = cluster_docs.mean(axis=0).A1
                
                # Sélectionner les termes les plus pertinents
                top_indices = avg_tfidf.argsort()[-5:][::-1]
                topics[int(label)] = ' '.join([feature_names[i] for i in top_indices])
            
            # Étiquette pour les outliers
            if -1 in labels:
                topics[-1] = "Contenu non classé"
            
            return topics
            
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction des thèmes: {e}")
            # Fallback sur des étiquettes simples
            unique_labels = set(labels)
            return {label: f"Cluster {label}" for label in unique_labels}



def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python3 02_analyse.py <crawl_id>")
        return
    
    try:
        crawl_id = sys.argv[1]
        documents = get_documents_from_redis(crawl_id)
        
        if not documents:
            logging.error("No documents found for the given crawl ID.")
            return
        
        logging.info(f"Analyse de {len(documents)} documents...")
        
        analyzer = SemanticAnalyzer()
        contents = [doc["content"] for doc in documents]
        
        logging.info("Génération des embeddings avec CamemBERT...")
        embeddings = analyzer.compute_embeddings(contents)
        
        logging.info("Clustering des documents...")
        clusters = analyzer.cluster_documents(embeddings)
        
        logging.info("Extraction des thèmes...")
        topics = analyzer.extract_topics(contents, clusters)
        
        logging.info("Sauvegarde des résultats...")
        save_results_to_redis(crawl_id, documents, clusters, topics)
        save_graph_to_redis(crawl_id, documents, clusters, topics)
        
        # Message final avec statistiques et URL
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_outliers = sum(clusters == -1)
        ddev_url = get_ddev_url()
        
        logging.info(f"""
        ✅ Analyse terminée avec succès:
        - Nombre de clusters: {n_clusters}
        - Documents non classés: {n_outliers}
        - Thèmes principaux: {topics}
        
        🌐 Pour visualiser les résultats:
        1. Ouvrez votre navigateur
        2. Accédez au tableau de bord: {ddev_url}/dashboard.html
        3. Sélectionnez le crawl avec l'ID: {crawl_id}
        """)
        
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()