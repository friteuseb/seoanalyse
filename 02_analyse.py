import redis
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from nltk.corpus import stopwords
import sys
import logging
import subprocess
nltk.download('stopwords', quiet=True)

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_redis_port():
    try:
        port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
        return int(port.strip())
    except Exception as e:
        print(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

# Connexion à Redis en utilisant le port dynamique
r = redis.Redis(host='localhost', port=get_redis_port(), db=0)


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Utilisation des stopwords français de NLTK
french_stop_words = list(set(stopwords.words('french')))

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

def compute_embeddings(contents):
    return model.encode(contents)

def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans

def reduce_dimensions(embeddings):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def determine_cluster_labels(contents, clusters, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    X = vectorizer.fit_transform(contents)
    terms = vectorizer.get_feature_names_out()
    
    labels = []
    for i in range(n_clusters):
        cluster_center = X[clusters == i].mean(axis=0).A.flatten()
        sorted_indices = cluster_center.argsort()[::-1]
        top_terms = [terms[idx] for idx in sorted_indices[:5]]
        labels.append(' '.join(top_terms))
    
    return labels

def save_results_to_redis(crawl_id, documents, clusters, labels):
    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        r.hset(doc_id, "cluster", int(clusters[i]))
        r.hset(doc_id, "label", labels[clusters[i]])

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

def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python3 02_analyse.py <crawl_id>")
        return
    
    crawl_id = sys.argv[1]
    documents = get_documents_from_redis(crawl_id)
    
    if not documents:
        logging.error("No documents found for the given crawl ID.")
        return
    
    logging.info(f"Number of documents: {len(documents)}")
    
    contents = [doc["content"] for doc in documents]
    
    logging.info("Computing embeddings...")
    embeddings = compute_embeddings(contents)
    logging.info(f"Shape of embeddings: {embeddings.shape}")
    
    n_clusters = min(5, len(documents) - 1)
    logging.info(f"Clustering embeddings with {n_clusters} clusters...")
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=n_clusters)
    
    logging.info("Reducing dimensions for visualization...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    logging.info("Determining cluster labels...")
    labels = determine_cluster_labels(contents, clusters, n_clusters=n_clusters)
    
    logging.info("Saving results to Redis...")
    save_results_to_redis(crawl_id, documents, clusters, labels)
    
    logging.info("Saving network graph to Redis and JSON files...")
    save_graph_to_redis(crawl_id, documents, clusters, labels)

    logging.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()