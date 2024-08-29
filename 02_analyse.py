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

nltk.download('stopwords', quiet=True)

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connexion à Redis
r = redis.Redis(host='localhost', port=32768, db=0)

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

def save_graph_to_redis(crawl_id, documents, clusters, labels):
    nodes = []
    links = []
    url_to_index = {}

    for i, doc in enumerate(documents):
        url_to_index[doc["url"]] = i
        nodes.append({
            "id": doc["url"],
            "label": doc["url"].split('/')[-1],
            "group": int(clusters[i]),
            "title": labels[clusters[i]],
            "internal_links_count": len(doc["internal_links_out"])
        })

    for i, doc in enumerate(documents):
        for link in doc["internal_links_out"]:
            if link in url_to_index:
                links.append({
                    "source": doc["url"],  # Utiliser l'URL au lieu de l'index
                    "target": link,        # Utiliser l'URL cible directement
                    "value": 1
                })

    graph_data = {"nodes": nodes, "links": links}
    
    # Sauvegarde dans Redis
    r.set(f"{crawl_id}_simple_graph", json.dumps(graph_data, default=convert_to_serializable))
    logging.info(f"Simple view saved to Redis with key {crawl_id}_simple_graph")

    r.set(f"{crawl_id}_clustered_graph", json.dumps(graph_data, default=convert_to_serializable))
    logging.info(f"Clustered view saved to Redis with key {crawl_id}_clustered_graph")

    # Suppression de la sauvegarde dans les fichiers JSON
    # Ces lignes sont supprimées :
    # with open(f"{crawl_id}_simple_graph.json", "w") as f:
    #     json.dump(simple_graph, f, default=convert_to_serializable)
    # with open(f"{crawl_id}_clustered_graph.json", "w") as f:
    #     json.dump(clustered_graph, f, default=convert_to_serializable)
    # logging.info(f"Graph data saved to JSON files")

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