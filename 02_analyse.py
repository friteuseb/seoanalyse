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

nltk.download('stopwords')

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

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
        internal_links_out = doc_data.get(b'internal_links_out', b'').decode('utf-8').split(',')
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
    url_to_id = {doc["url"]: doc["doc_id"] for doc in documents}
    link_counts = {}

    for i, doc in enumerate(documents):
        internal_links_count = len(doc["internal_links_out"])
        nodes.append({
            "id": doc["url"],
            "label": doc["url"].split('/')[-1],
            "group": int(clusters[i]),
            "title": labels[clusters[i]],
            "internal_links_count": internal_links_count
        })
        for link in doc["internal_links_out"]:
            if link in url_to_id:
                link_key = (doc["url"], link)
                link_counts[link_key] = link_counts.get(link_key, 0) + 1

    for (source, target), weight in link_counts.items():
        links.append({
            "source": source,
            "target": target,
            "color": int(clusters[i]),
            "weight": weight
        })

    simple_graph = {"nodes": nodes, "links": links}
    r.set(f"{crawl_id}_simple_graph", json.dumps(simple_graph, default=convert_to_serializable))
    print(f"Simple view saved to Redis with key {crawl_id}_simple_graph")

    clustered_graph = {"nodes": nodes, "links": links}
    r.set(f"{crawl_id}_clustered_graph", json.dumps(clustered_graph, default=convert_to_serializable))
    print(f"Clustered view saved to Redis with key {crawl_id}_clustered_graph")

def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 02_analyse.py <crawl_id>")
        return
    
    crawl_id = sys.argv[1]
    documents = get_documents_from_redis(crawl_id)
    
    if not documents:
        print("No documents found for the given crawl ID.")
        return
    
    contents = [doc["content"] for doc in documents]
    
    print("Computing embeddings...")
    embeddings = compute_embeddings(contents)
    
    print("Clustering embeddings...")
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=5)
    
    print("Reducing dimensions for visualization...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("Determining cluster labels...")
    labels = determine_cluster_labels(contents, clusters, n_clusters=5)
    
    print("Saving results to Redis...")
    save_results_to_redis(crawl_id, documents, clusters, labels)
    
    print("Saving network graph to Redis...")
    save_graph_to_redis(crawl_id, documents, clusters, labels)

if __name__ == "__main__":
    main()
