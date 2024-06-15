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
import os
import time
import sys

nltk.download('stopwords')

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Utilisation des stopwords français de NLTK
french_stop_words = list(set(stopwords.words('french')))

def list_crawls():
    crawls = {}
    for key in r.scan_iter("*:doc_count"):
        crawl_id = key.decode('utf-8').split(':')[0]
        crawls[crawl_id] = r.get(key).decode('utf-8')
    return crawls

def select_crawl():
    crawls = list_crawls()
    print("Available crawls:")
    for i, (crawl_id, count) in enumerate(crawls.items(), 1):
        print(f"{i}. {crawl_id} (Documents: {count})")
    selected = int(input("Select the crawl number to analyze: ")) - 1
    return list(crawls.keys())[selected]

def get_documents_from_redis(crawl_id):
    documents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        doc_id = key.decode('utf-8')
        url = doc_data[b'url'].decode('utf-8')
        content = doc_data[b'content'].decode('utf-8')
        internal_links = doc_data.get(b'internal_links', b'').decode('utf-8').split(',')
        documents.append({
            "doc_id": doc_id,
            "url": url,
            "content": content,
            "internal_links": internal_links
        })
    return documents

def compute_embeddings(contents):
    print("Starting embeddings computation...")
    start_time = time.time()
    embeddings = model.encode(contents)
    print(f"Embeddings computed in {time.time() - start_time:.2f} seconds.")
    return embeddings

def cluster_embeddings(embeddings, n_clusters=5):
    print("Starting clustering...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")
    return clusters, kmeans

def reduce_dimensions(embeddings):
    print("Starting dimensionality reduction...")
    start_time = time.time()
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"Dimensionality reduction completed in {time.time() - start_time:.2f} seconds.")
    return reduced_embeddings

def determine_cluster_labels(contents, clusters, n_clusters=5):
    print("Starting cluster label determination...")
    start_time = time.time()
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    X = vectorizer.fit_transform(contents)
    terms = vectorizer.get_feature_names_out()
    
    labels = []
    for i in range(n_clusters):
        cluster_center = X[clusters == i].mean(axis=0).A.flatten()
        sorted_indices = cluster_center.argsort()[::-1]
        top_terms = [terms[idx] for idx in sorted_indices[:5]]
        labels.append(' '.join(top_terms))
    
    print(f"Cluster labels determined in {time.time() - start_time:.2f} seconds.")
    return labels

def save_results_to_redis(crawl_id, documents, clusters, labels):
    print("Starting to save results to Redis...")
    start_time = time.time()
    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        r.hset(doc_id, "cluster", int(clusters[i]))
        r.hset(doc_id, "label", labels[clusters[i]])
    print(f"Results saved to Redis in {time.time() - start_time:.2f} seconds.")

def save_network_to_json(documents, clusters, labels, filename="visualizations/network.json"):
    print("Starting to save network graph to JSON...")
    start_time = time.time()
    nodes = []
    links = []
    url_to_id = {doc["url"]: doc["doc_id"] for doc in documents}

    for i, doc in enumerate(documents):
        nodes.append({
            "id": doc["url"],
            "label": doc["url"].split('/')[-1],
            "group": int(clusters[i]),
            "title": labels[clusters[i]]
        })
        for link in doc["internal_links"]:
            if link in url_to_id:
                links.append({
                    "source": doc["url"],
                    "target": link
                })

    graph = {"nodes": nodes, "links": links}
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(graph, f)
    print(f"Network graph saved to {filename} in {time.time() - start_time:.2f} seconds.")

def main(crawl_id=None):
    if crawl_id is None:
        crawl_id = select_crawl()
    documents = get_documents_from_redis(crawl_id)
    
    if not documents:
        print("No documents found for the given crawl ID.")
        return
    
    contents = [doc["content"] for doc in documents]
    
    embeddings = compute_embeddings(contents)
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=5)
    reduced_embeddings = reduce_dimensions(embeddings)
    labels = determine_cluster_labels(contents, clusters, n_clusters=5)
    
    save_results_to_redis(crawl_id, documents, clusters, labels)
    save_network_to_json(documents, clusters, labels)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        crawl_id = sys.argv[1]
        main(crawl_id)
    else:
        main()
