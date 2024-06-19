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
import os

nltk.download('stopwords')

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

def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_network_to_json(documents, clusters, labels, filename="visualizations/network.json"):
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
        for link in doc["internal_links_out"]:
            if link in url_to_id:
                links.append({
                    "source": doc["url"],
                    "target": link
                })

    graph = {"nodes": nodes, "links": links}
    with open(filename, "w") as f:
        json.dump(graph, f, default=convert_to_serializable)
    print(f"Network graph saved to {filename}")

def list_json_files(directory="visualizations/crawls"):
    json_files = [f for f in os.listdir(directory) if f.endswith('_simple_view.json') or f.endswith('_clustered_view.json')]
    with open(os.path.join(directory, 'json_files.json'), 'w') as f:
        json.dump(json_files, f)
    print("JSON files list saved to visualizations/crawls/json_files.json")

def save_simple_and_clustered_views_to_json(documents, clusters, labels, crawl_id):
    simple_view_file = f"visualizations/crawls/{crawl_id}_simple_view.json"
    clustered_view_file = f"visualizations/crawls/{crawl_id}_clustered_view.json"
    
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
        for link in doc["internal_links_out"]:
            if link in url_to_id:
                links.append({
                    "source": doc["url"],
                    "target": link,
                    "color": int(clusters[i])
                })

    simple_graph = {"nodes": nodes, "links": links}
    with open(simple_view_file, "w") as f:
        json.dump(simple_graph, f, default=convert_to_serializable)
    print(f"Simple view saved to {simple_view_file}")

    clustered_graph = {"nodes": nodes, "links": links}
    with open(clustered_view_file, "w") as f:
        json.dump(clustered_graph, f, default=convert_to_serializable)
    print(f"Clustered view saved to {clustered_view_file}")

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
    
    print("Saving network graph to JSON...")
    save_network_to_json(documents, clusters, labels, filename=f"visualizations/crawls/{crawl_id}_network.json")

    print(f"Saving simple and clustered views to JSON...")
    save_simple_and_clustered_views_to_json(documents, clusters, labels, crawl_id)

    list_json_files()

if __name__ == "__main__":
    main()
