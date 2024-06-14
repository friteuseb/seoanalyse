import redis
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import json

nltk.download('stopwords')

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Utiliser un modèle de transformation en français
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

french_stop_words = list(set(stopwords.words('french')))

def get_documents_from_redis(crawl_id):
    urls = []
    contents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        url = doc_data[b'url'].decode('utf-8')
        content = doc_data[b'content'].decode('utf-8')
        urls.append(url)
        contents.append(content)
    return urls, contents

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
        cluster_center = np.asarray(X[clusters == i].mean(axis=0)).flatten()
        sorted_indices = cluster_center.argsort()[::-1]
        top_terms = [terms[idx] for idx in sorted_indices[:5]]
        labels.append(' '.join(top_terms))
    
    return labels

def save_results_to_json(crawl_id, urls, clusters, labels, file_name='network.json'):
    nodes = []
    for i, url in enumerate(urls):
        node = {
            "id": url,
            "label": url.split('/')[-1],
            "cluster": int(clusters[i]),
            "title": labels[clusters[i]]
        }
        nodes.append(node)
    
    graph = {"nodes": nodes, "links": []}
    
    with open(file_name, 'w') as f:
        json.dump(graph, f)
    print(f"Network graph saved to {file_name}")

def list_crawls():
    keys = r.keys('*:doc_count')
    crawl_ids = [key.decode('utf-8').split(':')[0] for key in keys]
    return crawl_ids

def main():
    crawl_ids = list_crawls()
    if not crawl_ids:
        print("No crawls found.")
        return
    
    print("Available crawls:")
    for i, cid in enumerate(crawl_ids, 1):
        print(f"{i}. {cid}")

    crawl_idx = int(input("Select the crawl number to analyze: ")) - 1
    crawl_id = crawl_ids[crawl_idx]

    urls, contents = get_documents_from_redis(crawl_id)
    
    if not contents:
        print("No content found for the given crawl ID.")
        return
    
    print("Computing embeddings...")
    embeddings = compute_embeddings(contents)
    
    print("Clustering embeddings...")
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=5)
    
    print("Reducing dimensions for visualization...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("Determining cluster labels...")
    labels = determine_cluster_labels(contents, clusters, n_clusters=5)
    
    print("Saving results to JSON...")
    save_results_to_json(crawl_id, urls, clusters, labels)
    
    print("Analysis complete. You can now visualize the results using the HTML file.")

if __name__ == "__main__":
    main()
