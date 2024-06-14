import os
import redis
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from nltk.corpus import stopwords
from threading import Thread
import http.server
import socketserver

# Download NLTK stopwords
nltk.download('stopwords')

# Define the cache directory for the Hugging Face models
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers/")

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

def compute_embeddings(contents):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", cache_folder=CACHE_DIR)
    embeddings = model.encode(contents)
    return embeddings

def determine_cluster_labels(contents, clusters, n_clusters):
    stop_words = stopwords.words('french')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(contents)
    cluster_labels = []

    for i in range(n_clusters):
        cluster_center = X[clusters == i].mean(axis=0)
        sorted_indices = np.argsort(np.array(cluster_center)).flatten()[::-1]
        feature_names = np.array(vectorizer.get_feature_names_out())
        label = feature_names[sorted_indices[:3]]
        cluster_labels.append(", ".join(label))
    
    return cluster_labels

def run_http_server():
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

def create_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Force Graph</title>
        <style>
            body { margin: 0; }
            canvas { display: block; }
        </style>
    </head>
    <body>
        <script src="https://unpkg.com/three@0.124.0/build/three.min.js"></script>
        <script src="https://unpkg.com/3d-force-graph"></script>
        <script src="https://unpkg.com/three-spritetext"></script>
        <script>
            fetch('network.json')
                .then(response => response.json())
                .then(data => {
                    const Graph = ForceGraph3D()
                        (document.body)
                        .graphData(data)
                        .nodeLabel('label')
                        .nodeAutoColorBy('cluster')
                        .nodeThreeObject(node => {
                            const sprite = new SpriteText(node.label);
                            sprite.color = node.color;
                            sprite.textHeight = 8;
                            return sprite;
                        })
                        .onNodeClick(node => {
                            window.open(node.id, "_blank");
                        })
                        .nodeRelSize(4)
                        .linkDirectionalParticles(2)
                        .linkDirectionalParticleWidth(2)
                        .onNodeHover(node => {
                            document.body.style.cursor = node ? 'pointer' : null;
                            if (node) {
                                const tooltip = document.createElement('div');
                                tooltip.style.position = 'fixed';
                                tooltip.style.top = `${event.clientY + 5}px`;
                                tooltip.style.left = `${event.clientX + 5}px`;
                                tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                                tooltip.style.color = 'white';
                                tooltip.style.padding = '5px';
                                tooltip.style.borderRadius = '5px';
                                tooltip.style.zIndex = '1000';
                                tooltip.textContent = `Thème: ${node.label}, Mots: ${node.words}`;
                                tooltip.id = 'tooltip';
                                document.body.appendChild(tooltip);
                            } else {
                                const tooltip = document.getElementById('tooltip');
                                if (tooltip) {
                                    tooltip.remove();
                                }
                            }
                        });
                });
        </script>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html_content)

def main():
    # Get list of crawls
    crawl_keys = [key.decode('utf-8') for key in r.scan_iter("*:doc_count")]
    crawl_ids = [key.split(":")[0] for key in crawl_keys]

    # Display available crawls
    print("Available crawls:")
    for i, crawl_id in enumerate(crawl_ids):
        print(f"{i + 1}. {crawl_id}")

    # Select a crawl to analyze
    selected_crawl = int(input("Select the crawl number to analyze: ")) - 1
    crawl_id = crawl_ids[selected_crawl]

    # Get URLs and contents from Redis
    urls = []
    contents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        urls.append(doc_data[b'url'].decode('utf-8'))
        contents.append(doc_data[b'content'].decode('utf-8'))

    print("Computing embeddings...")
    embeddings = compute_embeddings(contents)

    print("Clustering embeddings...")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)

    print("Reducing dimensions for visualization...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    print("Determining cluster labels...")
    labels = determine_cluster_labels(contents, clusters, n_clusters)

    print("Building network graph...")
    nodes = []
    links = []
    for i, url in enumerate(urls):
        node_id = url
        label = url.split('/')[-1]
        cluster = int(clusters[i])
        words = len(contents[i].split())
        nodes.append({"id": node_id, "label": label, "cluster": cluster, "words": words})

    for i, url in enumerate(urls):
        doc_key = f"{crawl_id}:doc:{i+1}"
        internal_links = r.hget(doc_key, "internal_links")
        if internal_links:
            internal_links = internal_links.decode('utf-8').split(',')
            for link in internal_links:
                if link in urls:
                    links.append({"source": url, "target": link})

    graph = {"nodes": nodes, "links": links}
    with open("network.json", "w") as f:
        json.dump(graph, f)

    print("Saving network graph to JSON...")
    print("Network graph saved to network.json")

    print("Creating HTML file for visualization...")
    create_html()

    print("Starting HTTP server for visualization...")
    server_thread = Thread(target=run_http_server)
    server_thread.start()

if __name__ == "__main__":
    main()
