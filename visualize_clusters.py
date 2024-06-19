import redis
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import umap
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse

# Connexion à Redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# Récupération des données depuis Redis
keys = r.keys('www_reseau-travaux_fr__*')
data = []
for key in keys:
    if r.type(key) == b'hash':
        doc_data = r.hgetall(key)
        url = doc_data.get(b'url').decode('utf-8')
        labels = json.loads(doc_data.get(b'labels').decode('utf-8'))
        data.append((url, labels))

# Préparation des données pour t-SNE et UMAP
urls = [item[0] for item in data]
labels_data = [item[1] for item in data]
all_labels = list(set(label for sublist in labels_data for label in sublist.keys()))

# Création de la matrice de caractéristiques
features = np.zeros((len(data), len(all_labels)))
for i, labels in enumerate(labels_data):
    for label, score in labels.items():
        features[i, all_labels.index(label)] = score

# Réduction de dimensionnalité avec t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(features)

# Réduction de dimensionnalité avec UMAP
umap_results = umap.UMAP(n_components=2).fit_transform(features)

# Création d'une matrice de similarité pour la heatmap
similarity_matrix = cosine_similarity(features)

# Fonction utilitaire pour obtenir le dernier segment d'une URL
def get_last_segment(url):
    path = urlparse(url).path
    return path.strip('/').split('/')[-1]

# Visualisation des Graphes de Réseaux
def visualize_network_graph():
    G = nx.DiGraph()
    for i, url in enumerate(urls):
        G.add_node(url, label=get_last_segment(url))
        for j, label in enumerate(all_labels):
            if features[i, j] > 0:
                G.add_edge(url, label, weight=features[i, j])

    pos = nx.spring_layout(G, k=0.15, iterations=20)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)
    plt.title('Graphe de Réseau Dirigé')
    plt.show()

# Visualisation des Graphes de Clusters
def visualize_clusters_graph():
    G = nx.Graph()
    for i, url in enumerate(urls):
        G.add_node(url, label=get_last_segment(url))
        for j, label in enumerate(all_labels):
            if features[i, j] > 0:
                G.add_edge(url, label, weight=features[i, j])

    pos = nx.spring_layout(G, k=0.15, iterations=20)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)
    plt.title('Graphe de Clusters')
    plt.show()

# Visualisation de la Heatmap de Similarité
def visualize_heatmap():
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=urls, yticklabels=urls, cmap='viridis')
    plt.title('Heatmap de Similarité')
    plt.show()

# Visualisation t-SNE
def visualize_tsne():
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    for i, url in enumerate(urls):
        x, y = tsne_results[i]
        plt.text(x, y, get_last_segment(url), fontsize=9)
    plt.title('Visualisation t-SNE des URLs')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# Visualisation UMAP
def visualize_umap():
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1])
    for i, url in enumerate(urls):
        x, y = umap_results[i]
        plt.text(x, y, get_last_segment(url), fontsize=9)
    plt.title('Visualisation UMAP des URLs')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# Exécution des visualisations
visualize_network_graph()
visualize_clusters_graph()
visualize_heatmap()
visualize_tsne()
visualize_umap()
