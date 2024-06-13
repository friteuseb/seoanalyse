import redis
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from trafilatura.spider import focused_crawler
from trafilatura import fetch_url, extract
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Configuration du modèle d'embedding
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Fonction pour sauvegarder les embeddings dans Redis
def save_embedding(id, url, content, embedding):
    r.hset(f'doc:{id}', mapping={'url': url, 'content': content, 'embedding': str(embedding)})

# Fonction pour convertir un embedding de Redis en liste
def embedding_from_redis(redis_embedding):
    return [float(x) for x in redis_embedding.decode('utf-8').strip('[]').split(', ')]

# Fonction de clustering et d'association des thématiques
def cluster_and_assign_themes():
    embeddings = []
    urls = []
    contents = []

    for key in r.scan_iter("doc:*"):
        doc_data = r.hgetall(key)
        embedding = embedding_from_redis(doc_data[b'embedding'])
        embeddings.append(embedding)
        urls.append(doc_data[b'url'].decode('utf-8'))
        contents.append(doc_data[b'content'].decode('utf-8'))

    # Convertir les embeddings en array numpy
    embeddings = np.array(embeddings)

    # Appliquer K-means pour le clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(embeddings)

    # Associer chaque document à un cluster
    clusters = kmeans.labels_

    # Sauvegarder les thématiques dans Redis
    for idx, cluster in enumerate(clusters):
        r.hset(f'doc:{idx}', 'cluster', cluster)
    
    log_message("Thématiques calculées et associées aux documents.")

    # Afficher les résultats dans la zone de texte
    for idx, cluster in enumerate(clusters):
        log_message(f"URL: {urls[idx]} - Thématique: {cluster}")

    # Visualiser les clusters
    visualize_clusters(embeddings, clusters)

# Fonction pour visualiser les clusters
def visualize_clusters(embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
    plt.title("Visualisation des clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=set(clusters))
    plt.show()

# Fonction pour poser une nouvelle question et interroger les embeddings
def ask_question():
    query = entry_query.get()
    if not query:
        messagebox.showerror("Erreur", "Veuillez entrer une requête.")
        return

    query_embedding = hf.embed_query(query)
    closest_id = None
    closest_distance = float('inf')

    for key in r.scan_iter("doc:*"):
        doc_data = r.hgetall(key)
        doc_embedding = embedding_from_redis(doc_data[b'embedding'])
        distance = sum((qe - de) ** 2 for qe, de in zip(query_embedding, doc_embedding)) ** 0.5
        
        if distance < closest_distance:
            closest_distance = distance
            closest_id = key

    if closest_id:
        result_doc = r.hget(closest_id, 'content').decode('utf-8')
        log_message(f"Le document le plus pertinent pour la requête '{query}' est :\n\n{result_doc}")
        messagebox.showinfo("Résultat", f"Le document le plus pertinent pour la requête '{query}' est :\n\n{result_doc}")

# Fonction de crawling et d'analyse
def analyze_site():
    url = entry_url.get()

    if not url:
        messagebox.showerror("Erreur", "Veuillez entrer l'URL.")
        return

    def crawl_and_analyze():
        # Crawler tout le site
        log_message(f"Début du crawl sur {url}")
        to_visit, known_links = focused_crawler(url, max_seen_urls=100, max_known_urls=1000)
        
        if not known_links:
            log_message("Erreur : Impossible d'extraire le contenu du site.")
            return

        # Télécharger, extraire le contenu et générer les embeddings pour chaque page
        for idx, link in enumerate(known_links):
            log_message(f"Crawling: {link}")
            content = extract(fetch_url(link))
            if content:
                embedding = hf.embed_query(content)
                save_embedding(idx, link, content, embedding)
                log_message(f"Page sauvegardée avec embedding: {link}")

        # Calculer les thématiques dominantes et les associer aux documents
        cluster_and_assign_themes()

    # Lancer le crawl dans un thread séparé
    threading.Thread(target=crawl_and_analyze).start()

# Fonction pour ajouter des messages de log à la zone de texte
def log_message(message):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, f"{message}\n")
    log_text.yview(tk.END)
    log_text.config(state=tk.DISABLED)

# Créer l'interface graphique
root = tk.Tk()
root.title("Analyse de Site Web")

tk.Label(root, text="URL du site à analyser:").grid(row=0, column=0, padx=10, pady=10)
entry_url = tk.Entry(root, width=50)
entry_url.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Requête:").grid(row=1, column=0, padx=10, pady=10)
entry_query = tk.Entry(root, width=50)
entry_query.grid(row=1, column=1, padx=10, pady=10)

btn_analyze = tk.Button(root, text="Analyser le site", command=analyze_site)
btn_analyze.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

btn_ask = tk.Button(root, text="Poser une question", command=ask_question)
btn_ask.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

log_text = scrolledtext.ScrolledText(root, state=tk.DISABLED, width=80, height=20)
log_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()

