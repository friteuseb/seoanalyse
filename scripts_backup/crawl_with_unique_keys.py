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
import uuid
import trafilatura

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

def save_embedding(crawl_id, id, url, content, embedding, internal_links):
    r.hset(f'{crawl_id}:doc:{id}', mapping={
        'url': url,
        'content': content,
        'embedding': str(embedding),
        'internal_links': str(internal_links)
    })

def embedding_from_redis(redis_embedding):
    return [float(x) for x in redis_embedding.decode('utf-8').strip('[]').split(', ')]

def cluster_and_assign_themes(crawl_id):
    embeddings = []
    urls = []
    contents = []

    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        embedding = embedding_from_redis(doc_data[b'embedding'])
        embeddings.append(embedding)
        urls.append(doc_data[b'url'].decode('utf-8'))
        contents.append(doc_data[b'content'].decode('utf-8'))

    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_

    for idx, cluster in enumerate(clusters):
        r.hset(f'{crawl_id}:doc:{idx}', 'cluster', cluster)
    
    log_message("Thématiques calculées et associées aux documents.")

    for idx, cluster in enumerate(clusters):
        log_message(f"URL: {urls[idx]} - Thématique: {cluster}")

    visualize_clusters(embeddings, clusters)

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

def ask_question():
    query = entry_query.get()
    if not query:
        messagebox.showerror("Erreur", "Veuillez entrer une requête.")
        return

    crawl_id = crawl_id_entry.get()
    if not crawl_id:
        messagebox.showerror("Erreur", "Veuillez entrer l'identifiant du crawl.")
        return

    query_embedding = hf.embed_query(query)
    closest_id = None
    closest_distance = float('inf')

    for key in r.scan_iter(f"{crawl_id}:doc:*"):
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

def analyze_site():
    url = entry_url.get()

    if not url:
        messagebox.showerror("Erreur", "Veuillez entrer l'URL.")
        return

    crawl_id = str(uuid.uuid4())

    def crawl_and_analyze():
        log_message(f"Début du crawl sur {url} avec l'ID {crawl_id}")
        to_visit, known_links = focused_crawler(url, max_seen_urls=100, max_known_urls=1000)
        
        if not known_links:
            log_message("Erreur : Impossible d'extraire le contenu du site.")
            return

        for idx, link in enumerate(known_links):
            log_message(f"Crawling: {link}")
            content = extract(fetch_url(link))
            if content:
                internal_links = list(trafilatura.extract_links(fetch_url(link), include="internal", url_filter=url))
                embedding = hf.embed_query(content)
                save_embedding(crawl_id, idx, link, content, embedding, internal_links)
                log_message(f"Page sauvegardée avec embedding et liens internes: {link}")

        cluster_and_assign_themes(crawl_id)

    threading.Thread(target=crawl_and_analyze).start()

def log_message(message):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, f"{message}\n")
    log_text.yview(tk.END)
    log_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Analyse de Site Web")

tk.Label(root, text="URL du site à analyser:").grid(row=0, column=0, padx=10, pady=10)
entry_url = tk.Entry(root, width=50)
entry_url.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Requête:").grid(row=1, column=0, padx=10, pady=10)
entry_query = tk.Entry(root, width=50)
entry_query.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="ID du Crawl:").grid(row=2, column=0, padx=10, pady=10)
crawl_id_entry = tk.Entry(root, width=50)
crawl_id_entry.grid(row=2, column=1, padx=10, pady=10)

btn_analyze = tk.Button(root, text="Analyser le site", command=analyze_site)
btn_analyze.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

btn_ask = tk.Button(root, text="Poser une question", command=ask_question)
btn_ask.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

log_text = scrolledtext.ScrolledText(root, state=tk.DISABLED, width=80, height=20)
log_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
