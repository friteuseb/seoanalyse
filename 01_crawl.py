import redis
import trafilatura
from trafilatura.sitemaps import sitemap_search
import uuid
import sys
import subprocess
import logging
import requests
from xml.etree import ElementTree as ET
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_redis_port():
    try:
        port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
        return int(port.strip())
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

# Connexion à Redis
r = redis.Redis(host='localhost', port=get_redis_port(), db=0)

# --- Fonctions pour le crawl web ---
def get_urls_from_sitemap(base_url):
    """Récupère les URLs depuis le sitemap."""
    sitemap_urls = sitemap_search(base_url)
    if sitemap_urls:
        return sitemap_urls

    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
    logging.info(f"Lecture du sitemap : {sitemap_url}")
    
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        urls = []
        for url in root.findall('.//ns:url/ns:loc', ns):
            urls.append(url.text)
            logging.info(f"URL trouvée : {url.text}")
        
        logging.info(f"📍 {len(urls)} URLs trouvées dans le sitemap")
        return urls
        
    except Exception as e:
        logging.warning(f"Erreur lors de la lecture du sitemap : {e}")
        return [base_url]

def crawl_url(url):
    """Crawl une URL et retourne son contenu."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        result = trafilatura.extract(downloaded)
        return result
    return None

# --- Fonctions pour le crawl local ---
def get_local_files(path):
    """Récupère tous les fichiers HTML/TXT d'un dossier."""
    all_files = []
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    
    for ext in ['*.html', '*.txt']:
        all_files.extend([str(f) for f in path.rglob(ext)])
    
    logging.info(f"📂 {len(all_files)} fichiers trouvés dans {path}")
    return all_files

def read_local_file(file_path):
    """Lit le contenu d'un fichier local."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de {file_path}: {e}")
        return None

# --- Fonctions communes ---
def save_to_redis(identifier, content, crawl_id):
    """Sauvegarde le contenu dans Redis."""
    doc_id = f"{crawl_id}:doc:{r.incr(f'{crawl_id}:doc_count')}"
    r.hset(doc_id, mapping={
        "url": identifier,
        "content": content,
    })
    logging.info(f"✅ Enregistré: {identifier}")

def crawl_web(url):
    """Fonction principale pour le crawl web."""
    # Nettoyer et normaliser l'URL pour créer un ID unique
    cleaned_url = url.split('//')[1].replace(':', '_').replace('.', '_').replace('/', '_')
    crawl_id = f"{cleaned_url}__{str(uuid.uuid4())}"
    
    urls = get_urls_from_sitemap(url)
    
    logging.info(f"🌐 Début du crawl web pour {url}")
    for url in urls:
        content = crawl_url(url)
        if content:
            save_to_redis(url, content, crawl_id)
            
    return crawl_id

def crawl_local(path):
    """Fonction principale pour le crawl local."""
    crawl_id = f"local_{str(uuid.uuid4())}"
    files = get_local_files(path)
    
    if not files:
        logging.error("❌ Aucun fichier trouvé")
        return None
        
    logging.info(f"📂 Début du crawl local pour {path}")
    for file_path in files:
        content = read_local_file(file_path)
        if content:
            relative_path = os.path.relpath(file_path)
            save_to_redis(relative_path, content, crawl_id)
            
    return crawl_id

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage: 
        Pour crawl web: python3 01_crawl.py https://example.com
        Pour crawl local: python3 01_crawl.py ./dossier_ou_fichier""")
        sys.exit(1)
    
    source = sys.argv[1]
    
    # Déterminer si c'est une URL ou un chemin local
    if source.startswith(('http://', 'https://')):
        crawl_id = crawl_web(source)
    else:
        crawl_id = crawl_local(source)
        
    if crawl_id:
        print(f"Crawl terminé. ID du crawl: {crawl_id}")