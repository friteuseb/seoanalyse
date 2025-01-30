import redis
import trafilatura
from trafilatura.sitemaps import sitemap_search
import uuid
import sys
import subprocess
import logging
import requests
import os
from pathlib import Path
from typing import List, Optional, Set
from sitemap_handler import SitemapHandler


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
def save_to_redis(identifier, content, crawl_id, exclude_patterns=None):
    """Sauvegarde le contenu dans Redis en respectant les patterns d'exclusion"""
    try:
        # Vérification des patterns d'exclusion
        if exclude_patterns:
            for pattern in exclude_patterns:
                if pattern.lower() in identifier.lower():
                    logging.info(f"❌ Page ignorée (pattern '{pattern}'): {identifier}")
                    return False

        doc_id = f"{crawl_id}:doc:{r.incr(f'{crawl_id}:doc_count')}"
        r.hset(doc_id, mapping={
            "url": identifier,
            "content": content,
        })
        logging.info(f"✅ Page stockée: {identifier}")
        return True
            
    except Exception as e:
        logging.error(f"Erreur lors du stockage de {identifier}: {e}")
        return False
    
def filter_sitemap_urls(urls, exclude_patterns):
    """Filtre les URLs du sitemap selon les patterns d'exclusion"""
    original_count = len(urls)
    filtered_urls = []
    excluded_urls = []
    
    for url in urls:
        should_exclude = any(pattern.lower() in url.lower() for pattern in exclude_patterns)
        if should_exclude:
            excluded_urls.append(url)
        else:
            filtered_urls.append(url)
    
    logging.info(f"""
    🔍 Filtrage du sitemap:
    • URLs totales: {original_count}
    • URLs retenues: {len(filtered_urls)}
    • URLs exclues: {len(excluded_urls)}
    • Patterns utilisés: {exclude_patterns}
    """)
    
    if excluded_urls:
        logging.debug("Examples d'URLs exclues:")
        for url in excluded_urls[:5]:
            logging.debug(f"⏭️  {url}")
    
    return filtered_urls



def get_urls_from_sitemap(url: str, scope_url: Optional[str] = None, exclude_patterns=None) -> List[str]:
    """Récupère les URLs depuis le sitemap"""
    try:
        base_url = "/".join(url.split("/")[:3])  # Extrait le domaine
        handler = SitemapHandler(base_url, scope_url=scope_url)
        urls = handler.discover_urls()

        if exclude_patterns:
            return filter_sitemap_urls(urls, exclude_patterns)
        return urls

    except Exception as e:
        logging.error(f"Erreur lors du crawl du sitemap: {e}")
        return [url]
    

def should_exclude_url(url, patterns):
    """Vérifie si l'URL contient un des patterns à exclure"""
    if not patterns:
        return False
    url_lower = url.lower()
    return any(pattern.lower() in url_lower for pattern in patterns)


def crawl_web(url, exclude_patterns=None):
    """Fonction principale pour le crawl web"""
    cleaned_url = url.split('//')[1].replace(':', '_').replace('.', '_').replace('/', '_')
    crawl_id = f"{cleaned_url}__{str(uuid.uuid4())}"
    
    # Utiliser l'URL complète comme scope si c'est une URL de page
    urls = get_urls_from_sitemap(url, scope_url=url, exclude_patterns=exclude_patterns)
    
    # Ajout du traitement des URLs
    if not urls:
        logging.error("❌ Aucune URL à crawler")
        return None
        
    processed = 0
    for url in urls:
        try:
            content = crawl_url(url)
            if content:
                if save_to_redis(url, content, crawl_id, exclude_patterns):
                    processed += 1
                    
        except Exception as e:
            logging.error(f"Erreur lors du crawl de {url}: {e}")
            continue
            
        # Log de progression
        if processed % 10 == 0 or processed == len(urls):
            logging.info(f"""
            📊 Progression:
            • Pages traitées : {processed}/{len(urls)}
            • Pourcentage : {(processed/len(urls))*100:.1f}%
            """)
    
    if processed > 0:
        return crawl_id
    return None

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
    if len(sys.argv) < 2:
        print("""Usage: 
        Pour crawl web: python3 01_crawl.py https://example.com [exclude_patterns...]
        Pour crawl local: python3 01_crawl.py ./dossier_ou_fichier""")
        sys.exit(1)
    
    source = sys.argv[1]
    exclude_patterns = sys.argv[2:] if len(sys.argv) > 2 else None
    
    # Déterminer si c'est une URL ou un chemin local
    if source.startswith(('http://', 'https://')):
        crawl_id = crawl_web(source, exclude_patterns)
    else:
        crawl_id = crawl_local(source)
        
    if crawl_id:
        print(f"Crawl terminé. ID du crawl: {crawl_id}")