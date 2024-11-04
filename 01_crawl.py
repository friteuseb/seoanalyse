import redis
import trafilatura
from trafilatura.sitemaps import sitemap_search
import uuid
import sys
import subprocess
import logging
import requests
from xml.etree import ElementTree as ET

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

def get_urls_from_sitemap(base_url):
    """Récupère les URLs depuis le sitemap."""
    # D'abord essayer trafilatura
    sitemap_urls = sitemap_search(base_url)
    if sitemap_urls:
        return sitemap_urls

    # Si trafilatura échoue, essayer la méthode directe
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

def crawl_site(url):
    """Crawl une URL et retourne son contenu."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        result = trafilatura.extract(downloaded)
        return result
    return None

def save_to_redis(url, content, crawl_id):
    """Sauvegarde le contenu dans Redis."""
    doc_id = f"{crawl_id}:doc:{r.incr(f'{crawl_id}:doc_count')}"
    r.hset(doc_id, mapping={
        "url": url,
        "content": content,
    })
    print(f"Enregistré dans Redis: {doc_id}")

def process_site(url, crawl_id):
    """Traite une URL individuelle."""
    content = crawl_site(url)
    if content:
        save_to_redis(url, content, crawl_id)
        return url, content
    return None, None

def crawl_and_store(url, crawl_id):
    """Fonction principale de crawl."""
    urls_to_crawl = get_urls_from_sitemap(url)
    print(f"URLs extraites du sitemap: {urls_to_crawl}")

    for url in urls_to_crawl:
        logging.info(f"Traitement de : {url}")
        process_site(url, crawl_id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 01_crawl.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    crawl_id = f"{url.split('//')[1].replace('.', '_').replace('/', '_')}_{str(uuid.uuid4())}"
    crawl_and_store(url, crawl_id)
    print(f"Crawl terminé. ID du crawl: {crawl_id}")