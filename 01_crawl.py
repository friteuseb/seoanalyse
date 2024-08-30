import redis
import trafilatura
from trafilatura.sitemaps import sitemap_search
import uuid
import sys
import subprocess


def get_redis_port():
    try:
        port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
        return int(port.strip())
    except Exception as e:
        print(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

# Connexion à Redis en utilisant le port dynamique
r = redis.Redis(host='localhost', port=get_redis_port(), db=0)


def crawl_site(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        result = trafilatura.extract(downloaded)
        return result
    return None

def save_to_redis(url, content, crawl_id):
    doc_id = f"{crawl_id}:doc:{r.incr(f'{crawl_id}:doc_count')}"
    r.hset(doc_id, mapping={
        "url": url,
        "content": content,
    })
    print(f"Enregistré dans Redis: {doc_id}")

def process_site(url, crawl_id):
    content = crawl_site(url)
    if content:
        save_to_redis(url, content, crawl_id)
        return url, content
    return None, None

def crawl_and_store(url, crawl_id):
    sitemap_urls = sitemap_search(url)
    urls_to_crawl = [url] + sitemap_urls
    print(f"URLs extraites du sitemap: {urls_to_crawl}")

    for url in urls_to_crawl:
        process_site(url, crawl_id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 01_crawl.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    crawl_id = f"{url.split('//')[1].replace('.', '_').replace('/', '_')}_{str(uuid.uuid4())}"
    crawl_and_store(url, crawl_id)
    print(f"Crawl terminé. ID du crawl: {crawl_id}")
