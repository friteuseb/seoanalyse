import redis
import trafilatura
from trafilatura.sitemaps import sitemap_search
from urllib.parse import urlparse
import uuid

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def crawl_site(url):
    downloaded = trafilatura.fetch_url(url)
    result = trafilatura.extract(downloaded, include_links=True)
    return result

def save_to_redis(url, content, crawl_id):
    doc_id = f"{crawl_id}:doc:{r.incr(f'{crawl_id}:doc_count')}"
    r.hmset(doc_id, {
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
    url = input("Entrez l'URL du site à crawler: ")
    parsed_url = urlparse(url)
    crawl_id = f"{parsed_url.netloc.replace('.', '_')}_{uuid.uuid4()}"
    crawl_and_store(url, crawl_id)
    print(f"Crawl terminé. ID du crawl: {crawl_id}")
