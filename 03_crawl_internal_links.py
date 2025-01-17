import redis
import trafilatura
from bs4 import BeautifulSoup
import sys
from urllib.parse import urljoin, urlparse, urlunparse
import json
import subprocess
from datetime import datetime
import logging
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_redis_port():
    try:
        port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
        return int(port.strip())
    except Exception as e:
        print(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

# Connexion à Redis en utilisant le port dynamique
r = redis.Redis(host='localhost', port=get_redis_port(), db=0)

def list_crawls():
    crawls = {}
    for key in r.scan_iter("*:doc_count"):
        crawl_id = key.decode('utf-8').split(':')[0]
        crawls[crawl_id] = r.get(key).decode('utf-8')
    return crawls

def select_crawl():
    crawls = list_crawls()
    print("Available crawls:")
    for i, (crawl_id, count) in enumerate(crawls.items(), 1):
        print(f"{i}. {crawl_id} (Documents: {count})")
    selected = int(input("Select the crawl number to analyze: ")) - 1
    return list(crawls.keys())[selected]

def get_documents_from_redis(crawl_id):
    try:
        documents = {}
        pattern = f"{crawl_id}:doc:*"
        keys = r.scan_iter(pattern)
        if not keys:
            logging.warning(f"Aucun document trouvé avec le pattern {pattern}")
            return documents

        for key in keys:
            try:
                doc_data = r.hgetall(key)
                doc_id = key.decode('utf-8')
                documents[doc_id] = {
                    'url': doc_data[b'url'].decode('utf-8'),
                    'label': doc_data.get(b'label', b'').decode('utf-8'),
                    'cluster': int(doc_data.get(b'cluster', b'0').decode('utf-8'))
                }
            except Exception as e:
                logging.error(f"Erreur lors du traitement du document {key}: {e}")
                continue
        return documents
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des documents: {e}")
        return {}

def extract_internal_links(base_url, content, selector=None, exclude_patterns=None):
    """
    Extrait les liens internes en gérant les patterns d'exclusion
    """
    if not content:
        return []
        
    soup = BeautifulSoup(content, 'html.parser')
    
    # Si aucun sélecteur n'est fourni ou si le sélecteur ne trouve rien,
    # utiliser le body entier
    if not selector:
        content_area = [soup.find('body')] if soup.find('body') else []
        logging.info(f"✅ Analyse de la page entière pour {base_url}")
    else:
        content_area = soup.select(selector)
        if not content_area:
            content_area = [soup.find('body')] if soup.find('body') else []
            logging.warning(f"⚠️ Sélecteur {selector} non trouvé pour {base_url}, analyse de la page entière")
    
    if not content_area:
        logging.warning(f"❌ Aucun contenu trouvé pour {base_url}")
        return []

    links = set()
    base_domain = urlparse(base_url).netloc
    
    for area in content_area:
        for link in area.find_all('a', href=True):
            href = link['href']
            
            # Ignorer les liens non valides
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
                
            try:
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # Vérifier si c'est un lien interne
                if parsed_url.netloc == base_domain:
                    normalized_url = urlunparse((
                        parsed_url.scheme,
                        parsed_url.netloc,
                        parsed_url.path.rstrip('/'),
                        '',  # pas de paramètres
                        '',  # pas de query
                        ''   # pas de fragment
                    ))
                    
                    # Vérifier les patterns d'exclusion
                    if exclude_patterns and any(pattern.lower() in normalized_url.lower() for pattern in exclude_patterns):
                        logging.debug(f"⏭️ Lien exclu (pattern match): {normalized_url}")
                        continue
                    
                    links.add(normalized_url)
                
            except Exception as e:
                logging.warning(f"Erreur lors du traitement de l'URL {href}: {e}")

    return list(links)


def crawl_with_retry(url, exclude_patterns=None, max_retries=3, delay=1):
    """
    Télécharge une page avec gestion des redirections et patterns d'exclusion
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = trafilatura.fetch_url(url)
            
            # Si la réponse est une chaîne (contenu HTML)
            if isinstance(response, str):
                return response

            # Si la réponse est un objet avec une URL (redirection)
            final_url = url
            if hasattr(response, 'url'):
                final_url = response.url
                
            # Vérifier les patterns d'exclusion sur l'URL finale
            if exclude_patterns and any(pattern.lower() in final_url.lower() for pattern in exclude_patterns):
                logging.info(f"⏭️ URL après redirection exclue (pattern match): {final_url}")
                return None
                
            # Extraire le contenu HTML
            if response:
                content = trafilatura.extract(response)
                if content:
                    return content
            
            delay *= 2  # Augmenter le délai après chaque échec
            
        except Exception as e:
            logging.warning(f"Tentative {attempt + 1}/{max_retries} échouée pour {url}: {e}")
            delay *= 2
            time.sleep(delay)
    
    logging.error(f"Échec du téléchargement de {url} après {max_retries} tentatives")
    return None


def should_exclude_url(url, patterns):
    """Vérifie si l'URL contient un des patterns à exclure"""
    if not patterns:
        return False
    url_lower = url.lower()
    return any(pattern.lower() in url_lower for pattern in patterns)


def save_internal_links_to_redis(crawl_id, documents, selector, exclude_patterns=None):
    """Sauvegarde les liens internes dans Redis en respectant les patterns d'exclusion."""
    total = len(documents)
    successful = 0
    failed = 0
    excluded = 0
    
    logging.info(f"""
    🔍 Analyse des liens internes:
    • Documents à traiter: {total}
    • Sélecteur CSS: {selector}
    • Patterns exclus: {exclude_patterns if exclude_patterns else 'Aucun'}
    """)
    
    for i, (doc_id, doc_info) in enumerate(documents.items(), 1):
        url = doc_info['url']
        
        downloaded = crawl_with_retry(url)
        if not downloaded:
            failed += 1
            continue
            
        internal_links = extract_internal_links(url, downloaded, selector, exclude_patterns)
        
        if internal_links:
            doc_data = {
                "internal_links_out": json.dumps(internal_links),
                "crawl_date": datetime.now().isoformat(),
                "content_length": len(downloaded),
                "links_count": len(internal_links)
            }
            r.hset(doc_id, mapping=doc_data)
            successful += 1
            
            logging.info(f"✅ {len(internal_links)} liens trouvés pour {url}")
        else:
            r.hset(doc_id, mapping={
                "internal_links_out": "[]",
                "crawl_date": datetime.now().isoformat(),
                "content_length": len(downloaded),
                "links_count": 0
            })
            
        if i % 10 == 0 or i == total:
            progress = (i/total) * 100
            logging.info(f"""
            📊 Progression: {i}/{total} ({progress:.1f}%)
            ✅ Succès: {successful}
            ❌ Échecs: {failed}
            ⏭️ Exclus: {excluded}
            """)
    
    logging.info(f"""
    🏁 Crawl terminé:
    • Documents traités: {total}
    • Succès: {successful}
    • Échecs: {failed}
    • Exclus: {excluded}
    • Taux de réussite: {(successful/(total-excluded))*100:.1f}%
    """)



def assign_cluster_colors(documents):
    cluster_colors = {}
    color_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    for doc_id, doc_info in documents.items():
        cluster = doc_info['cluster']
        if cluster not in cluster_colors:
            cluster_colors[cluster] = color_list[len(cluster_colors) % len(color_list)]
        doc_info['color'] = cluster_colors[cluster]
    return documents

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 03_crawl_internal_links.py <crawl_id> <CSS Selector> [-e pattern1 pattern2 ...]")
        sys.exit(1)
    
    crawl_id = sys.argv[1]
    selector = sys.argv[2]
    exclude_patterns = []
    
    # Récupérer les patterns d'exclusion
    if '-e' in sys.argv:
        idx = sys.argv.index('-e')
        exclude_patterns = sys.argv[idx+1:]

    documents = get_documents_from_redis(crawl_id)
    if not documents:
        print("No documents found for the given crawl ID.")
        return

    documents = assign_cluster_colors(documents)
    save_internal_links_to_redis(crawl_id, documents, selector, exclude_patterns)
    print("Internal links crawling complete.")

if __name__ == "__main__":
    main()