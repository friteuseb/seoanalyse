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

def extract_internal_links(base_url, content, selector=None):
    """
    Extrait uniquement les liens internes présents dans la zone de contenu spécifiée 
    ou dans toute la page si aucun sélecteur n'est fourni.
    
    Args:
        base_url (str): L'URL de base du site
        content (str): Le contenu HTML de la page
        selector (str, optional): Le sélecteur CSS pour cibler la zone de contenu principale.
            Si None, analyse toute la page.
    
    Returns:
        list: Liste des URLs internes trouvées
    """
    soup = BeautifulSoup(content, 'html.parser')
    
    # Si aucun sélecteur n'est fourni, utiliser le body entier
    if not selector:
        content_area = [soup.find('body')] if soup.find('body') else []
        logging.info(f"✅ Analyse de la page entière pour {base_url}")
    else:
        # Trouver la zone de contenu spécifiée
        content_area = soup.select(selector)
        if not content_area:
            logging.warning(f"❌ Aucune zone trouvée avec le sélecteur {selector} pour {base_url}")
            return []
        logging.info(f"✅ Zone de contenu trouvée pour {base_url}")
    
    links = set()
    base_domain = urlparse(base_url).netloc
    
    for area in content_area:
        # Debug pour voir le contenu de la zone
        logging.debug(f"Analyse de la zone : {area.get('class', 'no-class')} - {area.get('id', 'no-id')}")
        
        for link in area.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text().strip()
            
            # Ignorer les ancres seules et les liens javascript
            if href.startswith('#') or href.startswith('javascript:'):
                continue
                
            try:
                # Normalisation de l'URL
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # Vérification des liens internes
                if parsed_url.netloc == base_domain:
                    normalized_url = urlunparse((
                        parsed_url.scheme,
                        parsed_url.netloc,
                        parsed_url.path.rstrip('/'),
                        parsed_url.params,
                        parsed_url.query,
                        None  # Pas de fragment
                    ))
                    links.add(normalized_url)
                    logging.debug(f"Lien trouvé : {link_text} -> {normalized_url}")
                
            except Exception as e:
                logging.warning(f"Erreur lors du traitement de l'URL {href}: {e}")

    zone_type = "la page entière" if not selector else "la zone de contenu"
    logging.info(f"🔍 Liens trouvés dans {zone_type} : {len(links)}")
    return list(links)

def crawl_with_retry(url, max_retries=3, delay=1):
    """
    Télécharge une page avec gestion des erreurs et retry.
    """
    for attempt in range(max_retries):
        try:
            # Attente entre les requêtes pour éviter de surcharger le serveur
            time.sleep(delay)
            
            # Utilisation simple de fetch_url sans paramètres additionnels
            response = trafilatura.fetch_url(url)
            if response:
                return response
            
            # Augmenter le délai après chaque échec
            delay *= 2
            
        except Exception as e:
            logging.warning(f"Tentative {attempt + 1}/{max_retries} échouée pour {url}: {e}")
            time.sleep(delay)
    
    logging.error(f"Échec du téléchargement de {url} après {max_retries} tentatives")
    return None

def save_internal_links_to_redis(crawl_id, documents, selector):
    total = len(documents)
    successful = 0
    failed = 0
    
    logging.info(f"Démarrage du crawl des liens internes pour {total} documents")
    
    for i, (doc_id, doc_info) in enumerate(documents.items(), 1):
        url = doc_info['url']
        logging.info(f"Processing {i}/{total}: {url}")
        
        downloaded = crawl_with_retry(url)
        
        if not downloaded:
            failed += 1
            logging.error(f"❌ Échec du téléchargement pour {url}")
            continue
            
        internal_links_out = extract_internal_links(url, downloaded, selector)
        successful += 1
        
        if internal_links_out:
            doc_data = {
                "internal_links_out": json.dumps(internal_links_out),
                "crawl_date": datetime.now().isoformat(),
                "content_length": len(downloaded),
                "links_count": len(internal_links_out)
            }
            logging.info(f"✅ {len(internal_links_out)} liens trouvés pour {url}")
            r.hset(doc_id, mapping=doc_data)
        else:
            logging.info(f"ℹ️ Aucun lien trouvé pour {url}")
            r.hset(doc_id, mapping={
                "internal_links_out": "[]",
                "crawl_date": datetime.now().isoformat(),
                "content_length": len(downloaded),
                "links_count": 0
            })
            
        if i % 5 == 0 or i == total:
            percent = (i/total)*100
            logging.info(f"📊 Progression: {i}/{total} ({percent:.1f}%) - ✅ Réussis: {successful}, ❌ Échecs: {failed}")
    
    # Résumé final
    logging.info(f"""
    🏁 Crawl terminé:
    - Documents traités: {total}
    - Succès: {successful}
    - Échecs: {failed}
    - Taux de réussite: {(successful/total)*100:.1f}%
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
        print("Usage: python3 03_crawl_internal_links.py <crawl_id> <CSS Selector>")
        sys.exit(1)
    
    crawl_id = sys.argv[1]
    selector = sys.argv[2]
    
    documents = get_documents_from_redis(crawl_id)
    if not documents:
        print("No documents found for the given crawl ID.")
        return

    documents = assign_cluster_colors(documents)
    save_internal_links_to_redis(crawl_id, documents, selector)
    print("Internal links crawling complete.")

if __name__ == "__main__":
    main()