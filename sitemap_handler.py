import requests
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from typing import List, Set, Optional
import logging
import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class SitemapHandler:
    def __init__(self, base_url: str, scope_url: Optional[str] = None, max_workers: int = 5, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.scope_url = scope_url.rstrip('/') if scope_url else self.base_url
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        self.namespaces = {
            'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }
        
        # Configuration du user-agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SeoAnalyseBot/1.0)'
        })

    def _is_url_in_scope(self, url: str) -> bool:
        """Vérifie si une URL est dans la portée définie."""
        if not self.scope_url:
            return True
        return url.startswith(self.scope_url)

    def discover_urls(self) -> List[str]:
        """Découvre toutes les URLs des sitemaps en respectant le scope."""
        sitemaps = self.discover_sitemaps()
        all_urls = set()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sitemap = {
                executor.submit(self._fetch_sitemap_urls, sitemap): sitemap 
                for sitemap in sitemaps
            }
            
            for future in as_completed(future_to_sitemap):
                try:
                    # Filtrer les URLs selon le scope
                    urls = future.result()
                    scoped_urls = [url for url in urls if self._is_url_in_scope(url)]
                    all_urls.update(scoped_urls)
                    
                    logging.info(f"""
                    📍 URLs trouvées : {len(urls)}
                    • Dans le scope : {len(scoped_urls)}
                    • Scope : {self.scope_url}
                    """)
                    
                except Exception as e:
                    logging.warning(f"Erreur lors du traitement d'un sitemap: {e}")
                    
        return list(all_urls)
    
    def _fetch_url(self, url: str, compressed: bool = False) -> Optional[str]:
        """Récupère le contenu d'une URL avec gestion des erreurs."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            if compressed or response.headers.get('content-type', '').startswith('application/x-gzip'):
                return gzip.decompress(response.content).decode('utf-8')
            return response.text
        except Exception as e:
            logging.warning(f"Erreur lors de la récupération de {url}: {str(e)}")
            return None

    def _find_sitemaps_in_robots(self) -> List[str]:
        """Recherche les sitemaps dans le fichier robots.txt."""
        robots_url = f"{self.base_url}/robots.txt"
        sitemaps = []
        content = self._fetch_url(robots_url)
        if content:
            for line in content.splitlines():
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    sitemaps.append(sitemap_url)
        return sitemaps

    def _check_common_sitemaps(self) -> List[str]:
        """Vérifie les emplacements courants de sitemaps pour les CMS."""
        common_paths = [
            # WordPress
            '/wp-sitemap.xml',           # WordPress 5.5+
            '/wp-sitemap-posts-post-1.xml', # Posts
            '/wp-sitemap-posts-page-1.xml', # Pages
            '/sitemap.xml',              # Yoast SEO
            '/sitemap_index.xml',        # WordPress ancien
            '/post-sitemap.xml',         # All in One SEO
            '/page-sitemap.xml',
            
            # Autres CMS
            '/sitemap_index.xml',        
            '/sitemap.xml',              
            '/?type=1533906435',         # TYPO3
            '/media/sitemap.xml'         # Autre
        ]
        
        found_sitemaps = []
        for path in common_paths:
            url = urljoin(self.base_url, path)
            try:
                response = self.session.head(url, timeout=self.timeout)
                if response.status_code == 200:
                    found_sitemaps.append(url)
                    logging.info(f"✅ Sitemap trouvé : {url}")
            except Exception as e:
                logging.debug(f"❌ Sitemap non trouvé à {url}: {e}")
                continue
                
        return found_sitemaps


    def _parse_sitemap(self, content: str) -> tuple[list[str], list[str]]:
        """Parse un sitemap XML et extrait les URLs et sous-sitemaps."""
        try:
            # Nettoyer le contenu XML
            content = content.strip()
            if not content:
                return [], []
                
            root = ET.fromstring(content)
            
            # Détecter le type de sitemap
            if root.tag.endswith('sitemapindex'):
                logging.info("Détection d'un index de sitemap")
                sub_sitemaps = [url.text for url in root.findall('.//sm:sitemap/sm:loc', self.namespaces)]
                return [], sub_sitemaps
                
            elif root.tag.endswith('urlset'):
                urls = [url.text for url in root.findall('.//sm:url/sm:loc', self.namespaces)]
                return urls, []
                
            else:
                logging.warning(f"Type de sitemap inconnu: {root.tag}")
                return [], []
                
        except ET.ParseError as e:
            logging.warning(f"Erreur de parsing XML: {str(e)}")
            # Essayer de parser comme HTML
            try:
                soup = BeautifulSoup(content, 'html.parser')
                urls = [a['href'] for a in soup.find_all('a', href=True)]
                return urls, []
            except Exception:
                return [], []
        except Exception as e:
            logging.warning(f"Erreur inattendue lors du parsing: {str(e)}")
            return [], []


    def _fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Récupère et parse un sitemap donné."""
        content = self._fetch_url(sitemap_url, compressed=sitemap_url.endswith('.gz'))
        if not content:
            logging.warning(f"Pas de contenu pour {sitemap_url}")
            return []
            
        # Log du début du contenu pour debug
        logging.debug(f"Début du contenu de {sitemap_url}: {content[:200]}...")
            
        urls, sub_sitemaps = self._parse_sitemap(content)
        
        all_urls = urls.copy()
        for sub_sitemap in sub_sitemaps:
            sub_urls = self._fetch_sitemap_urls(sub_sitemap)
            all_urls.extend(sub_urls)
            
        return all_urls

    def discover_sitemaps(self) -> List[str]:
        """Découvre tous les sitemaps disponibles."""
        sitemaps = self._find_sitemaps_in_robots()
        if not sitemaps:
            sitemaps = self._check_common_sitemaps()
        return sitemaps

