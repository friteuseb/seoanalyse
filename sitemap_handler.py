import requests
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from typing import List, Set, Optional
import logging
import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class SitemapHandler:
    def __init__(self, base_url: str, max_workers: int = 5, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        self.namespaces = {
            'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }

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
            '/sitemap.xml', '/sitemap_index.xml', '/wp-sitemap.xml',
            '/?type=1533906435', '/media/sitemap.xml'
        ]
        found_sitemaps = []
        for path in common_paths:
            url = urljoin(self.base_url, path)
            if self._fetch_url(url):
                found_sitemaps.append(url)
        return found_sitemaps

    def _parse_sitemap(self, content: str) -> List[str]:
        """Parse un sitemap XML et extrait les URLs et sous-sitemaps."""
        try:
            root = ET.fromstring(content)
            urls = [url.text for url in root.findall('.//sm:url/sm:loc', self.namespaces)]
            sub_sitemaps = [url.text for url in root.findall('.//sm:sitemap/sm:loc', self.namespaces)]
            return urls, sub_sitemaps
        except ET.ParseError:
            logging.warning("Erreur de parsing XML")
            return [], []

    def _fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """Récupère et parse un sitemap donné."""
        content = self._fetch_url(sitemap_url, compressed=sitemap_url.endswith('.gz'))
        if not content:
            return []
        urls, sub_sitemaps = self._parse_sitemap(content)
        for sub_sitemap in sub_sitemaps:
            urls.extend(self._fetch_sitemap_urls(sub_sitemap))
        return urls

    def discover_sitemaps(self) -> List[str]:
        """Découvre tous les sitemaps disponibles."""
        sitemaps = self._find_sitemaps_in_robots()
        if not sitemaps:
            sitemaps = self._check_common_sitemaps()
        return sitemaps

    def discover_urls(self) -> List[str]:
        """Découvre toutes les URLs des sitemaps."""
        sitemaps = self.discover_sitemaps()
        all_urls = set()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sitemap = {
                executor.submit(self._fetch_sitemap_urls, sitemap): sitemap for sitemap in sitemaps
            }
            for future in as_completed(future_to_sitemap):
                try:
                    all_urls.update(future.result())
                except Exception as e:
                    logging.warning(f"Erreur lors du traitement d'un sitemap: {e}")
        return list(all_urls)
