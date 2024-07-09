import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
import redis
import uuid
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

class WebsiteCrawler(scrapy.Spider):
    name = 'website_crawler'

    def __init__(self, start_url=None, allowed_domains=None, restrict_path=None, exclude_paths=None, max_pages=None, *args, **kwargs):
        super(WebsiteCrawler, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [allowed_domains] if allowed_domains else []
        self.restrict_path = restrict_path
        self.exclude_paths = exclude_paths or []
        self.max_pages = int(max_pages) if max_pages else None
        self.crawl_id = f"{urlparse(start_url).netloc}__{uuid.uuid4()}"
        self.doc_count = 0

    def parse(self, response):
        if self.max_pages and self.doc_count >= self.max_pages:
            logging.info(f"Reached maximum number of pages ({self.max_pages}). Stopping crawler.")
            return

        if self.restrict_path and not response.url.startswith(self.start_urls[0] + self.restrict_path):
            logging.debug(f"Skipping URL outside of restricted path: {response.url}")
            return

        if any(exclude_path in response.url for exclude_path in self.exclude_paths):
            logging.debug(f"Skipping excluded URL: {response.url}")
            return

        content = response.css('body').get()
        if content:
            doc_id = f"{self.crawl_id}:doc:{self.doc_count}"
            r.hset(doc_id, mapping={
                'url': response.url,
                'content': content
            })
            self.doc_count += 1
            r.set(f"{self.crawl_id}:doc_count", self.doc_count)
            logging.info(f"Crawled page {self.doc_count}: {response.url}")

        le = LinkExtractor(allow_domains=self.allowed_domains)
        for link in le.extract_links(response):
            if self.restrict_path and not link.url.startswith(self.start_urls[0] + self.restrict_path):
                continue
            if any(exclude_path in link.url for exclude_path in self.exclude_paths):
                continue
            yield scrapy.Request(link.url, callback=self.parse)

def run_crawler(url, restrict_path=None, exclude_paths=None, max_pages=None):
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    parsed_url = urlparse(url)
    allowed_domain = parsed_url.netloc

    process.crawl(WebsiteCrawler, start_url=url, allowed_domains=allowed_domain, restrict_path=restrict_path, exclude_paths=exclude_paths, max_pages=max_pages)
    process.start()

    crawl_id = f"{parsed_url.netloc}__{uuid.uuid4()}"
    doc_count = r.get(f"{crawl_id}:doc_count")

    logging.info(f"Crawl terminé. ID du crawl: {crawl_id}")
    logging.info(f"Nombre de documents crawlés: {doc_count.decode('utf-8') if doc_count else 0}")

    return crawl_id

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 01_crawl.py <URL> [restrict_path] [exclude_paths] [max_pages]")
        print("Example: python3 01_crawl.py https://example.com /fr /en,/de 100")
        sys.exit(1)

    url = sys.argv[1]
    restrict_path = sys.argv[2] if len(sys.argv) > 2 else None
    exclude_paths = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    max_pages = sys.argv[4] if len(sys.argv) > 4 else None

    run_crawler(url, restrict_path, exclude_paths, max_pages)