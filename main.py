import subprocess
import sys
import redis
import time
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

def run_crawl(url):
    result = subprocess.run(['python3', '01_crawl.py', url], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Crawl errors: {result.stderr}")
    return result.stdout.split("ID du crawl: ")[1].strip()

def run_internal_links_crawl(crawl_id, selector):
    result = subprocess.run(['python3', '03_crawl_internal_links.py', crawl_id, selector], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Internal links crawl errors: {result.stderr}")

def run_analysis(crawl_id):
    result = subprocess.run(['python3', '02_analyse.py', crawl_id], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Analysis errors: {result.stderr}")

def start_server(port=8000):
    handler = SimpleHTTPRequestHandler
    with TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()

def main(url, selector, depth):
    print("Starting crawl process...")
    crawl_id = run_crawl(url)
    print(f"Crawl completed with ID: {crawl_id}")

    time.sleep(2)  # Wait for a short period to ensure data is written to Redis

    print("Starting internal links crawl process...")
    run_internal_links_crawl(crawl_id, selector)
    print("Internal links crawl completed.")

    print("Starting analysis process...")
    run_analysis(crawl_id)
    print("Analysis completed.")

    # Start the HTTP server
    start_server()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <URL> <CSS Selector> <Depth>")
        sys.exit(1)

    url = sys.argv[1]
    selector = sys.argv[2]
    depth = int(sys.argv[3])

    main(url, selector, depth)
