import subprocess
import sys
import os
import time

def run_crawl(url):
    print(f"Running crawl for URL: {url}")
    result = subprocess.run(['python3', '01_crawl.py', url], capture_output=True, text=True)
    print(f"Crawl output: {result.stdout}")
    print(f"Crawl errors: {result.stderr}")
    return result.stdout.split()[-1]  # Assumes the crawl ID is the last word in the output

def run_internal_links_crawl(crawl_id, selector):
    print(f"Running internal links crawl for ID: {crawl_id} with selector: {selector}")
    result = subprocess.run(['python3', '03_crawl_internal_links.py', crawl_id, selector], capture_output=True, text=True)
    print(f"Internal links crawl output: {result.stdout}")
    print(f"Internal links crawl errors: {result.stderr}")

def run_analysis(crawl_id):
    print(f"Running analysis for crawl ID: {crawl_id}")
    result = subprocess.run(['python3', '02_analyse.py', crawl_id], capture_output=True, text=True)
    print(f"Analysis output: {result.stdout}")
    print(f"Analysis errors: {result.stderr}")

def start_server():
    os.chdir('visualizations')
    print("Starting HTTP server for visualizations...")
    subprocess.run(['python3', '-m', 'http.server'])

def main(url, selector, depth):
    print("Starting crawl process...")
    crawl_id = run_crawl(url)
    print(f"Crawl completed with ID: {crawl_id}")

    print("Starting internal links crawl process...")
    run_internal_links_crawl(crawl_id, selector)
    print("Internal links crawl completed.")

    print("Starting analysis process...")
    run_analysis(crawl_id)
    print("Analysis process completed.")

    print("Launching visualizations...")
    start_server()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <URL> <CSS_SELECTOR> <DEPTH>")
        sys.exit(1)

    url = sys.argv[1]
    selector = sys.argv[2]
    depth = int(sys.argv[3])

    main(url, selector, depth)