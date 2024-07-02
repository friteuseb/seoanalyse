import redis
import json

def generate_link_suggestions(crawl_id):
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Exemple de données de suggestion de liens
    suggestions = [
        {
            "page": "https://example.com/page1",
            "links": ["https://example.com/page2", "https://example.com/page3"]
        },
        {
            "page": "https://example.com/page2",
            "links": ["https://example.com/page1", "https://example.com/page4"]
        }
    ]

    r.set(f"{crawl_id}:link_suggestions", json.dumps(suggestions))
    print(f"Suggestions de liens enregistrées pour le crawl ID {crawl_id}")

if __name__ == "__main__":
    generate_link_suggestions('example_crawl')
