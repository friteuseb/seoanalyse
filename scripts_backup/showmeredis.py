import redis

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def retrieve_data_from_redis():
    for key in r.scan_iter("doc:*"):
        doc_data = r.hgetall(key)
        url = doc_data[b'url'].decode('utf-8')
        content = doc_data[b'content'].decode('utf-8')
        cluster = doc_data.get(b'cluster')
        if cluster:
            cluster = cluster.decode('utf-8')
        else:
            cluster = "Non assigné"
        print(f"URL: {url}\nCluster: {cluster}\nContenu: {content[:100]}...\n")

retrieve_data_from_redis()

