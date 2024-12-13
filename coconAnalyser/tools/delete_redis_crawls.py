import redis
import subprocess
import sys


class RedisDeleteCLI:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.crawls = []
        self.graphs = []

    def list_available_crawls_and_graphs(self):
        """Liste tous les crawls (docs) et graphs disponibles dans Redis"""
        pattern_docs = "*:doc:*"
        pattern_graphs = "*_graph"
        crawl_ids = set()

        # Lister les crawls (docs)
        for key in self.redis.scan_iter(pattern_docs):
            try:
                key_str = key.decode("utf-8")
                crawl_id = key_str.split(":doc:")[0]
                if crawl_id:
                    crawl_ids.add(crawl_id)
            except Exception as e:
                print(f"Avertissement : Impossible de traiter la clé {key}: {str(e)}")

        self.crawls = sorted(crawl_ids)

        # Lister les graphs
        graph_ids = set()
        for key in self.redis.scan_iter(pattern_graphs):
            try:
                key_str = key.decode("utf-8")
                graph_id = key_str.split("_graph")[0]
                if graph_id:
                    graph_ids.add(graph_id)
            except Exception as e:
                print(f"Avertissement : Impossible de traiter la clé {key}: {str(e)}")

        self.graphs = sorted(graph_ids)

        return self.crawls, self.graphs

    def display_crawls_and_graphs(self):
        """Affiche les crawls et graphs disponibles dans un tableau"""
        print("\n=== CRAWLS (DOCS) DISPONIBLES ===")
        for index, crawl_id in enumerate(self.crawls, 1):
            print(f"{index}. {crawl_id}")

        print("\n=== GRAPHS DISPONIBLES ===")
        for index, graph_id in enumerate(self.graphs, 1):
            print(f"{index + len(self.crawls)}. {graph_id}")

        print("\nEntrez les numéros des éléments à supprimer, séparés par des virgules (ex: 1,2,3):")

    def delete_items(self, selected_ids):
        """Supprime les crawls (docs) et graphs correspondant aux IDs fournis"""
        deleted_keys_count = 0
        for item_id in selected_ids:
            patterns = [f"{item_id}:doc:*", f"{item_id}_graph*"]
            for pattern in patterns:
                for key in self.redis.scan_iter(pattern):
                    try:
                        self.redis.delete(key)
                        deleted_keys_count += 1
                    except Exception as e:
                        print(f"Avertissement : Impossible de supprimer la clé {key}: {str(e)}")
        return deleted_keys_count


def main():
    try:
        # Connexion à Redis
        # Connexion au service Redis dans DDEV
        redis_client = redis.Redis(host="127.0.0.1", port=32768, db=0, decode_responses=False)

        # Test de connexion
        redis_client.ping()

        # Création de l'outil
        cli = RedisDeleteCLI(redis_client)

        # Lister les crawls et graphs
        crawls, graphs = cli.list_available_crawls_and_graphs()
        if not crawls and not graphs:
            print("Aucun crawl ni graph disponible dans Redis.")
            return

        # Afficher les crawls et graphs disponibles
        cli.display_crawls_and_graphs()

        # Obtenir les choix de l'utilisateur
        while True:
            try:
                input_numbers = input("Votre choix : ").strip()
                selected_numbers = input_numbers.split(",")
                selected_numbers = [num.strip() for num in selected_numbers]

                # Mapping des numéros aux IDs
                all_items = cli.crawls + cli.graphs
                invalid_numbers = [num for num in selected_numbers if not num.isdigit() or int(num) < 1 or int(num) > len(all_items)]
                if invalid_numbers:
                    print(f"Numéros invalides : {', '.join(invalid_numbers)}")
                    continue

                # Convertir les numéros en IDs
                selected_ids = [all_items[int(num) - 1] for num in selected_numbers]

                # Supprimer les éléments sélectionnés
                deleted_count = cli.delete_items(selected_ids)
                print(f"\n{deleted_count} clés supprimées.")
                break
            except Exception as e:
                print(f"Erreur : {str(e)}")

    except Exception as e:
        print(f"Erreur lors de l'initialisation : {str(e)}")


if __name__ == "__main__":
    main()
