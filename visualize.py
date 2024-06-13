import redis
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def retrieve_data(crawl_id):
    nodes = []
    edges = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        url = doc_data[b'url'].decode('utf-8')
        internal_links = eval(doc_data[b'internal_links'])
        cluster = doc_data.get(b'cluster')
        if cluster:
            cluster = int(cluster.decode('utf-8'))
        else:
            cluster = -1  # Utiliser -1 pour indiquer les documents non assignés
        nodes.append((url, cluster))
        for link in internal_links:
            edges.append((url, link))

    return nodes, edges

def visualize_graph(crawl_id):
    nodes, edges = retrieve_data(crawl_id)
    G = nx.DiGraph()
    for node, cluster in nodes:
        G.add_node(node, cluster=cluster)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    edge_trace = []
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Cluster',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
        node_trace['marker']['color'] += tuple([G.nodes[node]['cluster']])

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='<br>Graph des liens internes',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Visualisation des clusters sous forme de graphes forcés",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    pio.show(fig)

if __name__ == "__main__":
    crawl_id = input("Entrez l'ID du crawl à visualiser: ")
    visualize_graph(crawl_id)
