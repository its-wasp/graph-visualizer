def floyd_warshall_steps(graph):
    nodes = list(graph.nodes)
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}
    for u in nodes:
        dist[u][u] = 0
    for u, v, data in graph.edges(data=True):
        dist[u][v] = min(dist[u][v], data.get('weight', 1))
    steps = []
    negative_cycle = False
    for k in nodes:
        for i in nodes:
            for j in nodes:
                step = {
                    'distances': {x: dict(dist[x]) for x in nodes},
                    'current_nodes': (k, i, j),
                    'negative_cycle': False
                }
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                steps.append(step)
    # Check for negative cycles
    for n in nodes:
        if dist[n][n] < 0:
            negative_cycle = True
    steps.append({'distances': {x: dict(dist[x]) for x in nodes}, 'current_nodes': None, 'negative_cycle': negative_cycle})
    return steps 