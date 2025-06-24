def bellman_ford_steps(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    steps = []
    edges = list(graph.edges(data=True))
    negative_cycle = False
    for i in range(len(graph.nodes) - 1):
        for u, v, data in edges:
            step = {
                'distances': dict(distances),
                'current_edge': (u, v),
                'iteration': i+1,
                'negative_cycle': False
            }
            if distances[u] + data.get('weight', 1) < distances[v]:
                distances[v] = distances[u] + data.get('weight', 1)
            steps.append(step)
    # Check for negative-weight cycles
    for u, v, data in edges:
        if distances[u] + data.get('weight', 1) < distances[v]:
            negative_cycle = True
    steps.append({'distances': dict(distances), 'current_edge': None, 'iteration': 'done', 'negative_cycle': negative_cycle})
    return steps 