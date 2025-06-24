def kahn_steps(graph):
    from collections import deque
    in_degree = {u: 0 for u in graph.nodes}
    for u in graph.nodes:
        for v in graph.adj[u]:
            in_degree[v] += 1
    queue = deque([u for u in graph.nodes if in_degree[u] == 0])
    sorted_list = []
    steps = []
    while queue:
        u = queue.popleft()
        step = {
            'in_degree': dict(in_degree),
            'queue': list(queue),
            'sorted_list': list(sorted_list),
            'current': u,
            'cycle_detected': False
        }
        sorted_list.append(u)
        for v in graph.adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        steps.append(step)
    cycle_detected = len(sorted_list) != len(graph.nodes)
    steps.append({'in_degree': dict(in_degree), 'queue': list(queue), 'sorted_list': list(sorted_list), 'current': None, 'cycle_detected': cycle_detected})
    return steps 