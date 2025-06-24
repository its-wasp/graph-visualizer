def cycle_undirected_steps(graph):
    visited = set()
    steps = []
    cycle_found = False
    def dfs(u, parent):
        nonlocal cycle_found
        visited.add(u)
        steps.append({'visited': set(visited), 'current': u, 'parent': parent, 'cycle_found': cycle_found})
        for v in graph.adj[u]:
            if v == parent:
                continue
            if v in visited:
                cycle_found = True
                steps.append({'visited': set(visited), 'current': v, 'parent': u, 'cycle_found': cycle_found})
                return
            dfs(v, u)
    for u in graph.nodes:
        if u not in visited and not cycle_found:
            dfs(u, None)
    steps.append({'visited': set(visited), 'current': None, 'parent': None, 'cycle_found': cycle_found})
    return steps 