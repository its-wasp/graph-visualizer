def cycle_directed_steps(graph):
    visited = set()
    rec_stack = set()
    steps = []
    cycle_found = False
    def dfs(u):
        nonlocal cycle_found
        visited.add(u)
        rec_stack.add(u)
        steps.append({'visited': set(visited), 'current': u, 'rec_stack': set(rec_stack), 'cycle_found': cycle_found})
        for v in graph.adj[u]:
            if v not in visited:
                dfs(v)
            elif v in rec_stack:
                cycle_found = True
                steps.append({'visited': set(visited), 'current': v, 'rec_stack': set(rec_stack), 'cycle_found': cycle_found})
                return
        rec_stack.remove(u)
    for u in graph.nodes:
        if u not in visited and not cycle_found:
            dfs(u)
    steps.append({'visited': set(visited), 'current': None, 'rec_stack': set(), 'cycle_found': cycle_found})
    return steps 