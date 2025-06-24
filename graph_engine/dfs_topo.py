def dfs_topo_steps(graph):
    visited = set()
    stack = []
    sorted_list = []
    steps = []
    cycle_detected = False
    temp_mark = set()
    def visit(u):
        nonlocal cycle_detected
        if u in temp_mark:
            cycle_detected = True
            return
        if u not in visited:
            temp_mark.add(u)
            steps.append({'visited': set(visited), 'stack': list(stack), 'current': u, 'sorted_list': list(sorted_list), 'cycle_detected': cycle_detected})
            for v in graph.adj[u]:
                visit(v)
            temp_mark.remove(u)
            visited.add(u)
            sorted_list.append(u)
            steps.append({'visited': set(visited), 'stack': list(stack), 'current': u, 'sorted_list': list(sorted_list), 'cycle_detected': cycle_detected})
    for u in graph.nodes:
        if u not in visited:
            visit(u)
    steps.append({'visited': set(visited), 'stack': list(stack), 'current': None, 'sorted_list': list(reversed(sorted_list)), 'cycle_detected': cycle_detected})
    return steps 