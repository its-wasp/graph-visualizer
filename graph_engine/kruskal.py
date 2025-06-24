def kruskal_steps(graph):
    # Kruskal's algorithm for MST (undirected only)
    parent = {}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            return True
        return False
    for node in graph.nodes:
        parent[node] = node
    edges = sorted(graph.edges(data=True), key=lambda x: x[2].get('weight', 1))
    mst = []
    steps = []
    mst_weight = 0
    for u, v, data in edges:
        step = {
            'edges_in_mst': list(mst),
            'current_edge': (u, v),
            'mst_weight': mst_weight
        }
        if union(u, v):
            mst.append((u, v))
            mst_weight += data.get('weight', 1)
        steps.append(step)
    # Final step
    steps.append({'edges_in_mst': list(mst), 'current_edge': None, 'mst_weight': mst_weight})
    return steps 