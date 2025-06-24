import heapq

def prim_steps(graph, start):
    visited = set([start])
    edges = []
    for to in graph.adj[start]:
        weight = graph.get_edge_data(start, to).get('weight', 1)
        heapq.heappush(edges, (weight, start, to))
    mst = []
    steps = []
    mst_weight = 0
    while edges:
        weight, u, v = heapq.heappop(edges)
        step = {
            'edges_in_mst': list(mst),
            'current_edge': (u, v),
            'mst_weight': mst_weight,
            'visited': set(visited)
        }
        if v not in visited:
            visited.add(v)
            mst.append((u, v))
            mst_weight += weight
            for to in graph.adj[v]:
                if to not in visited:
                    w = graph.get_edge_data(v, to).get('weight', 1)
                    heapq.heappush(edges, (w, v, to))
        steps.append(step)
    # Final step
    steps.append({'edges_in_mst': list(mst), 'current_edge': None, 'mst_weight': mst_weight, 'visited': set(visited)})
    return steps 