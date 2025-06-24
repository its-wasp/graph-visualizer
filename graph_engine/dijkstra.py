import heapq

def dijkstra_steps(graph, start):
    visited = set()
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    heap = [(0, start)]
    steps = []

    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        steps.append({
            "visited": list(visited),
            "current": node,
            "distances": dict(distances)
        })
        for neighbor in graph.adj[node]:
            weight = graph.get_edge_data(node, neighbor).get('weight', 1)
            if distances[neighbor] > dist + weight:
                distances[neighbor] = dist + weight
                heapq.heappush(heap, (distances[neighbor], neighbor))
    return steps
