from collections import deque

def bfs_steps(graph, start):
    visited = set()
    queue = deque([start])
    steps = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        steps.append({"visited": list(visited), "current": node})
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return steps
