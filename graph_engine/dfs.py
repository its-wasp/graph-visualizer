from collections import deque

def dfs_steps(graph, start):
    visited = set()
    stack = [start]
    steps = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        steps.append({"visited": list(visited), "current": node})
        # Add neighbors in reverse order for consistent traversal
        for neighbor in reversed(list(graph.adj[node])):
            if neighbor not in visited:
                stack.append(neighbor)
    return steps 