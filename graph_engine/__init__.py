import importlib
import os
import sys

ALGORITHM_FUNCS = {}

current_dir = os.path.dirname(__file__)
for fname in os.listdir(current_dir):
    if fname.endswith('.py') and fname not in ('__init__.py', 'graph.py'):
        modname = f"graph_engine.{fname[:-3]}"
        mod = importlib.import_module(modname)
        for attr in dir(mod):
            if attr.endswith('_steps') and callable(getattr(mod, attr)):
                # Use the algorithm name (e.g., 'bfs', 'dfs', 'dijkstra')
                algo_name = attr.replace('_steps', '').capitalize()
                ALGORITHM_FUNCS[algo_name] = getattr(mod, attr)

def get_algorithms():
    return ALGORITHM_FUNCS.copy()
