import streamlit as st
st.set_page_config(layout="wide")
from streamlit_agraph import agraph, Node, Edge, Config
import math
from graph_engine import get_algorithms
import pandas as pd
st.title("Graph Visualizer")

# Initialize session state
if "nodes" not in st.session_state:
    st.session_state["nodes"] = []
if "edges" not in st.session_state:
    st.session_state["edges"] = []
if "positions" not in st.session_state:
    st.session_state["positions"] = {}
if "bfs_steps" not in st.session_state:
    st.session_state["bfs_steps"] = []
if "bfs_step_idx" not in st.session_state:
    st.session_state["bfs_step_idx"] = 0
if "nodes_to_show" not in st.session_state:
    st.session_state["nodes_to_show"] = []
if "edges_to_show" not in st.session_state:
    st.session_state["edges_to_show"] = []

# Layout: controls on the left, graph on the right
left, right = st.columns([1, 2])  # Adjust ratio as you like

with left:
    graph_type = st.selectbox("Graph type", ["Undirected", "Directed"])

    mode = st.radio(
        "Choose graph creation mode:",
        ("Auto-generate nodes (1 to n)", "Manually define nodes and edges")
    )

    algorithms = get_algorithms()
    algorithm = st.selectbox(
        "Choose algorithm to visualize",
        list(algorithms.keys())
    )

    default_weight = st.number_input("Default edge weight", min_value=-1000000000, value=1, step=1)

    def assign_circular_positions(node_ids, radius=180, center=(0, 0)):
        n = len(node_ids)
        positions = {}
        for i, node_id in enumerate(node_ids):
            angle = 2 * math.pi * i / n
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions[node_id] = {"x": x, "y": y}
        return positions

    if mode == "Auto-generate nodes (1 to n)":
        n = st.number_input("Enter the number of nodes (n):", min_value=1, value=5, step=1)
        if st.button("Generate Nodes"):
            node_ids = [str(i) for i in range(1, n+1)]
            st.session_state["positions"] = assign_circular_positions(node_ids)
            st.session_state["nodes"] = [
                Node(id=node_id, label=node_id, x=st.session_state["positions"][node_id]["x"], y=st.session_state["positions"][node_id]["y"], fixed=True, size=15)
                for node_id in node_ids
            ]
            st.session_state["edges"] = []
            # Initialize default visualization
            st.session_state["nodes_to_show"] = [
                Node(
                    id=node_id,
                    label=node_id,
                    x=st.session_state["positions"][node_id]["x"],
                    y=st.session_state["positions"][node_id]["y"],
                    fixed=True,
                    color="#1f78b4",
                    font_color="#000",
                    font={"color": "#000"},
                    title=str(node_id),
                    size=15
                ) for node_id in node_ids
            ]
            st.session_state["edges_to_show"] = []
    elif mode == "Manually define nodes and edges":
        node_str = st.text_input("Enter nodes (comma-separated):", "A,B,C")
        if st.button("Generate Nodes"):
            node_list = [node.strip() for node in node_str.split(",") if node.strip()]
            st.session_state["positions"] = assign_circular_positions(node_list)
            st.session_state["nodes"] = [
                Node(id=node, label=node, x=st.session_state["positions"][node]["x"], y=st.session_state["positions"][node]["y"], fixed=True, size=15)
                for node in node_list
            ]
            st.session_state["edges"] = []
            # Initialize default visualization
            st.session_state["nodes_to_show"] = [
                Node(
                    id=node,
                    label=node,
                    x=st.session_state["positions"][node]["x"],
                    y=st.session_state["positions"][node]["y"],
                    fixed=True,
                    color="#1f78b4",
                    font_color="#000",
                    font={"color": "#000"},
                    title=str(node),
                    size=15
                ) for node in node_list
            ]
            st.session_state["edges_to_show"] = []

    # Add edge UI
    if st.session_state["nodes"]:
        node_ids = [node.id for node in st.session_state["nodes"]]
        src = st.selectbox("Source node for new edge", node_ids, key="src")
        tgt = st.selectbox("Target node for new edge", node_ids, key="tgt")
        if st.button("Add Edge"):
            if src != tgt:
                # Check if edge already exists
                found = False
                for edge in st.session_state["edges"]:
                    if edge.source == src and edge.to == tgt:
                        edge.label = str(default_weight)
                        edge.weight = default_weight
                        found = True
                        break
                    # For undirected, also check reverse
                    if graph_type == "Undirected" and edge.source == tgt and edge.to == src:
                        edge.label = str(default_weight)
                        edge.weight = default_weight
                        found = True
                        break
                if not found:
                    st.session_state["edges"].append(Edge(source=src, target=tgt, label=str(default_weight), weight=default_weight, width=3))
            else:
                st.warning("Source and target must be different.")

        # Remove edge UI
        if st.button("Remove Edge"):
            before = len(st.session_state["edges"])
            st.session_state["edges"] = [
                edge for edge in st.session_state["edges"]
                if not (
                    (edge.source == src and edge.to == tgt) or
                    (graph_type == "Undirected" and edge.source == tgt and edge.to == src)
                )
            ]
            after = len(st.session_state["edges"])
            if before == after:
                st.info("No such edge to remove.")
            else:
                st.success(f"Edge between {src} and {tgt} removed.")

        st.subheader(f"{algorithm} Visualization")
        # For algorithms that require a start node
        needs_start_algos = {"Bfs", "Dfs", "Dijkstra", "Bellman_ford", "Prim"}
        needs_start = algorithm in needs_start_algos
        # For shortest path algorithms, allow source and destination selection
        shortest_path_algos = {"Dijkstra", "Bellman_ford"}
        if algorithm in shortest_path_algos:
            default_src = node_ids[0] if node_ids else None
            default_dst = node_ids[-1] if node_ids else None
            sp_source = st.selectbox("Source node for shortest path", node_ids, index=0, key="sp_source")
            sp_target = st.selectbox("Destination node for shortest path", node_ids, index=len(node_ids)-1 if len(node_ids) > 1 else 0, key="sp_target")
        if needs_start:
            start_node = st.selectbox(f"Select start node for {algorithm}", node_ids, key="algo_start")
        else:
            start_node = None
        if st.button(f"Run {algorithm}"):
            import networkx as nx
            if graph_type == "Directed":
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            G.add_nodes_from(node_ids)
            for edge in st.session_state["edges"]:
                G.add_edge(edge.source, edge.to, weight=getattr(edge, 'weight', 1))
            try:
                if needs_start and start_node is not None:
                    steps = algorithms[algorithm](G, start_node)
                else:
                    steps = algorithms[algorithm](G)
                st.session_state["bfs_steps"] = steps
                st.session_state["bfs_step_idx"] = 0
                if algorithm in ["Kruskal", "Prim"]:
                    order = [str(step["current_edge"]) for step in steps if step.get("current_edge")]
                    st.success(f"{algorithm} Edge Order: {' → '.join(order)}")
                elif algorithm in ["Kahn", "Dfs_topo"]:
                    order = [str(step.get("current")) for step in steps if step.get("current")]
                    st.success(f"{algorithm} Node Order: {' → '.join(order)}")
                else:
                    order = [str(step.get("current", step.get("current_nodes", ''))) for step in steps]
                    st.success(f"{algorithm} Order: {' → '.join(order)}")
                # Save source and target for shortest path highlighting
                if algorithm in shortest_path_algos:
                    st.session_state["sp_source"] = sp_source
                    st.session_state["sp_target"] = sp_target
            except:
                pass

        # Show stepper only if algorithm is running
        if st.session_state["bfs_steps"]:
            step_idx = st.session_state["bfs_step_idx"]
            steps = st.session_state["bfs_steps"]
            current_step = steps[step_idx]
            col1, col2 = st.columns(2)

            # Handle step navigation
            with col1:
                if st.button("Previous Step", key="prev_step") and step_idx > 0:
                    st.session_state["bfs_step_idx"] -= 1
                    step_idx = st.session_state["bfs_step_idx"]
                    current_step = steps[step_idx]

            with col2:
                if st.button("Next Step", key="next_step") and step_idx < len(steps) - 1:
                    st.session_state["bfs_step_idx"] += 1
                    step_idx = st.session_state["bfs_step_idx"]
                    current_step = steps[step_idx]

            st.write(f"Step {step_idx+1} of {len(steps)}")

            # Display algorithm-specific information
            try:
                # Bellman-Ford
                if algorithm == "Bellman_ford":
                    current_edge = current_step.get('current_edge', 'N/A')
                    distances = current_step.get('distances', {})
                    iteration = current_step.get('iteration', 'N/A')
                    
                    st.write(f"Current edge: {current_edge}")
                    st.write(f"Distances: {distances}")
                    st.write(f"Iteration: {iteration}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('negative_cycle', False):
                            st.info("Negative weight cycle detected!")
                        else:
                            st.success("No negative weight cycle detected!")
                            # Display final distances in a table
                            st.subheader("Final Shortest Path Distances")
                            df_data = []
                            for node in sorted(distances.keys()):
                                dist = distances[node]
                                df_data.append({
                                    "Node": node,
                                    "Distance from Source": "∞" if dist == float('inf') else dist
                                })
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, hide_index=True)

                # Floyd-Warshall
                elif algorithm == "Floyd_warshall":
                    current_nodes = current_step.get('current_nodes', 'N/A')
                    distances = current_step.get('distances', {})
                    
                    st.write(f"Current nodes (k,i,j): {current_nodes}")
                    st.write(f"Distances: {distances}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('negative_cycle', False):
                            st.info("Negative weight cycle detected!")
                        else:
                            st.success("No negative weight cycle detected!")
                            # Display final distances in a table
                            st.subheader("All Pairs Shortest Path Distances")
                            # Convert the nested dictionary to a DataFrame
                            nodes = sorted(distances.keys())
                            df_data = []
                            for source in nodes:
                                row = {"Source": source}
                                for target in nodes:
                                    dist = distances[source][target]
                                    row[f"To {target}"] = "∞" if dist == float('inf') else dist
                                df_data.append(row)
                            df = pd.DataFrame(df_data)
                            st.dataframe(df, hide_index=True)

                # Kahn's
                elif algorithm == "Kahn":
                    st.write(f"Current: {current_step.get('current', 'N/A')}")
                    st.write(f"In-degree: {current_step.get('in_degree', {})}")
                    st.write(f"Queue: {current_step.get('queue', [])}")
                    st.write(f"Sorted list: {current_step.get('sorted_list', [])}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('cycle_detected', False):
                            st.info("Cycle detected! Topological sort not possible.")
                        else:
                            st.success("No cycle detected. Topological sort possible.")

                # DFS-based Topo Sort
                elif algorithm == "Dfs_topo":
                    st.write(f"Current: {current_step.get('current', 'N/A')}")
                    st.write(f"Visited: {list(current_step.get('visited', []))}")
                    st.write(f"Sorted list: {current_step.get('sorted_list', [])}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('cycle_detected', False):
                            st.info("Cycle detected! Topological sort not possible.")
                        else:
                            st.success("No cycle detected. Topological sort possible.")

                # Cycle Detection Undirected
                elif algorithm == "Cycle_undirected":
                    st.write(f"Current: {current_step.get('current', 'N/A')}")
                    st.write(f"Visited: {list(current_step.get('visited', []))}")
                    st.write(f"Parent: {current_step.get('parent', 'N/A')}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('cycle_found', False):
                            st.info("Cycle detected in undirected graph!")
                        else:
                            st.success("No cycle detected in undirected graph.")

                # Cycle Detection Directed
                elif algorithm == "Cycle_directed":
                    st.write(f"Current: {current_step.get('current', 'N/A')}")
                    st.write(f"Visited: {list(current_step.get('visited', []))}")
                    st.write(f"Recursion stack: {list(current_step.get('rec_stack', []))}")
                    
                    if step_idx == len(steps) - 1:
                        if current_step.get('cycle_found', False):
                            st.info("Cycle detected in directed graph!")
                        else:
                            st.success("No cycle detected in directed graph.")

                # MST
                elif algorithm in ["Kruskal", "Prim"]:
                    st.write(f"MST edges so far: {current_step.get('edges_in_mst', [])}")
                    st.write(f"Current edge considered: {current_step.get('current_edge', 'N/A')}")
                    st.write(f"MST total weight so far: {current_step.get('mst_weight', 0)}")
                    if 'visited' in current_step:
                        st.write(f"Nodes in MST so far: {list(current_step.get('visited', []))}")

                # Dijkstra
                elif 'distances' in current_step:
                    distances = current_step.get('distances', {})
                    current = current_step.get('current', '')
                    min_dist = distances.get(current, float('inf'))
                    st.write(f"Current distances: {distances}")
                    st.write(f"Minimum weight until current node ({current}): {min_dist}")
                    
                    if step_idx == len(steps) - 1:
                        # Display final distances in a table
                        st.subheader("Final Shortest Path Distances")
                        df_data = []
                        for node in sorted(distances.keys()):
                            dist = distances[node]
                            df_data.append({
                                "Node": node,
                                "Distance from Source": "∞" if dist == float('inf') else dist
                            })
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, hide_index=True)
            except:
                pass

with right:
    # Initialize visualization variables in session state if they don't exist
    if "nodes_to_show" not in st.session_state:
        st.session_state["nodes_to_show"] = []
    if "edges_to_show" not in st.session_state:
        st.session_state["edges_to_show"] = []

    try:
        if st.session_state["bfs_steps"]:
            try:
                step_idx = st.session_state["bfs_step_idx"]
                steps = st.session_state["bfs_steps"]
                current_step = steps[step_idx]
                visited = current_step.get("visited", [])
                current = current_step.get("current", None)
                mst_edges = current_step.get("edges_in_mst", [])
                highlight_edges = set()
                highlight_nodes = set()
                # Highlight shortest path for Dijkstra and Bellman-Ford
                if algorithm in ["Dijkstra", "Bellman_ford"] and step_idx == len(steps) - 1:
                    try:
                        distances = current_step.get("distances", {})
                        sp_source = st.session_state.get("sp_source")
                        sp_target = st.session_state.get("sp_target")
                        if distances and sp_source and sp_target and sp_source in distances and sp_target in distances and distances[sp_target] < float('inf'):
                            import networkx as nx
                            if graph_type == "Directed":
                                G = nx.DiGraph()
                            else:
                                G = nx.Graph()
                            node_ids = [node.id for node in st.session_state["nodes"]]
                            G.add_nodes_from(node_ids)
                            for edge in st.session_state["edges"]:
                                G.add_edge(edge.source, edge.to, weight=getattr(edge, 'weight', 1))
                            try:
                                path = nx.shortest_path(G, source=sp_source, target=sp_target, weight='weight')
                                highlight_nodes.update(path)
                                highlight_edges.update([(path[i], path[i+1]) for i in range(len(path)-1)])
                            except:
                                pass
                    except:
                        pass
                # Highlight cycle for cycle detection and topo sort/cycle detection algorithms
                cycle_algos = ["Cycle_undirected", "Cycle_directed", "Kahn", "Dfs_topo"]
                cycle_detected = False
                if algorithm in cycle_algos:
                    # For direct cycle detection
                    if algorithm in ["Cycle_undirected", "Cycle_directed"] and current_step.get('cycle_found', False):
                        cycle_detected = True
                    # For Kahn and Dfs_topo, check final step for cycle_detected
                    if algorithm in ["Kahn", "Dfs_topo"] and step_idx == len(steps) - 1 and current_step.get('cycle_detected', False):
                        cycle_detected = True
                if cycle_detected:
                    import networkx as nx
                    if graph_type == "Directed":
                        G = nx.DiGraph()
                    else:
                        G = nx.Graph()
                    node_ids = [node.id for node in st.session_state["nodes"]]
                    G.add_nodes_from(node_ids)
                    for edge in st.session_state["edges"]:
                        G.add_edge(edge.source, edge.to, weight=getattr(edge, 'weight', 1))
                    try:
                        if graph_type == "Directed":
                            cycle = next(nx.simple_cycles(G))
                            highlight_nodes.update(cycle)
                            highlight_edges.update([(cycle[i], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))])
                        else:
                            cycle = nx.find_cycle(G)
                            highlight_edges.update([(u, v) for u, v in cycle])
                            highlight_nodes.update([u for u, v in cycle] + [cycle[-1][1]])
                    except:
                        pass
                # Highlight nodes for Floyd-Warshall (k, i, j)
                if algorithm == "Floyd_warshall" and current_step.get("current_nodes"):
                    k, i, j = current_step["current_nodes"]
                    highlight_nodes.update([k, i, j])
                # Highlight nodes for Bellman-Ford (current edge)
                if algorithm == "Bellman_ford" and current_step.get("current_edge"):
                    u, v = current_step["current_edge"]
                    highlight_nodes.update([u, v])
                # Create temporary lists for visualization
                temp_nodes = []
                for node in [node.id for node in st.session_state["nodes"]]:
                    color = "#1f78b4"
                    if node == current:
                        color = "#e41a1c"
                    elif node in visited:
                        color = "#4daf4a"
                    elif algorithm in ["Kruskal", "Prim"] and node in visited:
                        color = "#ff9800"
                    if node in highlight_nodes:
                        if algorithm == "Floyd_warshall":
                            color = "#ff9800"  # orange for k, i, j
                        elif algorithm == "Bellman_ford":
                            color = "#ffd700"  # yellow for Bellman-Ford step
                        else:
                            color = "#9c27b0" if algorithm in ["Dijkstra", "Bellman_ford"] else "#d32f2f"
                    temp_nodes.append(
                        Node(
                            id=node,
                            label=node,
                            x=st.session_state["positions"][node]["x"],
                            y=st.session_state["positions"][node]["y"],
                            fixed=True,
                            color=color,
                            font_color="#000",
                            font={"color": "#000"},
                            title=str(node),
                            size=15
                        )
                    )

                temp_edges = []
                for edge in st.session_state["edges"]:
                    edge_color = "#848484"
                    edge_width = 3
                    if algorithm in ["Kruskal", "Prim"] and (edge.source, edge.to) in mst_edges or (edge.to, edge.source) in mst_edges:
                        edge_color = "#ff9800"
                        edge_width = 5
                    if (edge.source, edge.to) in highlight_edges or (edge.to, edge.source) in highlight_edges:
                        edge_color = "#9c27b0" if algorithm in ["Dijkstra", "Bellman_ford"] else "#d32f2f"
                        edge_width = 5
                    temp_edges.append(
                        Edge(source=edge.source, target=edge.to, label=edge.label, weight=edge.weight, color=edge_color, width=edge_width)
                    )

                # Update session state
                st.session_state["nodes_to_show"] = temp_nodes
                st.session_state["edges_to_show"] = temp_edges
            except:
                # If algorithm visualization fails, fall back to default visualization
                st.session_state["nodes_to_show"] = [
                    Node(
                        id=node.id,
                        label=node.id,
                        x=st.session_state["positions"][node.id]["x"],
                        y=st.session_state["positions"][node.id]["y"],
                        fixed=True,
                        color="#1f78b4",
                        font_color="#000",
                        font={"color": "#000"},
                        title=str(node.id),
                        size=15
                    ) for node in st.session_state["nodes"]
                ]
                st.session_state["edges_to_show"] = [
                    Edge(
                        source=edge.source,
                        target=edge.to,
                        label=edge.label,
                        weight=edge.weight,
                        color="#848484",
                        width=3
                    ) for edge in st.session_state["edges"]
                ]
        else:
            # Default visualization when no algorithm is running
            st.session_state["nodes_to_show"] = [
                Node(
                    id=node.id,
                    label=node.id,
                    x=st.session_state["positions"][node.id]["x"],
                    y=st.session_state["positions"][node.id]["y"],
                    fixed=True,
                    color="#1f78b4",
                    font_color="#000",
                    font={"color": "#000"},
                    title=str(node.id),
                    size=15
                ) for node in st.session_state["nodes"]
            ]
            st.session_state["edges_to_show"] = [
                Edge(
                    source=edge.source,
                    target=edge.to,
                    label=edge.label,
                    weight=edge.weight,
                    color="#848484",
                    width=3
                ) for edge in st.session_state["edges"]
            ]
    except:
        pass

    # Configure and display the graph
    config = Config(
        width=1200,
        height=1500,
        directed=(graph_type == "Directed"),
        physics=True,
        hierarchical=False,
    )

    if "zoom" in st.session_state:
        try:
            config.zoom = st.session_state["zoom"]
        except:
            pass

    # Use session state for visualization
    result = agraph(
        nodes=st.session_state["nodes_to_show"],
        edges=st.session_state["edges_to_show"],
        config=config
    )

    # Update positions and zoom
    if result and "nodes" in result:
        for node in result["nodes"]:
            node_id = node["id"]
            if "x" in node and "y" in node:
                st.session_state["positions"][node_id] = {"x": node["x"], "y": node["y"]}
    if result and ("zoom" in result or "scale" in result):
        st.session_state["zoom"] = result.get("zoom", result.get("scale"))