from fastmcp import FastMCP, Image
import pandas as pd
import numpy as np
from lingam import DirectLiNGAM
import networkx as nx
import matplotlib.pyplot as plt
import io
import logging

# 1) Create your MCP server
mcp = FastMCP("CausalGraphService")

# 2) Decorate your function as a tool
@mcp.tool(name="learn_and_plot_causal_graph",
          description="Learn & plot a causal graph from CSV.")
async def learn_and_plot_causal_graph_from_csv(
    csv_path: str,
    selected_cols: list[str],
    weight_threshold: float | None = None,
    figsize: tuple[int, int] = (14, 14),
    layout_seed: int = 56
) -> Image:
    # Load and preprocess
    import os
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return Image("Error: CSV file not found")
    except pd.errors.EmptyDataError:
        logging.error("Error: CSV file is empty")
        return Image("Error: CSV file is empty")
    except pd.errors.ParserError:
        logging.error("Error: Failed to parse CSV file")
        return Image("Error: Failed to parse CSV file")

    raw = df[selected_cols].copy()
    num_df = (
        raw
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
        .loc[:, lambda d: d.var() > 0]
    )

    # Fit DirectLiNGAM
    model = DirectLiNGAM()
    model.fit(num_df.values)
    B = model.adjacency_matrix_
    vars_ = num_df.columns.tolist()

    # Determine threshold if not provided
    nonzero = np.abs(B[np.nonzero(B)])
    if weight_threshold is None and nonzero.size:
        weight_threshold = np.percentile(nonzero, 10)
    weight_threshold = weight_threshold or 0.0

    # Build directed graph
    G = nx.DiGraph()
    G.add_nodes_from(vars_)
    for i, cause in enumerate(vars_):
        for j, effect in enumerate(vars_):
            w = B[i, j]
            if abs(w) > weight_threshold:
                G.add_edge(cause, effect, weight=w)

    # Plot into an in‑memory buffer
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=1500, arrowsize=20)
    labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    ax.set_title("Learned Causal Graph (DirectLiNGAM)")

    # Convert image to base64-encoded string (JSON compatible)
    import io
    import base64
    
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    
    # Convert to base64 string
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Save locally for reference
    with open("server_debug_image.png", "wb") as f:
        f.write(img_bytes)
    
    # Return a dictionary with the base64-encoded image
    return {"image_base64": img_base64}

# 3) Run your server over SSE
if __name__ == "__main__":
    # SSE transport is the default and works well with the client
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8000
        # Using default endpoint configuration
    )