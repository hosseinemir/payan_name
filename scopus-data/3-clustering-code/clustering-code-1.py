import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict
from tqdm import tqdm

# =========================
# CONFIG
# =========================
EDGE_FILE = "co_occurrence_edges.csv"
CLUSTER_WORD_FILE = "word_clusters.csv"
CLUSTER_TOP_FILE = "cluster_top30_words.csv"

MIN_EDGE_WEIGHT = 2   # حذف نویز، قابل تغییر

# =========================
# LOAD DATA WITH PROGRESS
# =========================
print("Loading edge data with progress...")
df = pd.read_csv(EDGE_FILE)
df = df[df["weight"] >= MIN_EDGE_WEIGHT]
print(f"Total edges after filtering: {len(df)}")

# =========================
# BUILD GRAPH
# =========================
print("Building graph...")
G = nx.Graph()
for _, row in tqdm(df.iterrows(), total=len(df)):
    G.add_edge(row["source"], row["target"], weight=row["weight"])

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================
# INTERACTIVE RESOLUTION
# =========================
while True:
    try:
        res = float(input("Enter Louvain resolution (e.g., 1.0, 1.2, 1.5): "))
    except ValueError:
        print("Invalid input, enter a number like 1.0")
        continue

    print("Running Louvain clustering...")
    partition = community_louvain.best_partition(G, weight="weight", resolution=res)

    num_clusters = len(set(partition.values()))
    print(f"\nNumber of clusters obtained: {num_clusters}")

    confirm = input("Do you want to proceed with this clustering? (y/n): ").strip().lower()
    if confirm == "y":
        break
    else:
        print("You can enter a new resolution value.\n")

# =========================
# SAVE WORD → CLUSTER
# =========================
print("Saving word-cluster mapping...")
pd.DataFrame(
    [(w, c) for w, c in partition.items()],
    columns=["word", "cluster_id"]
).sort_values("cluster_id").to_csv(CLUSTER_WORD_FILE, index=False)

# =========================
# CALCULATE WEIGHTED DEGREE PER NODE
# =========================
print("Calculating weighted degree per node...")
cluster_nodes = defaultdict(list)
for node, cid in tqdm(partition.items()):
    wd = G.degree(node, weight="weight")
    cluster_nodes[cid].append((node, wd))

# =========================
# TOP 30 WORDS PER CLUSTER
# =========================
print("Extracting top 30 words per cluster and saving output...")
rows = []
for cid in sorted(cluster_nodes.keys()):  # مرتب‌سازی صعودی cluster_id
    nodes = cluster_nodes[cid]
    top_words = sorted(nodes, key=lambda x: x[1], reverse=True)[:30]
    words_only = [f"\"{w}\"" for w, _ in top_words]
    members_str = "(" + ",".join(words_only) + ")"
    rows.append({"cluster_id": cid, "members": members_str})

top_df = pd.DataFrame(rows)
top_df.to_csv(CLUSTER_TOP_FILE, index=False)

print("\nClustering completed successfully!")
print(f"Files saved:\n- {CLUSTER_WORD_FILE}\n- {CLUSTER_TOP_FILE}")
