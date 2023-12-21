"""PageRank计算"""
import networkx as nx

def sybil_rank(graph, node):
    # 计算每个节点的信任分数
    trust_scores = nx.pagerank(graph, alpha=0.85)

    # 计算SybilRank分数
    sybil_scores = {}
    for neighbor in graph.neighbors(node):
        sybil_scores[neighbor] = trust_scores[neighbor] * graph[node][neighbor]['weight']

    return sybil_scores

# 创建一个有向图
G = nx.DiGraph()
# G.add_edge('A', 'B', weight=0.5)
# G.add_edge('A', 'C', weight=0.3)
# G.add_edge('B', 'D', weight=0.7)
# G.add_edge('C', 'E', weight=0.4)
# G.add_edge('D', 'E', weight=0.2)

G.add_edge('A', 'B', weight=0.5)
G.add_edge('A', 'C', weight=0.5)
G.add_edge('B', 'C', weight=0.33)
G.add_edge('B', 'E', weight=0.33)
G.add_edge('C', 'D', weight=0.33)
G.add_edge('D', 'E', weight=0.5)
G.add_edge('E', 'F', weight=0.33)
G.add_edge('F', 'G', weight=0.33)
G.add_edge('F', 'H', weight=0.33)
G.add_edge('G', 'H', weight=0.33)
G.add_edge('G', 'I', weight=0.33)
G.add_edge('H', 'I', weight=0.33)
# 计算SybilRank分数
node = 'C'
sybil_scores = sybil_rank(G, node)
print("SybilRank分数：", sybil_scores)


