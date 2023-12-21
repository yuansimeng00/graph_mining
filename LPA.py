import random
import networkx as nx
import matplotlib.pyplot as plt
# 应该封装成类的形式
def lpa(G):
    '''
    异步更新方式
    G：图
    return：None
    通过改变节点的标签，最后通过标签来划分社区
    算法终止条件：迭代次数超过设定值
    '''
    max_iter_num = 0  # 迭代次数

    while max_iter_num < 10:
        max_iter_num += 1
        print('迭代次数', max_iter_num)

        for node in G:
            count = {}  # 记录邻居节点及其标签

            for nbr in G.neighbors(node):  # 获取node的邻居节点
                label = G.nodes[nbr]['labels']# 获得邻居节点的标签
                # 获取邻居节点的标签在字典 count 中对应的值，如果该标签在字典中不存在，则将其添加到字典中，并设置默认值为0。然后再将这个值加上1，表示计数加1
                count[label] = count.setdefault(label, 0) + 1
            # 找到出现次数最多的标签 x[-1] 表示取键值对的最后一个元素，即出现次数。-x[-1] 表示对出现次数进行取反，即按照降序排列。
            count_items = sorted(count.items(), key=lambda x: -x[-1]) # (节点、标签)
            best_labels = [k for k, v in count_items if v == count_items[0][1]] #降序排列，所以count_items[0][1]是标签出现次数最大值
            # 当多个标签最大技术值相同时随机选取一个标签
            label = random.sample(best_labels, 1)[0]  # 返回的是列表，所以需要[0]
            G.nodes[node]['labels'] = label  # 更新标签


def draw_picture(G):
    # 画图,获取节点的标签值。然后将标签值转换为浮点型,
    node_color = [float(G.nodes[v]['labels']) for v in G]
    pos = nx.spring_layout(G)  # 节点的布局为spring型
    plt.figure(figsize=(8, 6))  # 图片大小
    nx.draw_networkx(G, pos=pos,node_color=node_color)
    plt.show()


if __name__ == "__main__":
    G = nx.karate_club_graph()  # 空手道数据集
    # 给节点添加标签
    for node in G:
        G.add_node(node, labels=node)  # 给图G中的节点添加labels属性
    lpa(G)
    com = set([G.nodes[node]['labels'] for node in G])
    print('社区数量', len(com))
    draw_picture(G)

