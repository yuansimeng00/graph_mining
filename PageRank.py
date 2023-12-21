# import networkx as nx
# import matplotlib.pyplot as plt
# import random
#
# Graph = nx.DiGraph()
# Graph.add_nodes_from(range(0, 10))
# for i in range(20):
#     j = random.randint(0, 10)
#     k = random.randint(0, 10)
#     Graph.add_edge(k, j)
#
# # 绘图
# nx.draw(Graph, with_labels=True)
# plt.show()
"""PageRank计算"""
from numpy import *
import numpy as np
# 构造转移矩阵（输入矩阵列归一化），其实就是把每一列的值的和归一化得到转移概率矩阵
def graphMove(a):
    b = transpose(a) # 矩阵转置
    c = zeros(a.shape,dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c[i][j]=a[i][j]/(b[j].sum())
    print('转移概率矩阵:\n', c)
    return c
# 初始化pr值
def firstPr(c):
    pr = zeros((c.shape[0],1),dtype=float)
    for i in range(len(c[0])):
        pr[i]=float(1)/len(c[0])
    return pr
# 参数分别为阻尼系数、转移概率矩阵、初始向量、最大迭代次数、收敛约束因子
def PageRank(d,c,pr,max_itrs,con_rate):
    e = np.ones(shape=(len(c[0]), 1))
    # 修正转移概率矩阵
    A = d * c + ((1 - d) / len(c[0])) * np.dot(e, e.T)
    for i in range(max_itrs):
        old_PR = pr
        pr = np.dot(A, pr)
        # 如果所有网页PR值的前后误差 都小于 自定义的误差阈值，则停止迭代
        D = np.array([old - new for old, new in zip(old_PR, pr)])
        ret = [e < con_rate for e in D]
        if ret.count(True) == len(c[0]):
            print('迭代次数:%d, succeed PR:\n' % (i + 1), pr)
            break
    return pr
a = array([[0.0,1.0,0.0,1.0],
         [1.0,0.0,0.0,1.0],
         [1.0,1.0,1.0,0.0],
         [1.0,0.0,0.0,0.0]])
print('出链矩阵:\n', a)
c=graphMove(a)
pr=firstPr(c)
print('初始化PR值:\n', pr)

print('最终的PR值:\n',PageRank(0.85,c,pr,1000,0.0005))


