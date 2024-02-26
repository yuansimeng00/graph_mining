"""Implementation of gSpan."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools
import time

from .graph import AUTO_EDGE_ID
from .graph import Graph
from .graph import VACANT_GRAPH_ID
from .graph import VACANT_VERTEX_LABEL

import pandas as pd


def record_timestamp(func):
    """Record timestamp before and after call of `func`."""
    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()
    return deco


class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
        )


class DFScode(list):
    """DFScode is a list of DFSedge."""

    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list()

    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
        )

    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not inplemented yet.')

    def build_rmpath(self):
        """Build right most path.
        对于每个元素，获取其起始节点frm和目标节点to。如果frm小于to，并且old_frm为空或者to等于old_frm，说明当前边是右边最长路径的一部分
        将当前元素的索引i添加到右边最长路径列表self.rmpath中，并更新old_frm为当前边的起始节点frm。
        最后，返回构建好的右边最长路径self.rmpath作为函数的结果。
        """
        self.rmpath = list()
        old_frm = None
        for i in range(len(self) - 1, -1, -1): # 倒序输出self中的元素
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):
                self.rmpath.append(i)
                old_frm = frm
        return self

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))


class PDFS(object):
    """PDFS class."""

    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev


class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self


class History(object):
    """History class."""

    def __init__(self, g, pdfs):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
        if pdfs is None:
            return
        while pdfs:
            e = pdfs.edge
            self.edges.append(e)
            (self.vertices_used[e.frm],
                self.vertices_used[e.to],
                self.edges_used[e.eid]) = 1, 1, 1

            pdfs = pdfs.prev
        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history.检查具有特定eid的边是否存在于历史记录中"""
        return self.edges_used[eid] == 1


class gSpan(object):
    """`gSpan` algorithm."""

    def __init__(self,
                 database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize gSpan instance."""
        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        # 包含顶点数小于等于上限，大于等于2的子图。
        self._frequent_subgraphs = list()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self.timestamps = dict()
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'run']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('Mine:\t{} s'.format(
            time_deltas['run'] - time_deltas['_read_graphs']))
        print('Total:\t{} s'.format(time_deltas['run']))

        return self

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f.readlines()]#读入数据
            tgraph, graph_cnt = None, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')
                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None
                    if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                        break
                    tgraph = Graph(graph_cnt,
                                   is_undirected=self._is_undirected,
                                   eid_auto_increment=True)
                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2])
                elif cols[0] == 'e':
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3]) # 边ID，源顶点ID，目标顶点ID，边标签
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

    @record_timestamp
    def _generate_1edge_frequent_subgraphs(self): # 生成频繁的单顶点子图
        vlb_counter = collections.Counter() # 用于计算每个顶点标签的出现次数
        vevlb_counter = collections.Counter() # 用于计算每个顶点标签对和边标签的出现次数。
        vlb_counted = set()
        vevlb_counted = set()
        for g in self.graphs.values(): # 对于每个图g中的每个顶点v
            for v in g.vertices.values():
                if (g.gid, v.vlb) not in vlb_counted: # 如果顶点标签v.vlb还没有被计数过，则将其添加到vlb_counter中，并将其计数设置为1。
                    vlb_counter[v.vlb] += 1
                vlb_counted.add((g.gid, v.vlb))
                for to, e in v.edges.items(): # 对于顶点v的每条边e，获取边的目标顶点标签g.vertices[to].vlb。
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    if self._is_undirected and vlb1 > vlb2: # 如果图是无向图且顶点标签v.vlb大于顶点标签g.vertices[to].vlb，则交换两个顶点标签的位置。
                        vlb1, vlb2 = vlb2, vlb1
                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter: # 如果顶点标签对(vlb1, e.elb, vlb2)还没有被计数过，则将其添加到vevlb_counter中，并将其计数设置为1。
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))
        # add frequent vertices.
        for vlb, cnt in vlb_counter.items(): # 对于每个顶点标签vlb和计数cnt
            if cnt >= self._min_support: # 如果计数cnt大于等于最小支持度self._min_support
                g = Graph(gid=next(self._counter),
                          is_undirected=self._is_undirected)
                g.add_vertex(0, vlb) # 则创建一个新的图g，添加一个顶点(0, vlb)
                self._frequent_size1_subgraphs.append(g) # 将其添加到频繁单顶点子图列表self._frequent_size1_subgraphs中
                if self._min_num_vertices <= 1: # 并调用_report_size1函数报告该频繁子图。
                    self._report_size1(g, support=cnt)
            else:
                continue
        if self._min_num_vertices > 1:  # 如果最小顶点数要求大于1，则将计数器self._counter重置为一个无限迭代器。
            self._counter = itertools.count()
    @record_timestamp
    def run(self):
        """Run the gSpan algorithm."""
        self._read_graphs() # 读图数据
        self._generate_1edge_frequent_subgraphs() # 将频度与最小支持度数做比较，移除不频繁的边和点。
        if self._max_num_vertices < 2:
            return
        root = collections.defaultdict(Projected) # 存储以(vlb1, e.elb, vlb2)为键和Projected对象为值的映射
        for gid, g in self.graphs.items(): # 使用两层循环遍历self.graphs字典中的每个图对象g和其中的每个顶点对象v。
            for vid, v in g.vertices.items(): # 对于每个顶点对象，获取与其相邻的边。
                edges = self._get_forward_root_edges(g, vid)
                for e in edges: # 并将其添加到root字典中以(vlb1, e.elb, vlb2)为键，PDFS(gid, e, None)对象为值的映射中
                    root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                        PDFS(gid, e, None)
                    )

        for vevlb, projected in root.items(): # 使用root.items()遍历root字典中的每个键值对，其中键为vevlb，值为projected。
            self._DFScode.append(DFSedge(0, 1, vevlb)) # 对于每个键值对，将(0, 1, vevlb)作为参数创建一个DFSedge对象，并将其添加到self._DFScode列表中。
            self._subgraph_mining(projected)
            self._DFScode.pop() # 将self._DFScode列表中的最后一个元素移除
            print(self._report_df)

    def _get_support(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _report_size1(self, g, support):
        g.display()
        print('\nSupport: {}'.format(support))
        print('\n-----------------\n')

    def _report(self, projected): # 报告发现的频繁子图
        self._frequent_subgraphs.append(copy.copy(self._DFScode)) # 将当前的DFS代码复制到频繁子图列表中。
        if self._DFScode.get_num_vertices() < self._min_num_vertices: # 如果DFS代码中的顶点数小于最小顶点数要求，则返回。
            return
        g = self._DFScode.to_graph(gid=next(self._counter),
                                   is_undirected=self._is_undirected) # 根据DFS代码生成一个图g，并将其显示为字符串。
        display_str = g.display()
        print('\nSupport: {}'.format(self._support)) # 打印出支持度。

        # Add some report info to pandas dataframe "self._report_df". 将一些报告信息添加到Pandas DataFrame self._report_df 中。
        self._report_df = self._report_df.append(
            pd.DataFrame(
                {
                    'support': [self._support],
                    'description': [display_str],
                    'num_vert': self._DFScode.get_num_vertices()
                },
                index=[int(repr(self._counter)[6:-1])]
            )
        )
        if self._visualize: # 如果设置了可视化选项，将图g绘制出来。
            g.plot()
        if self._where: # 如果设置了_where选项，打印出projected中的每个投影的图ID。
            print('where: {}'.format(list(set([p.gid for p in projected]))))
        print('\n-----------------\n')

    def _get_forward_root_edges(self, g, frm): # 给定一个图g和一个顶点ID frm，该函数返回从顶点frm出发的所有前向根边。
        result = []
        v_frm = g.vertices[frm]
        for to, e in v_frm.edges.items():
            if (not self._is_undirected) or v_frm.vlb <= g.vertices[to].vlb:
                result.append(e)
        return result

    def _get_backward_edge(self, g, e1, e2, history): # 用于从图g中获取与边e1和边e2相关的后向边。
        if self._is_undirected and e1 == e2: # 如果图是无向图且边e1等于边e2，则返回None。
            return None
        for to, e in g.vertices[e2.to].edges.items(): # 对于图中与边e2的目标顶点相连的每个顶点to和边e
            if history.has_edge(e.eid) or e.to != e1.frm: # 如果历史记录中已经存在该边或者该边的目标顶点不等于边e1的源顶点，则继续到下一次循环。
                continue
            # if reture here, then self._DFScodep[0] != dfs_code_min[0]
            # should be checked in _is_min(). or:
            if self._is_undirected:
                if e1.elb < e.elb or (
                        e1.elb == e.elb and
                        g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
                    return e # 如果边e1的边标签小于边e的边标签或者边e1的边标签等于边e的边标签且边e1的目标顶点的标签小于等于边e2的目标顶点的标签，则返回边e。
            else:
                if g.vertices[e1.frm].vlb < g.vertices[e2.to].vlb or (
                        g.vertices[e1.frm].vlb == g.vertices[e2.to].vlb and
                        e1.elb <= e.elb):
                    return e # 如果边e1的源顶点的标签小于边e2的目标顶点的标签或者边e1的源顶点的标签等于边e2的目标顶点的标签且边e1的边标签小于等于边e的边标签，则返回边e。
            # if e1.elb < e.elb or (e1.elb == e.elb and
            #     g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
            #     return e
        return None

    def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history): # 用于从图g中获取与删除边rm_edge和最小顶点标签min_vlb相关的纯前向边。
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items(): # 对于图中与删除边的目标顶点相连的每个顶点to和边e
            if min_vlb <= g.vertices[e.to].vlb and (not history.has_vertex(e.to)): # 如果最小顶点标签小于等于边的目标顶点的标签且历史记录中不存在该顶点，则将该边添加到结果列表中。
                result.append(e)
        return result

    def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history): # 用于从图g中获取与删除的边rm_edge和最小顶点标签min_vlb相关的前向边。
        result = [] # 创建一个空列表result用于存储结果
        to_vlb = g.vertices[rm_edge.to].vlb # 获取拓展根边终点的点标签。
        for to, e in g.vertices[rm_edge.frm].edges.items(): # 获取拓展根边的最右拓展边的终点以及边标签
            new_to_vlb = g.vertices[to].vlb # 获取拓展根边的最右拓展边的终点的点标签。
            if (rm_edge.to == e.to or min_vlb > new_to_vlb or history.has_vertex(e.to)): # 如果拓展根边的终点=拓展边的终点/拓展边的终点标签小于最小边标签/历史对象中已经存在拓展边的终点。
                continue
            if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and
                                       to_vlb <= new_to_vlb):
                result.append(e) # 如果拓展根边的边标签小于最右拓展边的边标签/就算边标签相等，但拓展边终点标签>=拓展根边的重点标签，这样保证添加拓展边标签的DFS编码不能比你之前的小
        return result

    def _is_min(self): # 检查当前DFS编码是否最小
        if self._verbose: # 用来在DFS代码搜索树中进行深度优先搜索时，打印每个访问的DFS代码
            print('is_min: checking {}'.format(self._DFScode))
        if len(self._DFScode) == 1:
            return True
        g = self._DFScode.to_graph(gid=VACANT_GRAPH_ID,
                                   is_undirected=self._is_undirected)
        dfs_code_min = DFScode() # 初始化一个DFScode对象dfs_code_min来存储投影图
        root = collections.defaultdict(Projected) # 初始化一个defaultdict对象root来存储投影图
        for vid, v in g.vertices.items(): # 对图中的每个顶点进行迭代，并使用_get_forward_root_edges()方法获取前向根边
            edges = self._get_forward_root_edges(g, vid)
            for e in edges: # 然后，将拓展图追加到root字典中，以顶点标签、边标签和目标顶点标签作为键。
                root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                    PDFS(g.gid, e, None))
        min_vevlb = min(root.keys())
        dfs_code_min.append(DFSedge(0, 1, min_vevlb)) # 计算最小顶点-边-顶点标签（vevlb），即在root字典中的最小键。然后，创建一个DFS边，源顶点为0，目标顶点为1，边标签为vevlb的最小值。将该DFS边追加到dfs_code_min对象中
        # No need to check if is min code because of pruning in get_*_edge*.

        def project_is_min(projected): # 检查拓展图是否是最小的。该函数从dfs_code_min对象构建最右路径（rmpath）。
            dfs_code_min.build_rmpath()
            rmpath = dfs_code_min.rmpath
            min_vlb = dfs_code_min[0].vevlb[0] # 并初始化最小顶点标签（min_vlb）
            maxtoc = dfs_code_min[rmpath[0]].to # 初始化最大目标顶点（maxtoc）的变量

            backward_root = collections.defaultdict(Projected) # 创建一个backward_root defaultdict来存储反向边的投影图。代码反向遍历rmpath中的边，并对于每个投影图，使用_get_backward_edge()方法检查是否存在反向边。
            flag, newto = False, 0,
            end = 0 if self._is_undirected else -1
            for i in range(len(rmpath) - 1, end, -1):
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    e = self._get_backward_edge(g,
                                                history.edges[rmpath[i]],
                                                history.edges[rmpath[0]],
                                                history)
                    if e is not None: #如果找到反向边，则将投影图追加到backward_root字典中，并更新新的目标顶点（newto），并设置标志为True
                        backward_root[e.elb].append(PDFS(g.gid, e, p))
                        newto = dfs_code_min[rmpath[i]].frm
                        flag = True
            if flag: # 如果找到反向边，则计算最小边标签（backward_min_elb），即在backward_root字典中的最小键。
                backward_min_elb = min(backward_root.keys())
                dfs_code_min.append(DFSedge(
                    maxtoc, newto,
                    (VACANT_VERTEX_LABEL,
                     backward_min_elb,
                     VACANT_VERTEX_LABEL)
                )) # 然后，创建一个新的DFS边，源顶点为maxtoc，目标顶点为newto，顶点标签和边标签分别为backward_root中的最小值。将该DFS边追加到dfs_code_min对象中。
                idx = len(dfs_code_min) - 1
                if self._DFScode[idx] != dfs_code_min[idx]: # 代码检查dfs_code_min中的最后一条DFS边是否与当前DFS编码中的相应DFS边。
                    return False # 如果它们不相等，则表示当前DFS编码不是最小的
                return project_is_min(backward_root[backward_min_elb])

            forward_root = collections.defaultdict(Projected)
            flag, newfrm = False, 0 # 用于存储前向边的投影图。然后，初始化flag和newfrm为False和0。
            for p in projected: # 对于每个投影图p，通过创建History对象和使用_get_forward_pure_edges函数获取与rmpath[0]相关的前向边。
                history = History(g, p)
                edges = self._get_forward_pure_edges(g,
                                                     history.edges[rmpath[0]],
                                                     min_vlb,
                                                     history)
                if len(edges) > 0: # 。如果存在前向边，则将flag设置为True，并将newfrm设置为maxtoc。然后，将这些前向边添加到forward_root中
                    flag = True
                    newfrm = maxtoc
                    for e in edges:
                        forward_root[
                            (e.elb, g.vertices[e.to].vlb)
                        ].append(PDFS(g.gid, e, p))
            for rmpath_i in rmpath: # 对于每个rmpath_i（删除路径中的路径）。
                if flag: # 如果flag为True，则跳出循环。
                    break
                for p in projected: # 对于每个投影图p，通过创建History对象和使用_get_forward_rmpath_edges函数获取与rmpath_i相关的前向边
                    history = History(g, p)
                    edges = self._get_forward_rmpath_edges(g,
                                                           history.edges[
                                                               rmpath_i],
                                                           min_vlb,
                                                           history)
                    if len(edges) > 0: # 如果存在前向边，则将flag设置为True，并将newfrm设置为dfs_code_min[rmpath_i].frm。然后，将这些前向边添加到forward_root中
                        flag = True
                        newfrm = dfs_code_min[rmpath_i].frm
                        for e in edges:
                            forward_root[
                                (e.elb, g.vertices[e.to].vlb)
                            ].append(PDFS(g.gid, e, p))

            if not flag: # 如果flag为False，则返回True，表示前向边的投影图是最小的。
                return True

            forward_min_evlb = min(forward_root.keys()) # 找到forward_root中最小的键，并使用该键创建一个新的DFS边，源顶点为newfrm，。
            dfs_code_min.append(DFSedge(
                newfrm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
            ) # 目标顶点为maxtoc + 1，边标签为最小键的值。将该DFS边追加到dfs_code_min中
            idx = len(dfs_code_min) - 1 #检查更新后的dfs_code_min是否与当前DFS编码相等。
            if self._DFScode[idx] != dfs_code_min[idx]: # 如果不相等，则返回False，表示当前DFS编码不是最小的。
                return False
            return project_is_min(forward_root[forward_min_evlb]) #否则，递归调用project_is_min函数来进一步检查前向边的投影图是否是最小的。

        res = project_is_min(root[min_vevlb]) # 返回project_is_min(root[min_vevlb])的结果。
        return res

    def _subgraph_mining(self, projected):
        self._support = self._get_support(projected) # 计算当前基础子图的支持度
        if self._support < self._min_support: # 与最小支持度进行比较。如果支持度小于最小支持度，则直接返回，不再进行后续挖掘。
            return
        if not self._is_min(): # 判断当前基础子图是否是最小的频繁子图。如果不是最小的频繁子图，则直接返回，不再进行后续挖掘
            return
        self._report(projected) # 将当前基础子图报告为频繁子图。

        num_vertices = self._DFScode.get_num_vertices() # 获取当前基础子图的顶点数
        self._DFScode.build_rmpath() # 并调用build_rmpath函数记录最右路径（rmpath）
        rmpath = self._DFScode.rmpath
        maxtoc = self._DFScode[rmpath[0]].to # DFS编码中的最大目标顶点
        min_vlb = self._DFScode[0].vevlb[0] # 点标签

        forward_root = collections.defaultdict(Projected) # 存储前向扩展的结果。
        backward_root = collections.defaultdict(Projected) # 存储后向扩展的结果。
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p) # 创建一个History对象，用于记录历史边和顶点
            # backward 代码通过遍历最右路径中的边，使用_get_backward_edge函数获取每个边的后向边，并将其添加到backward_root中。
            for rmpath_i in rmpath[::-1]:
                e = self._get_backward_edge(g,
                                            history.edges[rmpath_i],
                                            history.edges[rmpath[0]],
                                            history) # 由于rmpath倒序输出了，所以rmpath[0]是最后的一个点
                if e is not None:   # e为None表示无后向边
                    backward_root[
                        (self._DFScode[rmpath_i].frm, e.elb)
                    ].append(PDFS(g.gid, e, p))
            # pure forward 在纯前向扩展部分，如果当前基础子图的顶点数小于最大顶点数限制，则使用_get_forward_pure_edges函数获取纯前向边，并将其添加到forward_root中。
            if num_vertices >= self._max_num_vertices:
                continue
            edges = self._get_forward_pure_edges(g,
                                                 history.edges[rmpath[0]],
                                                 min_vlb,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))
            # rmpath forward 在最右路径前向扩展部分，代码通过遍历最右路径中的边，使用_get_forward_rmpath_edges函数获取每个边的最右路径前向边，并将其添加到forward_root中。
            for rmpath_i in rmpath:
                edges = self._get_forward_rmpath_edges(g,
                                                       history.edges[rmpath_i],
                                                       min_vlb,
                                                       history)
                for e in edges:
                    forward_root[
                        (self._DFScode[rmpath_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].append(PDFS(g.gid, e, p))

        # backward 对于每个(to, elb)对，创建一个新的DFS边，源顶点为maxtoc，目标顶点为to，边标签为(VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL)。
        for to, elb in backward_root:
            self._DFScode.append(DFSedge(
                maxtoc, to,
                (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
            )
            self._subgraph_mining(backward_root[(to, elb)]) # 将该DFS边追加到DFScode中。然后，递归调用_subgraph_mining函数来继续在后向边的投影图上进行子图挖掘
            self._DFScode.pop() # 最后，将刚刚添加的DFS边从DFScode中弹出
        # forward
        # No need to check if num_vertices >= self._max_num_vertices.
        # Because forward_root has no element.
        for frm, elb, vlb2 in forward_root: # 对于每个(frm, elb, vlb2)三元组，创建一个新的DFS边，
            self._DFScode.append(DFSedge(
                frm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, elb, vlb2))
            ) # 为什么是frm,maxtoc + 1?  其实在forward_root中frm对应可能是frm,也可能是maxtoc
            self._subgraph_mining(forward_root[(frm, elb, vlb2)])
            self._DFScode.pop()

        return self
