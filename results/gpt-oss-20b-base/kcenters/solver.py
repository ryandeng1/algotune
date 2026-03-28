from collections import defaultdict, deque
from heapq import heappush, heappop
import math
from bisect import bisect_right

INF = float('inf')


def dijkstra(graph, start):
    """Return dict of shortest distances from start to all reachable nodes."""
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heappop(heap)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heappush(heap, (nd, v))
    return dist


class Distances:
    """Pre‑computed all‑pairs shortest‑path distances."""

    def __init__(self, graph):
        self._dist = {}
        for u in graph:
            self._dist[u] = dijkstra(graph, u)

    def dist(self, u, v):
        return self._dist[u].get(v, INF)

    def max_dist(self, centers):
        best = -INF
        for v in self._dist:
            min_d = min(self.dist(c, v) for c in centers)
            if min_d > best:
                best = min_d
        return best

    def vertices_in_range(self, u, limit):
        return [v for v, d in self._dist[u].items() if d <= limit]

    def sorted_dists(self):
        arr = []
        for d in self._dist.values():
            arr.extend(d.values())
        return sorted(arr)


def solve(problem):
    """Solve the k‑center problem using a deterministic greedy–then‑refine method."""
    G_dict, k = problem
    # Build graph: adjacency list, undirected
    graph = defaultdict(list)
    for u, adj in G_dict.items():
        for v, w in adj.items():
            graph[u].append((v, w))
            graph[v].append((u, w))

    if k == 0:
        return []

    distances = Distances(graph)

    # --- Greedy heuristic: farthest‑point algorithm ---
    nodes = set(graph)
    # choose first center with smallest maximum distance to all nodes
    first = min(nodes, key=lambda c: max(distances.dist(c, u) for u in nodes))
    centers = [first]
    nodes.remove(first)

    while len(centers) < k and nodes:
        # pick node farthest from current centers
        u = max(nodes, key=lambda v: min(distances.dist(c, v) for c in centers))
        centers.append(u)
        nodes.remove(u)

    # --- Refinement: try to reduce maximum distance iteratively ---
    cur_obj = distances.max_dist(centers)
    sorted_distances = distances.sorted_dists()
    idx = bisect_right(sorted_distances, cur_obj)

    # iterate over candidate radii smaller than current objective
    for rad in reversed(sorted_distances[:idx-1]):
        # set cover with radius rad
        cover = set()
        un_covered = set(graph)
        while un_covered:
            # pick center covering most uncovered nodes
            best_node = max(un_covered, key=lambda v: len(set(distances.vertices_in_range(v, rad)) & un_covered))
            cover.add(best_node)
            un_covered -= set(distances.vertices_in_range(best_node, rad))
            if len(cover) > k:
                break
        if len(cover) <= k:
            centers = sorted(cover)[:k]
            cur_obj = rad
            sorted_distances = sorted_distances[:bisect_right(sorted_distances, cur_obj)]

    return centers