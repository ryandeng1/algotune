import networkx as nx

class Solver:
    def solve(self, problem):
        n, g1_edges, g2_edges = problem["num_nodes"], problem["edges_g1"], problem["edges_g2"]

        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G1.add_edges_from(g1_edges)
        G2.add_edges_from(g2_edges)

        matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not matcher.is_isomorphic():
            return {"mapping": [-1] * n}

        iso = next(matcher.isomorphisms_iter())
        mapping = [iso[u] for u in range(n)]
        return {"mapping": mapping}