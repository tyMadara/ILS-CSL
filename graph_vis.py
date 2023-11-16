import pydot
from rich import print as rprint

from utils import *


class GraphVis:
    def __init__(self, node_list: list = None, edge_list: list = None):
        self.correct2style = {"true": "solid", "false": "dashed"}
        self.prior2color = {"exist": "red", "frob": "blue", "normal": "black"}
        self.nodes = node_list
        self.edges = edge_list
        self.i2p = {i: p for i, p in enumerate(node_list)}
        self.p2i = {p: i for i, p in enumerate(node_list)}

    def init_aux_edges(self, exist_edges, forb_edges, true_edges):
        self.exist_edges = exist_edges
        self.forb_edges = forb_edges
        self.true_edges = true_edges

    def summary_info(self):
        info_list = []
        for edge in self.edges:
            correct, prior = self._edge_type(edge)
            edge_info = {"edge": edge, "correct": correct, "prior": prior}
            info_list.append(edge_info)
        return info_list

    def construct_digraph(self, edge_info_list):
        graph = pydot.Dot(graph_type='digraph')
        for node in self.nodes:
            graph.add_node(pydot.Node(node))
        for edge_info in edge_info_list:
            edge = edge_info["edge"]
            color = self.prior2color[edge_info["prior"]]
            style = self.correct2style[edge_info["correct"]]
            graph.add_edge(pydot.Edge(
                edge[0], edge[1], color=color, style=style))
        return graph

    def _edge_type(self, edge):
        if edge in self.true_edges:  # edge in the true dag
            correct = "true"
        else:  # edge not in the true dag
            correct = "false"
        reversed_edge = [edge[1], edge[0]]
        if reversed_edge in self.exist_edges or edge in self.forb_edges:  # B or C
            if reversed_edge in self.exist_edges and reversed_edge in self.true_edges:
                prior = "exist"
            elif edge in self.forb_edges and edge not in self.true_edges:
                prior = "frob"
            else:
                prior = "normal"
        else:
            prior = "normal"
        return correct, prior

    def visualize(self, png_path: str = 'graph.png', svg_path: str = None):
        edge_info_list = self.summary_info()
        graph = self.construct_digraph(edge_info_list)
        graph.write_png(png_path)
        if svg_path:
            graph.write_svg(svg_path)
        rprint(
            f"📊 [bold yellow]Graphviz[/bold yellow] saved to [italic blue]{png_path}[/italic blue] and [italic blue]{svg_path}[/italic blue]")
        # edge_info string to nubmer
        for edge_info in edge_info_list:
            edge_info["edge"] = [self.p2i[p] for p in edge_info["edge"]]
        return edge_info_list


def matrix2edge(ev_dag, i2p, string=True):
    """
    output the edges of ev_dag in the format of [A, B],
    where A and B are the names of the nodes
    """
    candidate_edges = []
    for i in range(ev_dag.shape[0]):
        for j in range(ev_dag.shape[1]):
            if ev_dag[i, j] == 1:
                candidate_edges.append([i, j])
    if string:
        candidate_edges = [[i2p[e[0]], i2p[e[1]]]
                           for e in candidate_edges]
    return candidate_edges


def revisualize():
    for dataset in dataset_list:
        mapping_path = f"BN_structure/mappings/{dataset}.mapping"
        true_dag_path = f"BN_structure/{dataset}_graph.txt"
        mapping = np.loadtxt(mapping_path, dtype=str)
        i2p = {i: p for i, p in enumerate(mapping)}
        true_dag = np.loadtxt(true_dag_path, dtype=int)
        true_edges = matrix2edge(true_dag, i2p)
        GT_png_path = f"img/dag_true/{dataset}.png"
        GT_svg_path = f"img/dag_true/{dataset}.svg"
        graph = GraphVis(mapping, true_edges)
        graph.init_aux_edges([], [], true_edges)
        graph.visualize(GT_png_path, GT_svg_path)
        for datasize_index in [0, 1]:
            size = dataset2size[datasize_index][dataset]
            for data_index in data_index_list:
                for alg, score in alg_score_list:
                    exp_name = f"{dataset}-{size}-{data_index}-{alg}-{score}"
                    prior_iter_path = f"out/prior-iter/{exp_name}.json"
                    data_prior_iter_raw = read_json(prior_iter_path)
                    if len(data_prior_iter_raw) == 0:
                        rprint(
                            f"[bold red]Skip[/bold red] [italic blue] {exp_name} [/italic blue] due to no prior iter data")
                        continue
                    for iter, data in enumerate(data_prior_iter_raw):
                        edges = [[i2p[e["edge"][0]], i2p[e["edge"][1]]]
                                 for e in data["edges"]]
                        exist_edges = [[i2p[e[0]], i2p[e[1]]]
                                       for e in data["exist_edges"]]
                        forb_edges = [[i2p[e[0]], i2p[e[1]]]
                                      for e in data["forb_edges"]]
                        rprint(f"{exp_name}-iter{iter+1}")
                        rprint(f"exist_edges: {exist_edges}")
                        rprint(f"forb_edges: {forb_edges}")
                        png_path = f"img/graph/{exp_name}-iter{iter+1}.png"
                        svg_path = f"img/graph-svg/{exp_name}-iter{iter+1}.svg"
                        graph = GraphVis(mapping, edges)
                        graph.init_aux_edges(
                            exist_edges, forb_edges, true_edges)
                        graph.visualize(png_path, svg_path)


if __name__ == "__main__":
    result_dir = "out/metrics"
    dataset2size = [
        {"asia": 250, "child": 500, "insurance": 500, "alarm": 1000,
            "cancer": 250, "mildew": 8000, "water": 1000, "barley": 2000},
        {"asia": 1000, "child": 2000, "insurance": 2000, "alarm": 4000,
            "cancer": 1000, "mildew": 32000, "water": 4000, "barley": 8000}
    ]
    data_index_list = [1, 2, 3, 4, 5, 6]
    alg_score_list = [["CaMML", "mml"],  ["HC", "bdeu"], ["softHC", "bdeu"], ["hardMINOBSx", "bdeu"], ["softMINOBSx", "bdeu"],
                      ["HC", "bic"], ["softHC", "bic"], ["hardMINOBSx", "bic"], ["softMINOBSx", "bic"]]

    dataset_list = ["cancer", "asia", "child",
                    "insurance", "alarm", "mildew", "water", "barley"]
    revisualize()
