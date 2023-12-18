import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import print as rprint

from GPT import GPT
from config import load_config
from evaluation_DAG import MetricsDAG
from graph_vis import GraphVis
from perform import (CaMML_unit, hc_test, seperate_MINOBSx_unit)
from utils import *

from statics_plot import plot_metrics_with_iter
warnings.filterwarnings("ignore")


class Iteration_CSL:
    def __init__(self, dataset: str, model: str = "gpt-4", alg: str = "MINOBSx", data_index: int = 1, datasize_index: int = 0, score: str = 'bdeu', is_soft: bool = False) -> None:
        self.dataset = dataset
        self.LLM_model = model
        self.data_index = data_index
        self.algorithm = alg
        self.datasize_index = datasize_index
        self.score = score
        self.is_soft = is_soft

        self._load_path_info(dataset, model)
        self._load_dataset_info()
        self._load_files()

        self.data_size = self.dataset2size[self.datasize_index][self.dataset]
        self.score_filepath = f"data/score/{self.score}/{dataset}_{self.data_size}_{self.data_index}.txt"
        self.config = {"d": self.dataset, "s": self.data_size, "exist_edges": [], "forb_edges": [], 'conf': 0.99999, "order": [], "ancs": [], "forb_ancs": [
        ], "r": self.data_index, "true_dag": self.true_dag, "score": self.score, "palim": 3, "nopruning": True, "score_filepath": self.score_filepath, "is_soft": self.is_soft}
        self.exp_name = f"{self.dataset}-{self.data_size}-{self.data_index}-{self.algorithm}-{self.score}"

        self.metrics_path = f"out/metrics/{self.exp_name}.csv"
        self.prior_iter_path = f"out/prior-iter/{self.exp_name}.json"
        self.result_data = pd.DataFrame()
        self.gpt_queries = []
        self.prior_iter = []

    def _load_dataset_info(self):
        self.dataset_list = ("asia", "water", "mildew",
                             "insurance", "child", "barley", "alarm", "cancer")
        assert self.dataset in self.dataset_list
        self.BDeu_dataset_list = ("cancer", "asia", "child")
        self.dataset2size = [{"asia": 250, "child": 500, "insurance": 500, "alarm": 1000, "cancer": 250, "mildew": 8000, "water": 1000, "barley": 2000},
                             {"asia": 1000, "child": 2000, "insurance": 2000, "alarm": 4000, "cancer": 1000, "mildew": 32000, "water": 4000, "barley": 8000}]
        self.data_index_list = (1, 2, 3, 4, 5, 6)
        assert self.data_index in self.data_index_list
        self.algorithm_list = ("MINOBSx", "CaMML", "DP", "Astar",
                               "HC", "MMHC", "PGMINOBSx", "GroundTruth", "HC-BIC", "MINOBSx-BIC", "softMINOBSx", "hardMINOBSx", "softHC", "hardHC")
        assert self.algorithm in self.algorithm_list

    def _load_path_info(self, dataset, model):
        self.metrics_img_dir = f"img/iter"
        self.mapping_path = f"BN_structure/mappings/{dataset}.mapping"
        self.true_dag_path = f"BN_structure/{dataset}_graph.txt"
        self.history_path = f"out/gpt_history/{model}/{dataset}_history.json"
        self.history_path_toquery = f"out/gpt_history/{model}/{dataset}_history_toquery.json"
        self.wrong_history_path = f"out/gpt_history/{model}/{dataset}_wrong.json"
        self.summary_path = f"out/summary/{model}/{dataset}_summary.json"
        mkdir(f"out/gpt_history/{model}")
        mkdir(f"out/summary/{model}")
        if not os.path.exists(self.history_path):
            write_json([], self.history_path)

    def extract_answer(self, answer_sentence):
        """
        Extract the answer from the answer sentence of GPT output.
        """
        pattern = r"<Answer>(.*?)</Answer>"
        answer = re.findall(pattern, answer_sentence)
        if len(answer) == 0:
            # not found the best answer, try to extract the answer sentence
            answer = answer_sentence.split("\n")[-1].strip()
            # find answer in the answer sentence
            candidates = re.findall(r"[A-D]\.", answer)
            if len(candidates) == 1:
                return candidates[0][0]
            else:
                rprint(
                    f"[red]Failed[/red] to extract the answer. candidates: {candidates}")
                raise Exception("Failed to extract the answer.")
        elif len(answer) == 1:
            # found the best answer, select the first one
            answer = answer[0].strip()
            if len(answer) > 1 and answer[0] in ["A", "B", "C", "D"]:
                answer = answer[0]
        # successfully extract the answer
        if answer in ["A", "B", "C", "D"]:
            return answer
        # exacted the answer sentence
        elif len(answer) > 1 and answer[:2] in ["A.", "B.", "C.", "D."]:
            return answer[0]
        else:
            rprint("[red]Failed[/red] to extract the answer.")
            raise Exception("Failed to extract the answer.")

    def _load_files(self):
        self.start = read_txt("prompt/start.txt")
        self.background = read_txt("prompt/background.txt")
        self.question = read_txt("prompt/question.txt")

        self.domain = read_json(f"prompt/domain.json")[self.dataset]
        self.desc_dict = read_json(f"prompt/description/{self.dataset}.json")

        # load the mapping of the nodes
        self.mapping = np.loadtxt(self.mapping_path, dtype=str)
        self.i2p = {i: p for i, p in enumerate(self.mapping)}
        self.p2i = {p: i for i, p in enumerate(self.mapping)}

        # load the true DAG
        self.true_dag = np.loadtxt(self.true_dag_path, dtype=int)

        self.true_edges = self.matrix2edge(
            self.true_dag)  # convert the matrix to edges

    def matrix2edge(self, ev_dag, string=True):
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
            candidate_edges = [[self.i2p[e[0]], self.i2p[e[1]]]
                               for e in candidate_edges]
        # return [f"{c[0]} -> {c[1]}" for c in candidate_edges]
        return candidate_edges

    def edge2matrix(self, edges, string=False):
        """
        output the edges of ev_dag in the format of "A->B",
        where A and B are the names of the nodes
        """
        if string:
            edges = [[self.p2i[e[0]], self.p2i[e[1]]] for e in edges]
        ev_dag = np.zeros_like(self.true_dag)
        for edge in edges:
            ev_dag[edge[0], edge[1]] = 1
        return ev_dag

    def merge_prompt(self, edges: list):
        assert len(edges) == 2
        desc_1 = self.desc_dict[edges[0]]
        desc_2 = self.desc_dict[edges[1]]
        prompt = f"{self.start}{self.domain}\n{self.background}\n\n{edges[0]}: {desc_1}\n\n{edges[1]}: {desc_2}\n\n{self.question} {edges[0]} {edges[1]}"
        return prompt

    def causal_structure_learning(self):
        start_time = time.time()
        search_stats = {}
        if self.algorithm == 'CaMML':
            ev_dag = CaMML_unit("BI-CaMML", "", self.i2p, **self.config)
        elif self.algorithm == 'softMINOBSx':
            self.config["is_soft"] = True
            ev_dag = seperate_MINOBSx_unit(
                MINOBSx_base="minobsx", prefix=f"soft{self.score}", timeout=600, iter=10, prior_confidence=0.99999, **self.config)
        elif self.algorithm == 'hardMINOBSx':
            self.config["is_soft"] = False
            ev_dag = seperate_MINOBSx_unit(
                MINOBSx_base="minobsx", prefix=f"hard{self.score}", timeout=600, iter=10, **self.config)
        elif self.algorithm == 'softHC':
            self.config["is_soft"] = True
            ev_dag = hc_test(prefix=self.score, **self.config)
        elif self.algorithm == 'HC':
            ev_dag = hc_test(**self.config)
        elif self.algorithm == 'GroundTruth':
            ev_dag = self.true_dag
        else:
            raise Exception("Invalid CSL algorithm.")

        self.time = round(time.time() - start_time, 4)
        self.metrics = MetricsDAG(ev_dag, self.true_dag).metrics
        self.metrics["iter_num"] = self.iter_num
        self.metrics["time"] = self.time
        self.metrics["exist_mode"] = "edge"
        if 'time' in search_stats:
            self.metrics["calculate_time"] = search_stats['time']
        rprint(f"Metrics: {self.metrics}")

        candidate_edges = self.matrix2edge(
            ev_dag)  # convert the matrix to edges
        rprint(f"Candidate edges: {len(candidate_edges)}")
        return ev_dag, candidate_edges

    def iter_causal_structure_learning(self):
        continue_iter_flag = True
        self.iter_num = 0

        self.edge_prior = EdgePrior(self.true_dag, self.i2p, self.p2i)
        while continue_iter_flag:
            continue_iter_flag = False
            self.iter_num += 1
            rprint(
                f"🔁 Iteration: [bold purple]{self.iter_num:2d}[/bold purple]")
            rprint(
                f"Existing edges: {[[self.i2p[edge[0]],self.i2p[edge[1]]] for edge in self.config['exist_edges']]}")
            rprint(
                f"Forbidden edges: {[[self.i2p[edge[0]],self.i2p[edge[1]]] for edge in self.config['forb_edges']]}")

            ev_dag, ev_edges = self.causal_structure_learning()  # CSL
            # return

            if ev_dag.sum() == 0:  # if the DAG is empty, stop the iteration
                rprint(f"❌ [red]Empty[/red] DAG, stop the iteration.")
                break

            self.chatgpt = GPT(self.LLM_model)  # init the chatgpt
            self.history = read_json(self.history_path)  # load the history

            self.init_edge_prior()

            for edge in ev_edges:  # for each edge in candidate
                result = self.GPT_quiz(edge)  # query the edge
                if self.add_prior(result):  # update the prior
                    # if the prior has been updated, continue the iteration
                    continue_iter_flag = True

            summary = self.edge_prior.output_summary()
            write_json(summary, self.summary_path)

            # update config
            self.config["exist_edges"] = self.edge_prior.output_exist_edges()
            self.config["forb_edges"] = self.edge_prior.output_forb_edges()

            self.result_data = self.result_data.append(
                self.metrics, ignore_index=True)
            plot_metrics_with_iter(
                self.exp_name, self.result_data, img_dir=self.metrics_img_dir)
            # plot the graphviz and log the output edge info
            self.config["edges"] = self._plot_graphviz(ev_edges)
            self._debug_find_wrong()  # a debug function to output the wrong answers
            self.prior_iter.append(self.config.copy())
        self.result_data.to_csv(self.metrics_path, index=False)
        # delete true_dag from prior_iter
        for item in self.prior_iter:
            if "true_dag" in item:
                del item["true_dag"]
        write_json(self.prior_iter, self.prior_iter_path)

    def init_edge_prior(self):
        for item in self.history:
            if item["model"] == self.LLM_model:
                self.edge_prior.add_queried_edge(
                    item["start"], item["end"], string=True)

        for edge in self.config["exist_edges"]:
            self.edge_prior.add_exist_edge(edge[0], edge[1])
        for edge in self.config["forb_edges"]:
            self.edge_prior.add_forb_edge(edge[0], edge[1])

    def _del_history(self, edge):
        start, end = edge
        print(len(self.history))
        for i, item in enumerate(self.history):
            if item["start"] == start and item["end"] == end:
                self.history.pop(i)
                print(len(self.history))
        write_json(self.history, self.history_path)

    def _load_result_templates(self, edge, prompt=None):
        start, end = edge
        edge_str = f"{start}->{end}"
        if prompt is None:
            prompt = self.merge_prompt(edge)
        self.result_notsufficient = {"start": start, "end": end, "answer": "D", "model": "gpt-4", "input": prompt,
                                     "output": "The variables given are not sufficient. so the answer is <Answer>D</Answer>", "edge": edge_str}
        self.result_notadjacent = {"start": start, "end": end, "answer": "C", "model": "gpt-4", "input": prompt,
                                   "output": "The variables do not present in the adjacent stage, so the answer is <Answer>C</Answer>", "edge": edge_str}
        self.result_toquery = {"start": start, "end": end, "answer": "D", "model": "sim-gpt", "input": prompt,
                               "output": "", "edge": edge_str}

    def GPT_quiz(self, edge: list):
        """
        Let GPT to do the quiz of the edge existence.
        A cache technique: if the edge has been queried before, directly return the historical result.
        """
        start, end = edge
        edge_toprint = ColorP.edge(edge)
        GT = self.GT_bot(edge)

        prompt = self.merge_prompt(edge)
        self._load_result_templates(edge, prompt)

        if start == "" or end == "":
            self.history.append(self.result_notsufficient)
            canswer = ColorP.answer("D", GT)
            rprint(
                f"❌ [red]Insufficient[/red] description: {edge_toprint} {canswer}")
            return result

        if self.edge_prior.check_queried_edge(start, end, string=True):
            # Check if the edge has been queried before
            for history_item in self.history:
                if history_item["model"] == self.LLM_model:
                    # If the edge has been queried before, directly return historical result
                    if history_item["start"] == start and history_item["end"] == end:
                        canswer = ColorP.answer(
                            history_item["answer"], GT)
                        rprint(
                            f"🔍 [green]Already[/green] queried: {edge_toprint} {canswer}")
                        return history_item
            # if not found, check if the reverse edge has been queried
            for history_item in self.history:
                if history_item["model"] == self.LLM_model:
                    if history_item["start"] == end and history_item["end"] == start:
                        reversed_history_item = history_item.copy()
                        if history_item["answer"] == "A":
                            reversed_history_item.update(
                                {"start": start, "end": end, "answer": "B", "edge": f"{end}->{start}"})
                        elif history_item["answer"] == "B":
                            reversed_history_item.update(
                                {"start": start, "end": end, "answer": "A", "edge": f"{end}->{start}"})
                        elif history_item["answer"] in ("C", "D"):
                            reversed_history_item.update(
                                {"start": start, "end": end, "edge": f"{end}->{start}"})
                        canswer = ColorP.answer(
                            reversed_history_item["answer"], GT)
                        rprint(
                            f"🔍 [green]Already[/green] queried (reversed): {edge_toprint} {canswer}")
                        return reversed_history_item

        rprint(
            f"🔄 [yellow]Not yet[/yellow] queried: {edge_toprint} {ColorP.GT(GT)}")
        # if the edge has not been queried before, query it
        gpt_out = self.chatgpt.chatgpt_QA(prompt, quiet=True)
        answer = self.extract_answer(gpt_out["output"])
        result = {"start": start, "end": end, "answer": answer}
        result.update(gpt_out)
        result.update({"edge": f"{end}->{start}"})
        canswer = ColorP.answer(answer, GT)
        cmodel = ColorP.model(self.LLM_model)
        rprint(
            f"✅ [green]Successfully[/green] queried {cmodel}: {edge_toprint} {canswer}")
        self.history.append(result)  # update the history
        self.edge_prior.add_queried_edge(
            start, end, string=True)  # update the queried edges
        write_json(self.history, self.history_path)
        return result

    def GT_bot(self, edge, string=True):
        if string:
            edge = [self.p2i[edge[0]], self.p2i[edge[1]]]
        start, end = edge
        if self.true_dag[start][end] == 1:
            return "A"
        elif self.true_dag[end][start] == 1:
            return "B"
        else:
            return "C"

    def add_prior(self, result: dict):
        """
        Update the prior based on the result of the quiz.
        If the piroir has been updated, return True; otherwise, return False.
        """
        answer = result["answer"]
        start = self.p2i[result["start"]]
        end = self.p2i[result["end"]]
        if answer == "A":  # changing V1 causes a change in V2.
            return False
        elif answer == "B":  # changing V2 causes a change in V1
            return self.edge_prior.add_exist_edge(end, start)
        elif answer == "C":  # changes in V1 and in V2 are not correlated.
            return self.edge_prior.add_forb_edge(start, end)
        elif answer == "D":  # uncertain.
            return False
        else:
            rprint(f"Invalid answer: {answer}")
            raise Exception("Invalid answer.")

    def _debug_find_wrong(self):
        write_json([], self.wrong_history_path)
        failed_data = []
        all_wrong_num = 0
        for data in self.history:
            truth = self.GT_bot([data["start"], data["end"]])
            if truth != data["answer"]:
                all_wrong_num += 1
                if data["answer"] in ["B", "C"]:
                    data["GT"] = truth
                    failed_data.append(data)
        rprint(
            f"❌ Wrong LLM answers in the history: {len(failed_data)} / {all_wrong_num} / {len(self.history)}")
        write_json(failed_data, self.wrong_history_path)

    def _plot_graphviz(self, ev_edges):
        exist_edges = self.edge_prior.output_exist_edges(string=True)
        forb_edges = self.edge_prior.output_forb_edges(string=True)
        graphviz_path = f"img/graph/{self.exp_name}-iter{self.iter_num}.png"
        graphviz_svg_path = f"img/graph-svg/{self.exp_name}-iter{self.iter_num}.svg"
        graphviz = GraphVis(self.mapping, ev_edges)
        graphviz.init_aux_edges(exist_edges, forb_edges, self.true_edges)
        edge_info_list = graphviz.visualize(graphviz_path, graphviz_svg_path)
        return edge_info_list


class EdgePrior:
    """
    Edge prior:

    0: not certain
    1: edge exist
    -1: reverse edge exist
    2: edge forbidden

    queried_edges:
    0: not queried
    1: queried
    """

    def __init__(self, true_dag, i2p=None, p2i=None) -> None:
        self.true_dag = true_dag
        node_num = true_dag.shape[0]
        self.prior_matrix = np.zeros((node_num, node_num))
        self.queried_edges = np.zeros((node_num, node_num))
        self.node_num = node_num
        if i2p is not None and p2i is not None:
            self.i2p = i2p
            self.p2i = p2i

    def matrix2edge(self, ev_dag, string=True):
        """
        output the edges of ev_dag in the format of "A->B",
        where A and B are the names of the nodes
        """
        candidate_edges = []
        for i in range(ev_dag.shape[0]):
            for j in range(ev_dag.shape[1]):
                if ev_dag[i, j] == 1:
                    candidate_edges.append([i, j])
        if string:
            candidate_edges = [[self.i2p[e[0]], self.i2p[e[1]]]
                               for e in candidate_edges]
        return candidate_edges

    def add_queried_edge(self, start, end, string=False):
        if string:
            start = self.p2i[start]
            end = self.p2i[end]
        self._check_legality(start, end)
        self.queried_edges[start, end] = 1
        self.queried_edges[end, start] = 1

    def check_queried_edge(self, start, end, string=False):
        if string:
            start = self.p2i[start]
            end = self.p2i[end]
        self._check_legality(start, end)
        return True if self.queried_edges[start, end] == 1 else False

    def _check_legality(self, start, end):
        if start == end:
            raise Exception("Invalid edge.")
        if start >= self.node_num or end >= self.node_num:
            raise Exception("Invalid edge.")

    def add_exist_edge(self, start, end, string=False):
        if string:
            start = self.p2i[start]
            end = self.p2i[end]
        self._check_legality(start, end)
        if self.prior_matrix[start, end] == 0 and self.prior_matrix[end, start] == 0:
            self.prior_matrix[start, end] = 1
            self.prior_matrix[end, start] = -1
            return True
        else:
            return False

    def add_forb_edge(self, start, end, string=False):
        if string:
            start = self.p2i[start]
            end = self.p2i[end]
        self._check_legality(start, end)
        if self.prior_matrix[start, end] == 0 and self.prior_matrix[end, start] == 0:
            self.prior_matrix[start, end] = 2
            self.prior_matrix[end, start] = 2
            return True
        else:
            return False

    def output_exist_edges(self, string=False):
        exist_edges = []
        for i in range(self.prior_matrix.shape[0]):
            for j in range(self.prior_matrix.shape[1]):
                if self.prior_matrix[i, j] == 1:
                    exist_edges.append([i, j])
        if string:
            exist_edges = [[self.i2p[e[0]], self.i2p[e[1]]]
                           for e in exist_edges]
        return exist_edges

    def output_forb_edges(self, string=False):
        forb_edges = []
        for i in range(self.prior_matrix.shape[0]):
            for j in range(self.prior_matrix.shape[1]):
                if self.prior_matrix[i, j] == 2:
                    forb_edges.append([i, j])
        if string:
            forb_edges = [[self.i2p[e[0]], self.i2p[e[1]]] for e in forb_edges]
        return forb_edges

    def output_summary(self, string=True):
        true_edges = self.matrix2edge(self.true_dag, string=string)
        exist_edges = self.output_exist_edges(string=string)
        forb_edges = self.output_forb_edges(string=string)
        summary = []
        for edge in exist_edges:
            truth = "exist" if edge in true_edges else "not exist"
            summary.append(
                {"start": edge[0], "end": edge[1], "edge": f"{edge[0]}->{edge[1]}", "prior": "exist", "GT": truth})
        for edge in forb_edges:
            truth = "exist" if edge in true_edges else "not exist"
            summary.append(
                {"start": edge[0], "end": edge[1], "edge": f"{edge[0]}->{edge[1]}", "prior": "forbid", "GT": truth})
        return summary


if __name__ == "__main__":
    args = load_config()
    icsl = Iteration_CSL(args.dataset, model=args.model, alg=args.alg, data_index=args.data_index,
                                         datasize_index=args.datasize_index, score=args.score)
    icsl.iter_causal_structure_learning()
    # dataset_list = ["asia", "child"]
    # alg_score_list = [["CaMML", "mml"],  ["HC", "bdeu"], ["softHC", "bdeu"], ["hardMINOBSx", "bdeu"], ["softMINOBSx", "bdeu"],
    #                   ["HC", "bic"], ["softHC", "bic"], ["hardMINOBSx", "bic"], ["softMINOBSx", "bic"]]
    # for alg_score in alg_score_list:
    #     for dataset in dataset_list:
    #         for data_index in [1, 2, 3, 4, 5, 6]:
    #             for datasize_index in [0, 1]:  # 0: small, 1: large
    #                 args.alg, args.score = alg_score
    #                 args.dataset = dataset
    #                 args.data_index = data_index
    #                 args.datasize_index = datasize_index
    #                 icsl = Iteration_CSL(dataset, model=args.model, alg=args.alg, data_index=args.data_index,
    #                                      datasize_index=args.datasize_index, score=args.score)
    #                 icsl.iter_causal_structure_learning()
