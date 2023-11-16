
import numpy as np

from main import Iteration_CSL
from utils import *


class GraphSampler:
    def __init__(self, ev_dag, i2p=None, dataset="dataset"):
        self.i2p = i2p if i2p is not None else self.i2p
        self.ev_dag = ev_dag
        self.study_dir = f"pairwise_samples/{dataset}"
        os.makedirs(self.study_dir, exist_ok=True)

    def floyd_warshall_reachability(self):
        n = len(self.ev_dag)
        reachability = self.ev_dag.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    reachability[i, j] = reachability[i, j] or (
                        reachability[i, k] and reachability[k, j])
        return reachability

    def _direct_connections(self):
        rows, cols = np.where(self.ev_dag > 0)
        return list(zip(rows, cols))

    def _indirect_connections(self):
        reachability_matrix = self.floyd_warshall_reachability()
        rows, cols = np.where((reachability_matrix - self.ev_dag) > 0)
        return list(zip(rows, cols))

    def _no_connections(self):
        reachability_matrix = self.floyd_warshall_reachability()
        no_connection_pairs = []
        n = reachability_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if reachability_matrix[i, j] == 0 and reachability_matrix[j, i] == 0:
                    no_connection_pairs.append((i, j))
        return no_connection_pairs

    def sample_pairs(self, num_samples=20):
        sampling_methods = {
            "direct_connections": self._direct_connections,
            "indirect_connections": self._indirect_connections,
            "no_connection": self._no_connections
        }

        sampled_results = {}

        for key, method in sampling_methods.items():
            pairs = method()
            num = min(num_samples, len(pairs))
            sampled_indices = np.random.choice(
                len(pairs), num, replace=False)
            sampled_pairs = [pairs[i] for i in sampled_indices]
            sampled_results[key] = sorted(sampled_pairs)
            rprint(f"{key.capitalize()} connections: {len(pairs)}, Sampled: {num}")

        # Save the results to files
        for key, pairs in sampled_results.items():
            write_txt(f"{self.study_dir}/{key}.txt",
                      "\n".join([f"{pair[0]},{pair[1]}" for pair in pairs]))
            write_txt(f"{self.study_dir}/{key}_str.txt",
                      "\n".join([f"{self.i2p[pair[0]]},{self.i2p[pair[1]]}" for pair in pairs]))
            
        return sampled_results


if __name__ == "__main__":
    for dataset in ["alarm", "asia", "insurance", "mildew", "child", "cancer", "water", "barley"]:
        icsl = Iteration_CSL(dataset)
        ev_dag = icsl.true_dag  # adjacency matrix
        from rich import print as rprint
        sampler = GraphSampler(ev_dag,icsl.i2p,dataset)
        sampled_results = sampler.sample_pairs(20)
