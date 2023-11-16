import copy

import numpy as np
import pandas as pd


class MetricsDAG:
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP / (TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP / (TP + FP)
    recall: TP / (TP + FN)
    F1: 2*(recall*precision) / (recall + precision)
    gscore: max(0, (TP - FP)) / (TP + FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):

        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0

        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]

        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(
            pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(
            pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(
            cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'extra': len(extra_lower), 'missing': len(missing_lower), 'reverse': len(reverse), 'fdr': fdr, 'tpr': tpr,
              'fpr': fpr, 'shd': shd, 'nnz': pred_size, 'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)

        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        num_true = W_true.sum(axis=1).sum()

        # true_positives
        num_tp = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                         2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        num_fn_r = (W_p - W_true).applymap(lambda elem: 1 if elem ==
                                           1 else 0).sum(axis=1).sum()
        if num_true != 0:
            score = np.max((num_tp-num_fn_r, 0))/num_true
        else:
            score = 0
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
        TP = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                     2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)

        return precision, recall, F1

    def _count_forb_edge(self, B_est, forb_edges=None, order=None, ancs=None, forb_ancs=None, prior_exist=[]):
        if prior_exist['fe'][0] == True:
            self.metrics["fe_st"] = 0
            self.metrics["fe_nost"] = 0
            for prior in forb_edges:
                if B_est[prior[0], prior[1]] == 0:
                    self.metrics["fe_st"] += 1
                else:
                    self.metrics["fe_nost"] += 1

        if prior_exist['o'][0] == True:
            self.metrics["o_st"] = 0
            self.metrics["o_nost"] = 0
            for prior in order:
                if check_path(B_est, prior[0], prior[1]):
                    self.metrics["o_st"] += 1
                else:
                    self.metrics["o_nost"] += 1

        if prior_exist['a'][0] == True:
            self.metrics["a_st"] = 0
            self.metrics["a_nost"] = 0
            for prior in ancs:
                if check_path(B_est, prior[0], prior[1]):
                    self.metrics["a_st"] += 1
                else:
                    self.metrics["a_nost"] += 1
        if prior_exist['fa'][0] == True:
            raise KeyError

    def _count_anc(self, B_est, ancs):
        self.metrics["ancs_statisfy"] = 0
        for prior in ancs:
            if check_path(B_est, prior[0], prior[1]):
                self.metrics["ancs_statisfy"] += 1


def check_path(dag: np.array, source, dest):
    # Check if there is a directed path
    n = dag.shape[0]
    visited = np.zeros(n)
    queue = [source]
    while len(queue) > 0:
        v = queue.pop(0)
        for i in range(n):
            if dag[v][i] == 1 and visited[i] == 0:
                visited[i] = 1
                queue.append(i)
    return visited[dest] == 1
