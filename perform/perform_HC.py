import numpy as np
import pandas as pd

from hc.DAG import DAG
from hc.hc import hc_prior
from utils import parse_parents_score, soft_constraint, write_parents_score


def hc_test(prefix="",prior_confidence=0.99999, **kwargs):
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    # change dtype = 'float64'/'category' if data is continuous/categorical
    data = pd.read_csv(filename, dtype='category')
    D = DAG(list(data.columns))

    D.load_data(data, test=None, score=None)
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")


    exist_edges = kwargs['exist_edges']
    
    score_filepath = kwargs['score_filepath'] if 'score_filepath' in kwargs else None
    
    if kwargs["is_soft"]:
        soft_score_path = f"data/score/tmp/{prefix}hc_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
        score_dict = parse_parents_score(f"{score_filepath}")
        edge_cons, forb_cons = kwargs["exist_edges"], kwargs['forb_edges']
        soft_scorer = soft_constraint(obligatory=edge_cons, forbidden=forb_cons, lamdba=[[prior_confidence for _ in range(len(edge_cons))],[prior_confidence for _ in range(len(forb_cons))]])
        for var in score_dict:
            for i, ls in enumerate(score_dict[var]):
                score, parent_set = ls
                prior_bonus = soft_scorer.calculate(var=int(var),parent=[int(p) for p in parent_set])
                new_score = score + prior_bonus
                score_dict[var][i] = (new_score, parent_set)
        write_parents_score(score_dict=score_dict, file_path=soft_score_path)
        score_filepath = soft_score_path
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        tmp[np.diag_indices_from(tmp)] = 0
        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])
    else:
        tmp = np.ones((D.data.shape[1], D.data.shape[1]))
        for prior in kwargs['forb_edges']:
            tmp[prior[0], prior[1]] = 0
        tmp[np.diag_indices_from(tmp)] = 0
        for par, var in exist_edges:
            D.a_prior[D.varnames[var]]['par'].append(D.varnames[par])

        for prior in list(zip(*np.where(tmp))):
            D.pc[D.varnames[prior[1]]].append(D.varnames[prior[0]])

    
    hc_prior(D, score_filepath=score_filepath)

    D.dag2graph()
    return D.graph
