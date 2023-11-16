
import os
import re

import numpy as np

from utils import parse_parents_score, soft_constraint, write_parents_score


def parse_screen_output(file_path, n):
    res = np.zeros((n, n))
    with open(file_path, "r", encoding="utf-8") as ifh:
        for line in ifh:
            mt = re.match(r".*=\s(\d+).*\{(.*)\}.*", line)
            if not mt:
                continue
            child = mt.group(1)
            parents = mt.group(2)
            parents = re.split(r"\s+", parents.strip())
            c = int(child)
            for p in parents:
                if p == "":
                    continue
                p = int(p)
                res[p, c] = 1
    return res

def seperate_MINOBSx_unit(MINOBSx_base="minobsx", timeout=None, iter=10, prefix="", prior_confidence=0.99999, **kwargs):

    os.chdir(MINOBSx_base)

    anc_path = f"anc_file/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.anc"
    out_path = f"out_BNs/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.dne"
    
    score_filepath = kwargs["score_filepath"]
    
    if kwargs["is_soft"] == False:
        with open(anc_path, "w", encoding="utf-8") as ofh:
            ofh.write(f"{len(kwargs['exist_edges'])}\n")
            for c in kwargs['exist_edges']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"0\n")  # undirected edges
            ofh.write(f"{len(kwargs['forb_edges'])}\n")
            for c in kwargs['forb_edges']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['order'])}\n")
            for c in kwargs['order']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['ancs'])}\n")
            for c in kwargs['ancs']:
                ofh.write(f"{c[0]} {c[1]}\n")
            ofh.write(f"{len(kwargs['forb_ancs'])}\n")
            for c in kwargs['forb_ancs']:
                ofh.write(f"{c[0]} {c[1]}\n")
        if timeout is None:
            os.system(
                f"./run-one-case.sh ../{score_filepath} {anc_path} tmp.output {iter} > {out_path}")
        else:
            os.system(
                f"timeout {timeout} ./run-one-case.sh ../{score_filepath} {anc_path} tmp.output {iter} > {out_path}")
        ev_dag = parse_screen_output(f"{out_path}", kwargs['true_dag'].shape[0])
        os.system(f"rm {anc_path}")
        os.system(f"rm {out_path}")
        os.chdir("..")
        return ev_dag
    else:
        with open(anc_path, "w", encoding="utf-8") as ofh:
            ofh.write(f"0\n0\n0\n0\n0\n0\n")
        soft_score_path = f"score/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.score"
        score_dict = parse_parents_score(f"../{score_filepath}")
        edge_cons, forb_cons = kwargs["exist_edges"], kwargs['forb_edges']
        soft_scorer = soft_constraint(obligatory=edge_cons, forbidden=forb_cons, lamdba=[[prior_confidence for _ in range(len(edge_cons))],[prior_confidence for _ in range(len(forb_cons))]])
        for var in score_dict:
            for i, ls in enumerate(score_dict[var]):
                score, parent_set = ls
                prior_bonus = soft_scorer.calculate(var=int(var),parent=[int(p) for p in parent_set])
                new_score = score + prior_bonus
                score_dict[var][i] = (new_score, parent_set)
        write_parents_score(score_dict=score_dict, file_path=soft_score_path)
        if timeout is None:
            os.system(
                f"./run-one-case.sh {soft_score_path} {anc_path} tmp.output {iter} > {out_path}")
        else:
            os.system(
                f"timeout {timeout} ./run-one-case.sh {soft_score_path} {anc_path} tmp.output {iter} > {out_path}")
        ev_dag = parse_screen_output(f"{out_path}", kwargs['true_dag'].shape[0])
        os.system(f"rm {anc_path}")
        os.system(f"rm {out_path}")
        os.system(f"rm {soft_score_path}")
        os.chdir("..")
        return ev_dag