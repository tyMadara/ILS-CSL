import os
import re

import numpy as np


def parse_CaMML_output(file_path, dag_flag=True):
    node_pattern = r"node\s+(\S+)\s+{.*"
    parents_pattern = r".*parents.*\((.*)\).*"
    node_list = []
    parent_list = []
    with open(file_path, "r", encoding="utf-8") as ifh:
        for line in ifh.readlines():
            m1 = re.match(node_pattern, line)
            if m1:
                node = m1.group(1).strip()
                node_list.append(node)
                continue
            m2 = re.match(parents_pattern, line)
            if m2:
                parents = m2.group(1).strip()
                if parents == "":
                    parent_list.append([])
                else:
                    tp = parents.split(",")
                    tp = [e.strip() for e in tp]
                    parent_list.append(tp)
    p_dict = {}
    n2i = {}
    for i, n in enumerate(node_list):
        n2i[n] = i
        p_dict[n] = parent_list[i]

    if not dag_flag:
        return n2i, p_dict

    n = len(n2i)
    CaMML_dag = np.zeros((n, n))
    for v in p_dict:
        v1 = n2i[v]
        for p in p_dict[v]:
            p1 = n2i[p]
            CaMML_dag[p1][v1] = 1
    return CaMML_dag


def CaMML_unit(CaMML_base, prefix, i2p, **kwargs):

    os.chdir(CaMML_base)

    anc_path = f"anc_file/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.anc"
    out_path = f"out_BNs/{prefix}{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.dne"

    with open(anc_path, "w") as ofh:
        reconf = 1-kwargs['conf']
        out_str = "arcs{\n"
        for a in kwargs['ancs']:
            v1, v2 = a
            out_str += f"{i2p[v1]} => {i2p[v2]} {kwargs['conf']};\n"
        for a in kwargs['forb_ancs']:
            v1, v2 = a
            out_str += f"{i2p[v1]} => {i2p[v2]} {reconf};\n"
        for a in kwargs['forb_edges']:
            v1, v2 = a
            out_str += f"{i2p[v1]} -> {i2p[v2]} {reconf:.5f};\n"
        for a in kwargs['exist_edges']:
            v1, v2 = a
            out_str += f"{i2p[v1]} -> {i2p[v2]} {kwargs['conf']};\n"
        out_str += "}"
        if len(kwargs['order']) > 0:
            out_str += "\ntier{\n"
            for order_instance in kwargs['order']:
                out_str += "<".join([i2p[i] for i in order_instance])
                out_str += ";\n"
            out_str += "}"
        ofh.write(out_str)

    os.system(
        f"./camml.sh -p {anc_path} ../data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv {out_path} > log.txt")
    ev_dag = parse_CaMML_output(out_path)

    os.system(f"rm {anc_path}")
    os.system(f"rm {out_path}")
    os.chdir("..")
    return ev_dag
