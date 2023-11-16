import json
import math
import os
import re
import shutil
import time
import warnings
from typing import Optional

import chardet
import networkx as nx
import numpy as np
import pandas as pd
import requests
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.GESUtils import *
from causallearn.utils.TXT2GeneralGraph import mod_endpoint, to_endpoint
from retry import retry
from rich import print as rprint

# from simulator import simDAG


def array2generalgraph(A: np.ndarray) -> GeneralGraph:
    n, m = A.shape
    g = GeneralGraph([])
    node_map = {}
    for i in range(n):
        node = 'X'+str(i+1)
        node_map[node] = GraphNode(node)
        g.add_node(node_map[node])
    B = A+A.T
    for i in range(n):
        for j in range(i+1, n):
            if B[i, j] == 1:
                node1 = 'X'+str(i+1)
                node2 = 'X'+str(j+1)
                edge = Edge(node_map[node1], node_map[node2],
                            Endpoint.CIRCLE, Endpoint.CIRCLE)
                if A[i, j] == 1:
                    mod_endpoint(edge, node_map[node2], Endpoint.ARROW)
                else:
                    mod_endpoint(edge, node_map[node2], Endpoint.TAIL)
                if A[j, i] == 1:
                    mod_endpoint(edge, node_map[node1], Endpoint.ARROW)
                else:
                    mod_endpoint(edge, node_map[node1], Endpoint.TAIL)
                g.add_edge(edge)
    return g


def dict2generalgraph(A: dict) -> GeneralGraph:
    g = GeneralGraph([])
    node_map = {}
    for key in A:
        node_map[key] = GraphNode(key)
        g.add_node(node_map[key])
    for key in A:
        for pa in A[key]['par']:
            edge = Edge(node_map[pa], node_map[key],
                        Endpoint.TAIL, Endpoint.ARROW)
            g.add_edge(edge)
    return g


def directed_edge2array(n: int, L: list) -> np.ndarray:
    A = np.zeros([n, n])
    for i in L:
        A[i[0], i[1]] = 1
    return A


def array2directed_edge(A: np.ndarray) -> list:
    a, b = np.where(A != 0)
    return list(zip(a, b))


def array2no_edge(A: np.ndarray) -> list:
    a, b = np.where(A == 0)
    return list(zip(a, b))


def ShowGraph(a: GeneralGraph):
    import io

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from causallearn.utils.GraphUtils import GraphUtils
    pyd = GraphUtils.to_pydot(a)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def truth_score(X: ndarray, score_func: str = 'local_score_BIC', G: GeneralGraph = None, maxP: Optional[float] = None, parameters: Optional[Dict[str, Any]] = None):

    if X.shape[0] < X.shape[1]:
        warnings.warn(
            "The number of features is much larger than the sample size!")

    X = np.mat(X)
    # % k-fold negative cross validated likelihood based on regression in RKHS
    if score_func == 'local_score_CV_general':
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_general, parameters=parameters)

    # negative marginal likelihood based on regression in RKHS
    elif score_func == 'local_score_marginal_general':
        parameters = {}
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_general, parameters=parameters)

    # k-fold negative cross validated likelihood based on regression in RKHS
    elif score_func == 'local_score_CV_multi':
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'kfold': 10, 'lambda': 0.01,
                          'dlabel': {}}  # regularization parameter
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_cv_multi, parameters=parameters)

    # negative marginal likelihood based on regression in RKHS
    elif score_func == 'local_score_marginal_multi':
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)

    # Greedy equivalence search with BIC score
    elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)

    elif score_func == 'local_score_BDeu':  # Greedy equivalence search with BDeu score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None)

    else:
        raise Exception('Unknown function!')
    score_func = localScoreClass

    score = score_g(X, G, score_func, parameters)  # initialize the score

    return score


def dict2list(D, G, P):
    result = []
    for edge in D:
        i, j = edge[1]
        if G.graph[i-1, j-1] == -1:
            flag1 = 'y'
        else:
            if G.graph[i-1, j-1] == 1:
                flag1 = 'r'
            else:
                flag1 = 'n'
        if P.graph[i-1, j-1] == -1:
            flag2 = 'y'
        else:
            if P.graph[i-1, j-1] == 1:
                flag2 = 'r'
            else:
                flag2 = 'n'
        result.append(str((i, j, edge[0], flag1, flag2))+';')
    return result


def array2dict(A: np.ndarray, varnames):
    dag = {}
    n, m = A.shape
    for i in range(n):
        dag[varnames[i]] = {}
        dag[varnames[i]]['par'] = []
        dag[varnames[i]]['nei'] = []
    for i in range(n):
        for j in range(m):
            if A[i, j] == 1:
                dag[varnames[j]]['par'].append(varnames[i])
    return dag


def generate_name(prior_state, prior=True, prior_type=True):
    st = ''
    for key in prior_state:
        if prior_state[key][0] == True:
            if prior:
                st += key
            if prior_type:
                if prior_state[key][1] == True:
                    st += 'r'
                elif prior_state[key][1] == False:
                    st += 'p'
            st += ','
    st = st.strip(',')
    if st == '':
        st = 'n'
    return st


def parse_experiment_results_perform(file_name, column):
    with open(file_name, "r") as f:
        lines = f.readlines()
    res = pd.DataFrame(columns=column)
    for tmp in ["s", "r", "palim"]:
        if tmp in res.columns:
            res[tmp] = res[tmp].astype("int")
    tmp_record = {}
    for line in lines:
        if line[0] == '#':
            continue
        line = line.strip()
        if line.startswith("{"):
            line = line.replace("'", '"')
            line = line.replace("nan", 'null')
            tmp = json.loads(line)
            # add to the pd.dataframe res
            for key in tmp:
                # Add key as a new column name to res
                if key not in res.columns:
                    res[key] = None
                tmp_record[key] = float(
                    tmp[key]) if tmp[key] is not None else np.nan
            res = res.append(tmp_record, ignore_index=True)
            tmp_record = {}
        else:
            if line == '':
                continue
            v_list = line.split(" ")
            for k in v_list:
                key, value = k.split('=')
                try:
                    tmp_record[key] = eval(value)
                except:
                    tmp_record[key] = value
    return res


def parse_prior_results(file_name='exp/path_prior_evaluation.txt'):
    import json

    import pandas as pd
    with open(file_name, "r") as f:
        lines = f.readlines()
    res = pd.DataFrame(columns=["data"])
    props = ["data"]
    tmp_record = {}
    for line in lines:
        line = line.strip()
        if line.startswith("{"):
            line = line.replace("'", '"')
            line = line.replace("nan", 'null')
            tmp = json.loads(line)
            # add to the pd.dataframe res
            for key in tmp:
                # Add key as a new column name to res
                if key not in res.columns:
                    res[key] = None
                tmp_record[key] = float(
                    tmp[key]) if tmp[key] is not None else np.nan
            res = res.append(tmp_record, ignore_index=True)
            tmp_record = {}
        else:
            v_list = line.split(" ")
            for i, k in enumerate(props):
                tmp_record[k] = v_list[i]
    return res


def ReconstructData(src_dir='data/csv', dst_dir='data/txt'):
    '''
    reconstruct the data:
    AAA   BBB   CCC
    True  High  Right
    into:
    0 1 2 3 (index)
    2 2 2 2 (arities)
    0 1 0 1 (data)
    '''
    for path in os.listdir(src_dir):
        data = pd.read_csv(f'{src_dir}/{path}', dtype='category')
        array_data = data.apply(lambda x: x.cat.codes).to_numpy(dtype=int)
        path = path.split('.')[0]
        arities = np.array(data.nunique())
        strtmp = ' '.join([str(i) for i in range(len(data.columns))])
        strtmp += '\n'
        strtmp += ' '.join([str(i) for i in arities])
        np.savetxt(f'{dst_dir}/{path}.txt', array_data,
                   fmt='%d', header=strtmp, comments='')

def parse_parents_score(file_path):
    score_dict = {}
    
    with open(file_path, 'r') as f:
        # read the first line to get the number of nodes (this value is not used in the example)
        node_number = int(f.readline().strip())
        
        while True:
            line = f.readline().strip()
            if not line:
                break
                
            # Extract information from the line
            parts = line.split()
            node_index = int(parts[0])
            num_parent_set = int(parts[1])
            
            # Create an empty list to hold the score and parent information for the current child node
            score_list = []
            
            # Loop over the number of parent sets and read each set of scores and parents
            for _ in range(num_parent_set):
                line = f.readline().strip()
                parent_parts = line.split()
                
                # Extract score and parent indices
                score = float(parent_parts[0])
                parent_num = int(parent_parts[1])
                parents = [str(x) for x in parent_parts[2: 2 + parent_num]]
                
                # Append to the list of scores and parents for this child node
                score_list.append((score, parents))
            
            # Save to the score_dict
            score_dict[str(node_index)] = score_list
            
    return score_dict

def write_parents_score(score_dict, file_path):
    with open(file_path, 'w') as f:
        n = len(score_dict)
        f.write(f"{n}\n")
        for var in score_dict:
            f.write(f"{var} {len(score_dict[var])}\n")
            for score, parent_list in score_dict[var]:
                new_score = "{:.8f}".format(score)
                f.write(f"{new_score} {len(parent_list)} {' '.join(parent_list)}\n")
            

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


def delate_ancs(dag: np.array, ancs):
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    for anc, child in ancs:
        try:
            path = nx.shortest_path(G, source=anc, target=child)
        except:
            path = None
        while path != None:
            for i in range(len(path)-1):
                G.remove_edge(path[i], path[i+1])
            try:
                path = nx.shortest_path(G, source=anc, target=child)
            except:
                path = None

    return nx.to_numpy_array(G)


def check_acyclic(dag: np.array):
    n = dag.shape[0]
    visited = np.zeros(n)
    for i in range(n):
        if visited[i] == 0:
            if check_path(dag, i, i):
                return False
    return True


def cyclic2acyclic(dag: np.array):
    # chekc if the dag is cyclic, if so, make it acyclic
    n = dag.shape[0]
    visited = np.zeros(n)
    for i in range(n):
        if visited[i] == 0:
            if check_path(dag, i, i):
                # find a cycle
                # find the edge in the cycle
                for j in range(n):
                    if dag[i][j] == 1 and check_path(dag, j, i):
                        dag[i][j] = 0
    # print('The algorithm for cycle removement is relatively simple, it can only ensure that the result is acyclic, but cannot ensure that the removed edges are minimal!')
    return dag


def clearcycle(true_dag, order_constraints):
    dag = np.zeros(true_dag.shape)
    for edge in order_constraints:
        dag[edge[0], edge[1]] = 1

    dag = cyclic2acyclic(dag)
    # print(check_acyclic(dag))
    edges = np.argwhere(dag == 1)
    edges = [(edge[0], edge[1]) for edge in edges]
    return edges


def save_result(m, ev_dag, **kwargs):

    print(f"d={kwargs['d']} s={kwargs['s']} r={kwargs['r']} conf={kwargs['conf']} palim={kwargs['palim']} prior={kwargs['prior']} prior_type={kwargs['prior_type']} pruning={kwargs['nopruning']} score={kwargs['score']}  prior_source={kwargs['prior_source']}\n{m.metrics}")
    nowtime = re.sub('\s+', '/', time.asctime(time.localtime()))
    with open(kwargs['output'], 'a') as f:
        f.write(f"d={kwargs['d']} s={kwargs['s']} r={kwargs['r']} conf={kwargs['conf']} palim={kwargs['palim']} prior={kwargs['prior']} prior_type={kwargs['prior_type']} pruning={kwargs['nopruning']} score={kwargs['score']}  prior_source={kwargs['prior_source']} finish_time={nowtime}\n{m.metrics}\n")

    if kwargs['method'] in ['DP', 'Astar', 'ELSA', 'PGMINOBSx']:
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['palim']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['nopruning']}_{kwargs['score']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")
    elif kwargs['method'] == 'CaMML':
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['palim']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")
    else:
        np.savetxt(f"{kwargs['dag_path']}/{kwargs['method']}_{kwargs['d']}_{kwargs['s']}_{kwargs['r']}_{kwargs['conf']}_{kwargs['prior']}_{kwargs['prior_type']}_{kwargs['prior_source']}{kwargs['fix']}.txt", ev_dag, fmt="%d")


def noprior_dag_metric(input="exp/CaMML_noprior.txt", output="exp/CaMML_noprior_statistics.csv", cat_column=["prior", "prior_type", "palim", "d", "s"], merge_column=["r"], mean=True):
    res = parse_experiment_results_perform(
        input, column=cat_column+merge_column)
    # res=res.dropna()
    res['precision'] = res['precision'].fillna(0)
    res['F1'] = res['F1'].fillna(0)

    warnings.filterwarnings("ignore")
    if mean:
        res = res.groupby(cat_column).mean()
    else:
        res.sort_values(cat_column, inplace=True)
        res.reset_index(drop=True, inplace=True)
    res.drop(labels=['fdr', 'tpr', 'fpr', 'nnz', 'r', 'gscore', 'delp_fdr', 'delp_tpr',
             'delp_fpr', 'delp_nnz', 'delp_gscore'], axis=1).to_csv(output, float_format="%.2f")


def mkdir(path):
    """
    make directory, if the directory exists, do nothing
    """
    if not os.path.exists(path):
        os.makedirs(path)
        rprint(f"📂 Created folder [italic blue]{path}[/italic blue].")


def mkdir_rm(path):
    """
    make directory, if the directory exists, remove it and create a new one
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        rprint(f"🗑️  Removed folder [italic blue]{path}[/italic blue].")
    mkdir(path)


def auto_clean_folder(folder_path, max_num=50):
    """
    an automatic clean function for folder, keep the latest max_num files
    """
    folder_num = len(os.listdir(folder_path))
    if folder_num > max_num:
        file_list = os.listdir(folder_path)
        file_list.sort(key=lambda fn: os.path.getmtime(
            os.path.join(folder_path, fn)))
        num_to_remove = len(file_list) - max_num
        for i in range(num_to_remove):
            file_path = os.path.join(folder_path, file_list[i])
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.chmod(file_path, 0o777)
                shutil.rmtree(file_path)
            rprint(f"🗑️  Removed file [italic blue]{file_path}[/italic blue].")
        rprint(
            f"♻️  [bold yellow]Auto clean[/bold yellow] {num_to_remove} files in [italic blue]{folder_path}[/italic blue].")


def write_txt(txt_path, content,mode="w"):
    """
    write content to txt file
    """
    with open(txt_path, mode, encoding='utf-8') as f:
        f.write(content)
    if mode=="w":
        rprint(
            f"📝 Write [bold yellow]txt[/bold yellow] file to [italic blue]{txt_path}[/italic blue].")
    else:
        rprint(
            f"📝 Append [bold yellow]txt[/bold yellow] file to [italic blue]{txt_path}[/italic blue].")


def auto_detect_encoding(file_path):
    """
    detect encoding of file automatically
    """
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    rprint(
        f"🔍 Detected encoding: [bold yellow]{encoding}[/bold yellow] of [italic blue]{file_path}[/italic blue].")
    return encoding


def read_txt(txt_path, encoding='utf-8'):
    """
    read txt file
    use utf-8 as default encoding, if you want to auto detect encoding, set encoding='auto'
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(txt_path)
    with open(txt_path, "r", encoding=encoding) as f:
        content = f.read()
    rprint(
        f"📖 Read [bold yellow]txt[/bold yellow] file from [italic blue]{txt_path}[/italic blue].")
    return content


def write_json(content, json_path, encoding='utf-8', indent=4):
    """
    Write content to json file.

    Args:
        content: dict, list, or other json serializable object.
        json_path: str, path to json file.
        encoding: str, encoding of json file.
        indent: int, indent of json file.
    """
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(content, f, ensure_ascii=False, indent=indent)
    rprint(
        f"📝 Write [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")


def read_json(json_path, encoding='utf-8',quiet=False):
    """
    read json file
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    if not quiet:
        rprint(
            f"📖 Read [bold yellow]json[/bold yellow] file from [italic blue]{json_path}[/italic blue].")
    return content


def update_json(add_content, json_path, encoding='utf-8'):
    """
    update json file
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    content.update(add_content)
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    rprint(
        f"🔄 Update [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")


def append_json(add_content, json_path, encoding='utf-8'):
    """
    append json file
    """
    if encoding == 'auto':
        encoding = auto_detect_encoding(json_path)
    with open(json_path, "r", encoding=encoding) as f:
        content = json.load(f)
    content.append(add_content)
    with open(json_path, "w", encoding=encoding) as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    rprint(
        f"🔄 Update [bold yellow]json[/bold yellow] file to [italic blue]{json_path}[/italic blue].")


def warning_filter(func):
    """
    a decorator to filter warnings
    """
    def inner(*args, **kwargs):
        warnings.filterwarnings("ignore")
        result = func(*args, **kwargs)
        warnings.filterwarnings("default")
        return result
    return inner


def timer(func):
    """
    a decorator to calculate the time cost of a function
    """
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        rprint(
            f"⏱️ Function [italic blue]{func.__name__}[/italic blue] cost [bold yellow]{end-start:.4f}[/bold yellow] seconds.")
        return result
    return inner

def sort(list):
    result = {}
    for tuple in list:
        if tuple[0][1] not in result.keys():
            result[tuple[0][1]] = {}
            result[tuple[0][1]][tuple[0][0]] = tuple[1]
        else:
            result[tuple[0][1]][tuple[0][0]] = tuple[1]
    return result

class soft_constraint:
    def __init__(self, obligatory, forbidden, lamdba):
        # 3 parameters: obligatory edges, forbidden edges, lambda
        self.obligatory = sort([(x, y) for x, y in zip(obligatory, lamdba[0])])
        self.forbidden = sort([(x, y) for x, y in zip(forbidden, lamdba[1])])

    def calculate(self, var, parent):
        prior_score = 0
        if var in self.obligatory.keys():
            for p in self.obligatory[var].keys():
                if p in parent:
                    prior_score += math.log(self.obligatory[var][p])
                else:
                    prior_score += math.log(1-self.obligatory[var][p])
        if var in self.forbidden.keys():
            for p in self.forbidden[var].keys():
                if p in parent:
                    prior_score += math.log(1-self.forbidden[var][p])
                else:
                    prior_score += math.log(self.forbidden[var][p])
        return prior_score

class ColorP:
    def __init__(self) -> None:
        pass

    @staticmethod
    def edge(edge):
        start, end = edge
        return f"[purple]{start}[/purple]->[purple]{end}[/purple]"

    @staticmethod
    def answer(answer, true_ans):
        if answer == "D":  # answer is uncertain
            return f"(Ans: [yellow]{answer}[/yellow] / [green]{true_ans}[/green])"
        elif answer == true_ans:  # answer is correct
            if answer in ["B", "C"]:  # the correct answer makes effect
                return f"(Ans: [bold green]{answer}[/bold green] / [green]{true_ans}[/green])"
            else:  # the correct answer does not make effect
                return f"(Ans: [green]{answer}[/green] / [green]{true_ans}[/green])"
        elif answer == "A":  # answer is wrong, but does not make effect
            return f"(Ans: [yellow]{answer}[/yellow] / [green]{true_ans}[/green])"
        else:  # answer is wrong, and makes effect
            return f"(Ans: [bold red]{answer}[/bold red] / [green]{true_ans}[/green])"

    @staticmethod
    def GT(GT):
        return f"(TrueAns: [yellow]{GT}[/yellow])"

    @staticmethod
    def model(model):
        return f"[bold yellow]{model}[/bold yellow]"

    @staticmethod
    def path(path):
        return f"[italic blue]{path}[/italic blue]"

    @staticmethod
    def warning(content):
        return f"[red]{content}[/red]"

if __name__ == "__main__":
    score_dict = parse_parents_score("data/score/bdeu/asia_1000_1.txt")
    write_parents_score(score_dict,"test_score.tmp")