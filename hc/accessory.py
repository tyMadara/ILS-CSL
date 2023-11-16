from itertools import combinations, chain

import numpy as np
import pingouin as pg
from numba import njit
from scipy.stats import chi2, norm
from sklearn.linear_model import LinearRegression


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@njit(fastmath=True)
def bic(data, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ij = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ij = 0
        for i in range(len(cols)):
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_ij += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += 1
        for i in range(arities[cols[0]]):
            N_ij[idx_ij + i * strides[0]] += 1
    bic = 0
    for i in range(stride):
        if N_ijk[i] != 0:
            bic += N_ijk[i] * np.log(N_ijk[i] / N_ij[i])
    bic -= 0.5 * np.log(data.shape[0]) * (arities[cols[0]] - 1) * strides[0]
    return bic


def bic_g(data, arities, cols):
    # modified by clz
    #data = data.to_numpy()
    y = data[:, cols[0]]
    if len(cols) == 1:
        resids = np.mean(y) - y
    else:
        X = data[:, cols[1:]]
        reg = LinearRegression().fit(X, y)
        preds = reg.predict(X)
        resids = y - preds
    sd = np.std(resids)
    numparams = len(cols) + 1  # include intercept AND sd (even though latter is not a free param)
    bic = norm.logpdf(resids, scale=sd).sum() - np.log(data.shape[0]) / 2 * numparams
    return bic

def local_score_from_storage(cols, score_dict):
    cols = np.asarray(cols)
    variable_idx = cols[0]
    parent_idxs = cols[1:]
    str_parent_idxs = [str(x) for x in parent_idxs]
    str_variable_idx = str(variable_idx)
    for score, p_idx in score_dict[str_variable_idx]:
        if set(p_idx) == set(str_parent_idxs):
            return score
    return -np.nan

def local_score(data, arities, cols, score='default'):
    '''
    :param weight: weight for data
    :param data: numbered version of data set
    :param cols: the index of node and its parents, the first element represents the index of the node and the following elements represent the indices of its parents
    :param score_function: name of score function, currently support bic, nal, bic_g
    :return: local score of node (cols[0]) given its parents (cols[1:])
    '''
    if len(data) == 0:
        return np.nan
    else:
        if score == 'default':
            score = 'bic_g' if arities is None else 'bic'
        try:
            ls = globals()[score](data, arities, np.asarray(cols))
        except Exception as e:
            raise Exception('score function ' + str(
                e) + ' is undefined or does not fit to data type. Available score functions are: bic (BIC for discrete variables) and bic_g (BIC for continuous variables).')
        return ls


# statistical test
def independence_test(p_value, tar, pc, can, data, arities, varnames, test='g-test', threshold=0.05):
    '''
    statistical independence test
    :param p_value: a dictionary contains the maximum p-value of CI tests for each variable
    :param pc: parents and children set of the target variable
    :param can: candidate variable for the pc set of the target variable
    :param data: input data (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param prune: whether use prune method
    :param test: type of statistical test (currently support g-test)
    :param threshold: threshold for statistical test to determine independence
    :return: a dictionary contains the maximum p-value of CI test for each variable
    '''
    for can_var in can:
        if can_var not in p_value.keys():
            p_value[can_var] = 0
        for con in powerset(pc[0:-1]):
            # avoid checking the separation set that has been checked in previous iterations
            con = list(con)
            if len(pc) != 0:
                con.append(pc[-1])
            cols = np.array([varnames.index(x) for x in [tar, can_var] + con])
            if test == 'g-test':
                G, dof = it_counter(data, arities, cols)
                p = chi2.sf(G, dof)
                p_value[can_var] = max(p, p_value[can_var])
            elif test == 'z-test':
                # under construction
                r = pg.partial_corr(data=data, x=tar, y=can_var, covar=con)['r'][0]
                z = np.sqrt(data.shape[0] - len(con) - 3) * np.arctanh(r)
                p = 2 * min(norm.cdf(z), norm.cdf(-z))
                p_value[can_var] = max(p, p_value[can_var])
            else:
                raise Exception('statistical test ' + test + ' is undefined, currently supported tests are: g-test')
            if p_value[can_var] > threshold:
                break
    return p_value


@njit(fastmath=True)
def it_counter(data, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ik = np.zeros(stride)
    N_jk = np.zeros(stride)
    N_k = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ik = 0
        idx_jk = 0
        idx_k = 0
        for i in range(len(cols)):
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_jk += data[rowidx, cols[i]] * strides[i]
            if i != 1:
                idx_ik += data[rowidx, cols[i]] * strides[i]
            if (i != 0) & (i != 1):
                idx_k += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += 1
        for j in range(arities[cols[1]]):
            N_ik[idx_ik + j * strides[1]] += 1
        for i in range(arities[cols[0]]):
            N_jk[idx_jk + i * strides[0]] += 1
        for i in range(arities[cols[0]]):
            for j in range(arities[cols[1]]):
                N_k[idx_k + i * strides[0] + j * strides[1]] += 1
    G = 0
    for i in range(stride):
        if N_ijk[i] != 0:
            G += 2 * N_ijk[i] * np.log(N_ijk[i] * N_k[i] / N_ik[i] / N_jk[i])

    dof = max((arities[cols[0]] - 1) * (arities[cols[1]] - 1) * strides[1], 1)
    return G, dof


# convert the dag to bnlearn format
def to_bnlearn(dag):
    output = ''
    for var in dag:
        output += '[' + var
        if dag[var]['par']:
            output += '|'
            for par in dag[var]['par']:
                output += par + ':'
            output = output[:-1]
        output += ']'
    return output
