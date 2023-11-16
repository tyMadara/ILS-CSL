from hc.accessory import local_score, local_score_from_storage
from hc.DAG import DAG, check_cycle
from utils import parse_parents_score


def hc(data, arities, varnames, pc=None, score='default'):
    '''
    :param data: the training data used for learn BN (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param pc: the candidate parents and children set for each variable
    :param score: score function, including:
                   bic (Bayesian Information Criterion for discrete variable)
                   bic_g (Bayesian Information Criterion for continuous variable)

    :return: the learned BN (bnlearn format)
    '''
    if score == 'default':
        score = 'bic_g' if arities is None else 'bic'
    # initialize the candidate parents-set for each variable
    candidate = {}
    dag = {}
    cache = {}
    for var in varnames:
        if pc is None:
            candidate[var] = list(varnames)
            candidate[var].remove(var)
        else:
            candidate[var] = list(pc[var])
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        cache[var] = {}
        cache[var][tuple([])] = local_score(
            data, arities, [varnames.index(var)], score)
    diff = 1
    while diff > 0:
        diff = 0
        edge_candidate = []
        for vi in varnames:
            # attempt to add edges vi->vj
            for vj in candidate[vi]:
                cyc_flag = check_cycle(vi, vj, dag)
                if not cyc_flag:
                    par_sea = tuple(sorted(dag[vj]['par'] + [vi]))
                    if par_sea not in cache[vj]:
                        cols = [varnames.index(x) for x in (vj, ) + par_sea]
                        cache[vj][par_sea] = local_score(
                            data, arities, cols, score)
                    diff_temp = cache[vj][par_sea] - \
                        cache[vj][tuple(dag[vj]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [vi, vj, 'a']
            for par_vi in dag[vi]['par']:
                # attempt to reverse edges from vi<-par_vi to vi->par_vi
                cyc_flag = check_cycle(vi, par_vi, dag)
                if not cyc_flag:
                    par_sea_par_vi = tuple(sorted(dag[par_vi]['par'] + [vi]))
                    if par_sea_par_vi not in cache[par_vi]:
                        cols = [varnames.index(x)
                                for x in (par_vi, ) + par_sea_par_vi]
                        cache[par_vi][par_sea_par_vi] = local_score(
                            data, arities, cols, score)
                    par_sea_vi = tuple(
                        [x for x in dag[vi]['par'] if x != par_vi])
                    if par_sea_vi not in cache[vi]:
                        cols = [varnames.index(x) for x in (vi, ) + par_sea_vi]
                        cache[vi][par_sea_vi] = local_score(
                            data, arities, cols, score)
                    diff_temp = cache[par_vi][par_sea_par_vi] + cache[vi][par_sea_vi] - cache[par_vi][
                        tuple(dag[par_vi]['par'])] - cache[vi][tuple(dag[vi]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [vi, par_vi, 'r']
                # attempt to delete edges vi<-par_vi
                par_sea = tuple([x for x in dag[vi]['par'] if x != par_vi])
                if par_sea not in cache[vi]:
                    cols = [varnames.index(x) for x in (vi, ) + par_sea]
                    cache[vi][par_sea] = local_score(
                        data, arities, cols, score)
                diff_temp = cache[vi][par_sea] - \
                    cache[vi][tuple(dag[vi]['par'])]
                if diff_temp - diff > 1e-10:
                    diff = diff_temp
                    edge_candidate = [par_vi, vi, 'd']
        if edge_candidate:
            if edge_candidate[-1] == 'a':
                dag[edge_candidate[1]]['par'] = sorted(
                    dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                candidate[edge_candidate[0]].remove(edge_candidate[1])
                candidate[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                dag[edge_candidate[1]]['par'] = sorted(
                    dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                dag[edge_candidate[1]]['par'].remove(edge_candidate[0])
                candidate[edge_candidate[0]].append(edge_candidate[1])
                candidate[edge_candidate[1]].append(edge_candidate[0])
    return dag


def hc_prior(D: DAG, score_filepath=None):
    '''
    :param data: the training data used for learn BN (numpy array)
    :param arities: number of distinct value for each variable
    :param varnames: variable names
    :param pc: the candidate parents and children set for each variable
    :param score: score function, including:
                   bic (Bayesian Information Criterion for discrete variable)
                   bic_g (Bayesian Information Criterion for continuous variable)

    :return: the learned BN (bnlearn format)
    '''
    score_dict = {}
    if score_filepath is not None:
        score_dict = parse_parents_score(score_filepath)

    record_score = 0

    def update_cache(cache, var, par_sea):
        if par_sea not in cache[var]:
            cols = [D.varnames.index(x) for x in (var, ) + par_sea]
            cache[var][par_sea] = local_score_from_storage(
                cols, score_dict)

    if D.score == 'default':
        D.score = 'bic_g' if D.arities is None else 'bic'
    # initialize the candidate parents-set for each variable
    candidate = {}
    dag = {}
    cache = {}
    for var in D.varnames:
        # Edge forbidden
        if D.pc is None:
            candidate[var] = list(D.varnames)
            candidate[var].remove(var)
        else:
            candidate[var] = list(D.pc[var])
        dag[var] = {}
        dag[var]['par'] = []
        dag[var]['nei'] = []
        cache[var] = {}
        record_score += 1
        if score_filepath is not None:
            cache[var][tuple([])] = local_score_from_storage(
                [D.varnames.index(var)], score_dict)
        else:
            cache[var][tuple([])] = local_score(
                D.data, D.arities, [D.varnames.index(var)], D.score)
    for var in D.varnames:
        #Add prior
        dag[var]['par'] = sorted(dag[var]['par']+D.a_prior[var]['par'])
        par_sea = tuple(sorted(dag[var]['par']))
        if par_sea not in cache[var]:
            cols = [D.varnames.index(x) for x in (var, ) + par_sea]
            record_score+=1
            cache[var][par_sea] = local_score_from_storage(cols, score_dict)
        for par in D.a_prior[var]['par']:
            if par in candidate[var]:
                candidate[var].remove(par)
            if var in candidate[par]:
                candidate[par].remove(var)
        for par in D.r_prior[var]['par']:
            dag[par]['par'] = sorted(dag[par]['par']+[var])
            par_sea = tuple(sorted(dag[par]['par']))
            if par_sea not in cache[par]:
                cols = [D.varnames.index(x) for x in (par, ) + par_sea]
                record_score+=1
                cache[par][par_sea] = local_score_from_storage(cols, score_dict)
            candidate[var].remove(par)
            candidate[par].remove(var)
        for par in D.d_prior[var]['par']:
            candidate[var].remove(par)
            candidate[par].remove(var)
    diff = 1
    while diff > 0:
        diff = 0
        edge_candidate = []
        for vi in D.varnames:
            for vj in candidate[vi]:
                # attempt to add edges vi->vj
                if vj not in D.d_prior[vi]['par']:
                    cyc_flag = check_cycle(vi, vj, dag)
                    if not cyc_flag:
                        par_sea = tuple(sorted(dag[vj]['par'] + [vi]))
                        if par_sea not in cache[vj]:
                            cols = [D.varnames.index(x) for x in (vj, ) + par_sea]
                            record_score+=1
                            cache[vj][par_sea] = local_score_from_storage(cols, score_dict)
                        diff_temp = cache[vj][par_sea] - cache[vj][tuple(dag[vj]['par'])]
                        if diff_temp - diff > 1e-10:
                            diff = diff_temp
                            edge_candidate = [vi, vj, 'a']
            for par_vi in dag[vi]['par']:
                if (par_vi not in D.a_prior[vi]['par']) and (vi not in D.r_prior[par_vi]['par']):
                    # attempt to reverse edges from vi<-par_vi to vi->par_vi
                    if par_vi in candidate[vi]:
                        cyc_flag = check_cycle(vi, par_vi, dag)
                        if not cyc_flag:
                            # if par_vi=='FIO2' and vi=='PVSAT':
                            #     a=1
                            par_sea_par_vi = tuple(sorted(dag[par_vi]['par'] + [vi]))
                            if par_sea_par_vi not in cache[par_vi]:
                                cols = [D.varnames.index(x) for x in (par_vi, ) + par_sea_par_vi]
                                record_score+=1
                                cache[par_vi][par_sea_par_vi] = local_score_from_storage(cols, score_dict)
                            par_sea_vi = tuple([x for x in dag[vi]['par'] if x != par_vi])
                            if par_sea_vi not in cache[vi]:
                                cols = [D.varnames.index(x) for x in (vi, ) + par_sea_vi]
                                record_score+=1
                                cache[vi][par_sea_vi] = local_score_from_storage(cols, score_dict)
                            diff_temp = cache[par_vi][par_sea_par_vi] + cache[vi][par_sea_vi] - cache[par_vi][
                                tuple(dag[par_vi]['par'])] - cache[vi][tuple(dag[vi]['par'])]
                            if diff_temp - diff > 1e-10:
                                diff = diff_temp
                                edge_candidate = [vi, par_vi, 'r']
                    # attempt to delete edges vi<-par_vi
                    par_sea = tuple([x for x in dag[vi]['par'] if x != par_vi])
                    if par_sea not in cache[vi]:
                        cols = [D.varnames.index(x) for x in (vi, ) + par_sea]
                        record_score+=1
                        cache[vi][par_sea] = local_score_from_storage(cols, score_dict)
                    diff_temp = cache[vi][par_sea] - cache[vi][tuple(dag[vi]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [par_vi, vi, 'd']
        # if edge_candidate[0]==D.varnames[19] and edge_candidate[1]==D.varnames[18]:
        #     a=1
        if edge_candidate:
            if edge_candidate[-1] == 'a':
                dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                candidate[edge_candidate[0]].remove(edge_candidate[1])
                if edge_candidate[0] in candidate[edge_candidate[1]]:
                    candidate[edge_candidate[1]].remove(edge_candidate[0])
            elif edge_candidate[-1] == 'r':
                dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
            elif edge_candidate[-1] == 'd':
                dag[edge_candidate[1]]['par'].remove(edge_candidate[0])
                if edge_candidate[1] in D.pc[edge_candidate[0]]:
                    candidate[edge_candidate[0]].append(edge_candidate[1])
                candidate[edge_candidate[1]].append(edge_candidate[0])
    D.dag=dag
    print(record_score)
    return dag

