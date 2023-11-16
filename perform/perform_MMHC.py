import numpy as np
import pandas as pd

from hc.DAG import DAG
from mmhc.mmhc import mmhc_prior


def mmhc_test(**kwargs):
    filename = f"data/csv/{kwargs['d']}_{kwargs['s']}_{kwargs['r']}.csv"
    data = pd.read_csv(filename, dtype='category') # change dtype = 'float64'/'category' if data is continuous/categorical
    D=DAG(list(data.columns))
    
    D.load_data(data,test=None,score=None)
    D.load_truth_nptxt(f"BN_structure/{kwargs['d']}_graph.txt")
    #D.Add_prior(a_prior=edges)
    
    tmp=np.ones((D.data.shape[1],D.data.shape[1]))
    for prior in kwargs['forb_edges']:
        tmp[prior[0],prior[1]]=0
    tmp[np.diag_indices_from(tmp)] = 0    
    
    for prior in list(zip(*np.where(tmp))):
        D.pc[D.varnames[prior[0]]].append(D.varnames[prior[1]])
    mmhc_prior(D)
    
    D.dag2graph()
    return D.graph
    
