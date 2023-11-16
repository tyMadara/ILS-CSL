import copy
import json
import random
from collections import Counter
from os import path

import numpy as np
import pandas as pd
import pydot


def dict2array(A: dict,varnames) -> np.ndarray:
    g = np.zeros([len(A),len(A)])
    for key in A:
        for pa in A[key]['par']: 
            g[varnames.index(pa),varnames.index(key)]=1
    return g

def array2dict(A:np.ndarray,varnames):
    dag={}
    n,m=A.shape
    for i in range(n):
        dag[varnames[i]]={}
        dag[varnames[i]]['par'] = []
        dag[varnames[i]]['nei'] = []
    for i in range(n):
        for j in range(m):
            if A[i,j]==1:
                dag[varnames[j]]['par'].append(varnames[i])
    return dag

def stat_edge(A: dict):
    num=0
    for key in A:
        for pa in A[key]['par']: 
            num+=1
    return num

def check_cycle(vi, vj, dag):
    # whether adding or orientating edge vi->vj would cause cycle. In other words, this function check whether there is a direct path from vj to vi except the possible edge vi<-vj
    underchecked = [x for x in dag[vi]['par'] if x != vj]
    checked = []
    cyc_flag = False
    while underchecked:
        if cyc_flag:
            break
        underchecked_copy = list(underchecked)
        for vk in underchecked_copy:
            if dag[vk]['par']:
                if vj in dag[vk]['par']:
                    cyc_flag = True
                    break
                else:
                    for key in dag[vk]['par']:
                        if key not in checked + underchecked:
                            underchecked.append(key)
            underchecked.remove(vk)
            checked.append(vk)
    return cyc_flag

class DAG:
    def __init__(self,varnames) -> None:
        self.varnames = varnames
        self.n=len(self.varnames)
        self.truth_graph=np.zeros([self.n,self.n])
        self.truth_dag={}
        self.dag={}   
        self.a_prior={}
        self.r_prior={}
        self.d_prior={}
        self.pc={}
        for var in self.varnames:
            self.truth_dag[var] = {}
            self.truth_dag[var]['par'] = []
            self.a_prior[var] = {}
            self.a_prior[var]['par'] = []
            self.d_prior[var] = {}
            self.d_prior[var]['par'] = []
            self.r_prior[var] = {}
            self.r_prior[var]['par'] = []
            self.pc[var] = []
        self.edge_record={}
    
    def clear(self):
        self.dag=None        
        self.a_prior={}
        self.r_prior={}
        self.d_prior={}
        for var in self.varnames:
            self.a_prior[var] = {}
            self.a_prior[var]['par'] = []
            self.d_prior[var] = {}
            self.d_prior[var]['par'] = []
            self.r_prior[var] = {}
            self.r_prior[var]['par'] = []
        self.edge_record={}

    def load_data(self,data,test=None,score=None):
        # Read data to get data, arities and test methods
        if all(data[var].dtype.name == 'category' for var in data):
            self.arities = np.array(data.nunique())
            self.data = data.apply(lambda x: x.cat.codes).to_numpy()
            if test is None:
                self.test = 'g-test'
            if score is None:
                self.score = 'bic'
        elif all(data[var].dtype.name != 'category' for var in data):
            self.arities = None
            self.data=data.to_numpy()
            if test is None:
                self.test = 'z-test'
            if score is None or score == 'bic':
                self.score = 'bic_g'
        else:
            raise Exception('Mixed data is not supported.')

    
    def load_truth_framework(self,path):
        with open(path) as f:
            input_dict=json.load(f) 
            parents_dict=input_dict['parents_dict']
            for key in parents_dict:
                for i in parents_dict[key]:
                    self.truth_graph[self.varnames.index(i),self.varnames.index(key)]=1
            parents_dict={key:{'par':parents_dict[key]} for key in parents_dict}
            self.truth_dag=parents_dict

    def dag2graph(self):
        self.graph=np.zeros([self.n,self.n])
        for key in self.dag:
                for par in self.dag[key]['par']:
                    self.graph[self.varnames.index(par),self.varnames.index(key)]=1
        return self.graph

    def load_truth_nptxt(self,path):
        #Read the real result to get the real matrix and real dag
        self.truth_graph=np.loadtxt(path)
        for i,var in enumerate(self.varnames):
            for j in range(len(self.varnames)):
                if self.truth_graph[i,j]==1:
                    self.truth_dag[self.varnames[j]]['par'].append(var)  
    
    def Add_prior(self,a_prior:list=[],d_prior:list=[]):   
        for prior in a_prior:
            self.a_prior[self.varnames[prior[1]]]['par'].append(self.varnames[prior[0]])
        for prior in d_prior:
            self.d_prior[self.varnames[prior[1]]]['par'].append(self.varnames[prior[0]])
                
    def generate_prior(self,right=10,wrong=0,seed=2,ptype='all',first='right'):
        def directed_edge2array(n:int,L:list)->np.ndarray:
            A=np.zeros([n,n])
            for i in L:
                A[i[0],i[1]]=1
            return A
            
        def array2directed_edge(A:np.ndarray)->list:
            a,b=np.where(A!=0)
            return list(zip(a,b))

        def array2no_edge(A:np.ndarray)->list:
            a,b=np.where(A==0)
            return list(zip(a,b))

        if first=='right':
            self.clear()
            A=dict2array(self.truth_dag,self.varnames)
            random.seed(seed)
            L1=array2directed_edge(A)
            L2=array2no_edge(A)
            history1=random.sample(L1,right)
            history2=[]
            history3=[]
            random.shuffle(L2)
            self.a_prior=array2dict(directed_edge2array(A.shape[0],history1),self.varnames)
            while wrong>0:
                if len(L2)==0:
                    print("Too few desirable edges, sampling process stopped.")
                    break
                i,j=L2.pop()
                #The wrong edge needs to meet three conditions: it cannot refer to itself, it cannot violate known edges, and it cannot form a ring
                if i==j or (j,i) in history1+history2+history3:
                    continue
                if not check_cycle(self.varnames[i],self.varnames[j],self.a_prior):
                    flag=0
                    if ptype=='all':
                        flag=1
                    elif ptype=='direct':
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par']:
                            flag=1
                    elif ptype=='irancestor':
                        if check_cycle(self.varnames[i],self.varnames[j],self.truth_dag):
                            flag=1
                    elif ptype=='indirect':
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par'] or check_cycle(self.varnames[i],self.varnames[j],self.truth_dag):
                            flag=1
                    elif ptype=='ancestor':
                        if check_cycle(self.varnames[j],self.varnames[i],self.truth_dag):
                            flag=1
                    elif ptype=='irrelevant':
                        if (self.varnames[j] not in self.truth_dag[self.varnames[i]]['par']) and (not check_cycle(self.varnames[j],self.varnames[i],self.truth_dag)) and (not check_cycle(self.varnames[i],self.varnames[j],self.truth_dag)):
                            flag=1
                    if flag:
                        wrong-=1
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par']:
                            history2.append((i,j))
                        else:
                            history3.append((i,j))
                        self.a_prior[self.varnames[j]]['par'].append(self.varnames[i])
            #self.prior is a_prior+r_prior
            self.prior=copy.deepcopy(self.a_prior)
            self.history1=history1
            self.history2=history2
            self.history3=history3
        elif first=='wrong':
            self.clear()
            A=dict2array(self.truth_dag,self.varnames)
            random.seed(seed)
            L1=array2directed_edge(A)
            L2=array2no_edge(A)
            history1=[]
            history2=[]
            history3=[]
            random.shuffle(L2)
            random.shuffle(L1)
            while wrong>0:
                if len(L2)==0:
                    print("Too few desirable edges, sampling process stopped.")
                    break
                i,j=L2.pop()
                #The wrong edge needs to meet three conditions: it cannot refer to itself, it cannot violate known edges, and it cannot form a ring
                if i==j or (j,i) in history1+history2+history3:
                    continue
                if not check_cycle(self.varnames[i],self.varnames[j],self.a_prior):
                    flag=0
                    if ptype=='all':
                        flag=1
                    elif ptype=='direct':
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par']:
                            flag=1
                    elif ptype=='irancestor':
                        if check_cycle(self.varnames[i],self.varnames[j],self.truth_dag):
                            flag=1
                    elif ptype=='indirect':
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par'] or check_cycle(self.varnames[i],self.varnames[j],self.truth_dag):
                            flag=1
                    elif ptype=='ancestor':
                        if check_cycle(self.varnames[j],self.varnames[i],self.truth_dag):
                            flag=1
                    elif ptype=='irrelevant':
                        if (self.varnames[j] not in self.truth_dag[self.varnames[i]]['par']) and (not check_cycle(self.varnames[j],self.varnames[i],self.truth_dag)) and (not check_cycle(self.varnames[i],self.varnames[j],self.truth_dag)):
                            flag=1
                    if flag:
                        wrong-=1
                        if self.varnames[j] in self.truth_dag[self.varnames[i]]['par']:
                            history2.append((i,j))
                        else:
                            history3.append((i,j))
                        self.a_prior[self.varnames[j]]['par'].append(self.varnames[i])
            while right>0:
                if len(L1)==0:
                    print("Too few desirable edges, sampling process stopped")
                    break
                i,j=L1.pop()
                if i==j or (j,i) in history1+history2+history3:
                    continue
                if not check_cycle(self.varnames[i],self.varnames[j],self.a_prior):
                    right-=1
                    history1.append((i,j))
                    self.a_prior[self.varnames[j]]['par'].append(self.varnames[i])

            #self.prior is a_prior+r_prior
            self.prior=copy.deepcopy(self.a_prior)
            self.history1=history1
            self.history2=history2
            self.history3=history3



    def check_prior(self,vi,vj):
        if vj in self.a_prior[vi]['par']:
            return True
        if vi in self.r_prior[vj]['par']:
            return True
        return False

    def stat_subring_prior(self):
        record=[]
        for key in self.dag:
            for par in self.dag[key]['par']:
                for ppar in self.dag[par]['par']:
                    if ppar in self.dag[key]['par']:
                        #If the subring contains at least two priors, it is not counted, because most of them are priors
                        if len(self.history1)+len(self.history2)+len(self.history3)>=self.n:
                            if self.check_prior(key,par) and self.check_prior(key,ppar) and self.check_prior(par,ppar):
                                pass
                            else:
                                record.append([key,par,ppar])
                        else:
                            if self.check_prior(key,par) and self.check_prior(key,ppar):
                                pass
                            elif self.check_prior(key,ppar) and self.check_prior(par,ppar):
                                pass
                            elif self.check_prior(key,par) and self.check_prior(par,ppar):
                                pass
                            else:
                                record.append([key,par,ppar])
        
        self.subring=record
        count=Counter()
        for subring in record:
            count[(subring[0],subring[1],)]+=1
            count[(subring[1],subring[2],)]+=1
            count[(subring[0],subring[2],)]+=1
        stat_subring=[[count[key],key] for key in count]
        stat_subring.sort(reverse=True)
        self.stat_subring=stat_subring
        return record

    def stat_subring_dag(self,dag=None):
        # Count all subrings
        record=[]
        if dag==None:
            dag=self.dag
        for key in dag:
            for par in dag[key]['par']:
                for ppar in dag[par]['par']:
                    if ppar in dag[key]['par']:
                        record.append([key,par,ppar])
        return record

    def update_prior(self,threshold=0):
        for subring in self.stat_subring:
            if subring[0]>threshold:
                var=subring[1][0]
                par=subring[1][1]
                #Record the number of sub-rings in the three cases a priori, and the last digit is used to record whether it has been counted once
                flag=0
                if ((var,par) not in self.edge_record) and ((par,var) not in self.edge_record):
                    self.edge_record[(var,par)]=[0,0,0]
                if (var,par) in self.edge_record and (self.edge_record[(var,par)][-1]==0):
                    flag=1
                if (par,var) in self.edge_record and (self.edge_record[(par,var)][-1]==0):
                    flag=1
                if flag:
                    if par in self.a_prior[var]['par']:
                        self.a_prior[var]['par'].remove(par)
                        self.prior[var]['par'].remove(par)
                        self.edge_record[(var,par)][0]=subring[0]
                        if not check_cycle(var,par,self.prior):
                            self.r_prior[var]['par'].append(par)
                            self.prior[par]['par'].append(var)
                    elif var in self.r_prior[par]['par']:
                        self.edge_record[(par,var)][1]=subring[0]
                        self.edge_record[(par,var)][-1]=1
                        #Choose the last state based on the result of the triplicate
                        if np.min(self.edge_record[(par,var)])>1:
                            self.r_prior[par]['par'].remove(var)
                            self.prior[var]['par'].remove(par)
                            self.d_prior[par]['par'].append(var)
                        elif np.argmin(self.edge_record[(par,var)])==0:
                            self.r_prior[par]['par'].remove(var)
                            self.prior[var]['par'].remove(par)
                            self.a_prior[par]['par'].append(var)
                            self.prior[par]['par'].append(var)

    def to_pydot(self, dag=None,title: str = "", dpi: float = 200):

        nodes = self.varnames
        pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
        pydot_g.obj_dict["attributes"]["dpi"] = dpi
        for i, node in enumerate(nodes):
            pydot_g.add_node(pydot.Node(i, label=node))
            pydot_g.add_node(pydot.Node(i, label=node))

        if dag==None:
            dag=self.dag
        for key in dag:
            for par in dag[key]['par']:
                node1_id=self.varnames.index(key)
                node2_id=self.varnames.index(par)
                dot_edge = pydot.Edge(node2_id, node1_id, dir='both', arrowtail='none',
                                        arrowhead='normal')
                if par in self.a_prior[key]['par']:
                    if par in self.truth_dag[key]['par']:
                        # Green is correct prior
                        dot_edge.obj_dict["attributes"]["color"] = "green"
                    elif key in self.truth_dag[par]['par']:
                        # Red is the opposite prior
                        dot_edge.obj_dict["attributes"]["color"] = "red"
                    elif check_cycle(par,key,self.truth_dag):
                        # Purple is the opposite prior of the ancestor
                        dot_edge.obj_dict["attributes"]["color"] = "purple"
                    else:
                        # Grey is the prior of the ancestor or irrelevant prior
                        dot_edge.obj_dict["attributes"]["color"] = "grey"
                elif key in self.r_prior[par]['par']:
                    dot_edge.obj_dict["attributes"]["style"] = "dotted"
                    if par in self.truth_dag[key]['par']:
                        # Green is correct prior
                        dot_edge.obj_dict["attributes"]["color"] = "green"
                    elif key in self.truth_dag[par]['par']:
                        # Red is the opposite prior
                        dot_edge.obj_dict["attributes"]["color"] = "red" 
                    elif check_cycle(par,key,self.truth_dag):
                        # Purple is the opposite prior of the ancestor
                        dot_edge.obj_dict["attributes"]["color"] = "purple"
                    else:
                        # Grey is the prior of the ancestor or irrelevant prior
                        dot_edge.obj_dict["attributes"]["color"] = "grey"
                elif key in self.truth_dag[par]['par']:
                    # Orange is reversed
                    dot_edge.obj_dict["attributes"]["color"] = "orange"
                elif par not in self.truth_dag[key]['par']:
                    # Pink is not
                    dot_edge.obj_dict["attributes"]["color"] = "pink"
                pydot_g.add_edge(dot_edge)
        return pydot_g

    def ShowGraph(self,dag=None):
        import io

        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        pyd = self.to_pydot(dag)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    def SHD(self):
        shd=0
        for key in self.varnames:
            for par in self.dag[key]['par']:
                if (par not in self.truth_dag[key]['par']) and (key not in self.truth_dag[par]['par']):
                    shd+=1
            for par in self.truth_dag[key]['par']:
                if (par not in self.dag[key]['par']) and (key not in self.dag[par]['par']):
                    shd+=1
            for par in self.dag[key]['par']:
                if key in self.truth_dag[par]['par']:
                    shd+=1
        return shd
                
    def miss_extra_reverse_shd(self):
        miss,extra,reverse,shd=0,0,0,0
        for key in self.varnames:
            for par in self.dag[key]['par']:
                if (par not in self.truth_dag[key]['par']) and (key not in self.truth_dag[par]['par']):
                    extra+=1
            for par in self.truth_dag[key]['par']:
                if (par not in self.dag[key]['par']) and (key not in self.dag[par]['par']):
                    miss+=1
            for par in self.dag[key]['par']:
                if key in self.truth_dag[par]['par']:
                    reverse+=1
        shd=miss+extra+reverse
        return [miss,extra,reverse,shd]

    def stat_prior(self):

        rpr,rpw,wpr,wpw=0,0,0,0
        for tmp in self.history1:
            var=self.varnames[tmp[1]]
            par=self.varnames[tmp[0]]
            if par in self.dag[var]['par']:
                rpr+=1
            else:
                rpw+=1
        for tmp in self.history2:
            var=self.varnames[tmp[1]]
            par=self.varnames[tmp[0]]
            if var in self.dag[par]['par']:
                wpr+=1
            elif par not in self.dag[var]['par']:
                wpw+=0.5
                wpr+=0.5
            else:
                wpw+=1
        for tmp in self.history3:
            var=self.varnames[tmp[1]]
            par=self.varnames[tmp[0]]
            if var in self.dag[par]['par']:
                wpw+=1
            elif par in self.dag[par]['par']:
                wpw+=1
            else:
                wpr+=1        
    
        return [rpr,rpw,wpr,wpw]


if __name__=='__main__':
    network = 'alarm'
    datasize = 1000
    filename = '../data/TestData/'+network + '_' + str(datasize) + '.csv'
    if path.isfile(filename):
        data = pd.read_csv(filename, dtype='category') # change dtype = 'float64'/'category' if data is continuous/categorical
        D=DAG(list(data.columns))
        D.load_data(data,test=None,score=None)
    D.load_truth('../data/GraphData/alarm_framework.json')
                
