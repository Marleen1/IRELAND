from gurobipy import *
import numpy as np
import sys
import time
import json
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
from pathlib import Path
import datetime
from collections import defaultdict
import multiprocessing as mp

def load_data(datafolder,dataset):
    X = np.genfromtxt(datafolder+'/'+dataset+'_X.csv',delimiter=',',dtype=int)
    y = np.genfromtxt(datafolder+'/'+dataset+'_y.csv',delimiter=',',dtype=int)

    N = X.shape[0]
    P = X.shape[1]

    N_cases = np.sum(y)
    N_controls = N - N_cases

    w = np.zeros(N,dtype=float)
    w[np.where(y==1)[0]] = N_controls / (N_cases + N_controls)
    w[np.where(y==0)[0]] = N_cases / (N_cases + N_controls)
    w = np.ones(N)
    return (X,y,w,N,P)

def master(X,y,w,N,K,M,threads,timelimit):
    t = np.genfromtxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/t.csv',delimiter=';',dtype=float)
    if len(t.shape) > 1:
        K_ = t.shape[0]
    else:
        K_ = 1
        t = np.expand_dims(t,axis=0)
    samples = range(N)
    clauses = range(K_)

    m = Model(env=env)

    q = m.addVars(clauses, vtype=GRB.BINARY, name='q')
    yhat = m.addVars(samples, vtype=GRB.BINARY, name='yhat')

    # yhat_n - sum_k[t_nk] <= 0
    m.addConstrs(yhat[n] - quicksum(t[k,n]*q[k] for k in clauses) <= 0 for n in samples) 

    # - yhat_n + t_nk >= 0 for all k
    m.addConstrs(-yhat[n] + t[k,n]*q[k] <= 0 for k in clauses for n in samples)

    # sum_[k] q_k <= K
    m.addConstr(quicksum(q[k] for k in clauses) <= K)

    # minimize sum_[n: y_n = 0] yhat[n] + sum_[n:y_n = 1] (1-yhat_n)
    m.setObjective(quicksum(w[n] * yhat[n] for n in samples if y[n] == 0) + quicksum(w[n] * (1-yhat[n]) for n in samples if y[n]==1), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = threads
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/ModelStatus.txt','a') as f:
            f.write('Master problem: ')
            if m.status == GRB.Status.OPTIMAL:
                f.write('OPTIMAL\n')
            elif m.status == GRB.Status.TIME_LIMIT:
                f.write('TIME_LIMIT\n')
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Runtimes.txt','a') as f:
            f.write('Master problem: ')
            f.write(str(time.time()-starttime)+'\n')
        for var in m.getVars():
            try:
                test = var.X
            except:
                stop = 1
                continue
            break

        # Get the optimal/best solution so far
        qnew = np.empty((K_),dtype=int)
        ynew = np.empty((N),dtype=int)
        for var in m.getVars():
            splitname = var.VarName.split('[')
            if ',' in splitname[1]:
                splitname = [splitname[0],splitname[1].split(',')[0],splitname[1].split(',')[1]]
                splitname = [splitname[0],splitname[1],splitname[2].split(']')[0]]
            else:
                splitname = [splitname[0],splitname[1].split(']')[0]]
            if splitname[0] == 'q':
                qnew[int(splitname[1])] = var.X
            elif splitname[0] == 'yhat':
                ynew[int(splitname[1])] = var.X

        np.savetxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/q.csv',qnew,delimiter=';',fmt='%i')
        np.savetxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/y.csv',ynew,delimiter=';',fmt='%i')
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/ObjectiveValue_master.csv','a') as f:
            f.write(str(m.getObjective().getValue())+'\n')
        obj = m.getObjective().getValue()

    return (obj,ynew)

def subproblem(a):
    Nselect = a[0]
    iter = a[2]
    UB = a[3]
    initializing = a[4]
    X = X_[Nselect,:]
    y = y_[Nselect]

    N = X.shape[0]
    P = P_
    w = np.ones((N))

    if initializing == 0:
        s_prev = np.genfromtxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/s.csv',delimiter=';',dtype=float)
        if len(s_prev.shape) == 1:
            s_prev = np.expand_dims(s_prev,axis=0)
        K_prev = s_prev.shape[0]
        clauses = range(K_prev)

    # Solve sub problem for selected samples
    samples = range(N)
    features = range(P)

    m = Model()

    s = m.addVars(features, vtype=GRB.BINARY, name='s')
    t = m.addVars(samples, vtype=GRB.BINARY, name='t')

    # sum_p[s_pk] <= M for k in K
    m.addConstr(quicksum(s[p] for p in features) <= M)
    m.addConstr(quicksum(s[p] for p in features) >= 1)

    # sum_p[(1-X_np)*s_pk + P*t_nk <= P for k in K, for n in N1
    m.addConstrs(quicksum((1-X[n,p])*s[p] for p in features) + P*t[n] <= P for n in samples if y[n] == 1)

    # sum_p[(1-X_np)*s_pk + t_nk >= 1 for k in K, for n in N1
    A = np.hstack((np.array((1-X)),np.eye(len(y))))
    m.addMConstr(np.array(A),None,'>',np.ones((len(y),)))

    #sum_[n: y[n] ==0] t[n] <= UB
    m.addConstr(quicksum(t[n] for n in samples if y[n] == 0) <= UB)

    if initializing == 0:
        m.addConstrs(quicksum(s[p] for p in features if s_prev[k,p] == 1) + quicksum(1-s[p] for p in features if s_prev[k,p] == 0) <= P-1 for k in clauses)

    #m.setObjective(quicksum(-t[n] for n in samples if y[n] == 1) + (1/N**2) * quicksum(t[n] for n in samples if y[n] == 0), GRB.MINIMIZE)
    m.setObjective(quicksum(w[n]*(1-t[n]) for n in samples if y[n] == 1), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = 4
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        if m.status == GRB.Status.OPTIMAL:
            status = 'OPTIMAL\n'
        elif m.status == GRB.Status.TIME_LIMIT:
            status = 'TIME_LIMIT\n'
        runtime = str(time.time()-starttime)+'\n'
        for var in m.getVars():
            try:
                test = var.X
            except:
                stop = 1
                continue
            break

        obj = m.getObjective().getValue()

        snew = np.empty((P),dtype=int)
        for var in m.getVars():
            splitname = var.VarName.split('[')
            if ',' in splitname[1]:
                splitname = [splitname[0],splitname[1].split(',')[0],splitname[1].split(',')[1]]
                splitname = [splitname[0],splitname[1],splitname[2].split(']')[0]]
            else:
                splitname = [splitname[0],splitname[1].split(']')[0]]
            if splitname[0] == 's':
                snew[int(splitname[1])] = var.X

        tnew = np.floor(np.matmul(X_,snew)/np.sum(snew)).astype(int)

        result = {}
        result['iter'] = iter
        result['s'] = snew
        result['t'] = tnew
        result['runtime'] = runtime
        result['status'] = status
        result['obj'] = str(m.getObjective().getValue())+'\n'
    else:
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Messages.txt','a') as f:
            f.write('Error: ')
            f.write(str(m.status))
    return result

def collect_results(result):
    global results_collection
    iter = result['iter']
    del result['iter']
    results_collection[iter] = {}
    results_collection[iter] = result
    return

def process_results(results_coll):
    for k,v in results_coll.items():
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/ModelStatus.txt','a') as f:
            f.write('Sub problem: ')
            f.write(v['status'])
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Runtimes.txt','a') as f:
            f.write('Sub problem: ')
            f.write(v['runtime'])
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/t.csv','a') as f:
            f.write(';'.join(v['t'].astype(str))+'\n')
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/s.csv','a') as f:
            f.write(';'.join(v['s'].astype(str))+'\n')
        with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/ObjectiveValue_sub.csv','a') as f:
            f.write(v['obj'])

def generate_firstPool(X,y,w,N,P,Nsub,iters,UB):
    global results_collection

    po = mp.Pool(processes = threads)
    cases = [n for n in range(N) if y[n] == 1]
    controls = [n for n in range(N) if y[n] == 0]
    if Nsub >= len(cases):
        res = subproblem([[n for n in range(N)],np.ones((N)).tolist(),0,UB,1])
        collect_results(res)
        process_results(results_collection)
    else:
        Nselect = np.random.choice(cases,Nsub).tolist()
        for iter in range(iters):
            Nselect = np.random.choice(cases,Nsub).tolist()
            po.apply_async(subproblem, args=([Nselect+controls,np.ones((Nsub+len(controls))).tolist(),iter,UB,1],), callback=collect_results)
        po.close()
        po.join()
        process_results(results_collection)
    return

def keep_unique_rules_only(X):
    s = np.genfromtxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/s.csv',delimiter=';',dtype=int)
    if len(s.shape) == 1:
        s = np.expand_dims(s,axis=0)
    s = np.unique(s, axis=0)
    np.savetxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/s.csv',s,delimiter=';',fmt='%i')
    for row in range(s.shape[0]):
        s[row,:] = np.true_divide(s[row,:],np.sum(s[row,:]))
    t = np.transpose(np.floor(np.matmul(X,np.transpose(s))))
    np.savetxt(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/t.csv',t,delimiter=';',fmt='%i')
    return

def init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

def add_rules(X,y,w,N,P,Nsub,iters,ynew,UB):
    global results_collection

    fn = [n for n in range(N) if y[n] == 1 and ynew[n] == 0]
    controls = [n for n in range(N) if y[n] == 0]
    cases = [n for n in range(N) if y[n] == 1]
    master_necessary = 1
    obj = float('inf')
    if len(fn) > Nsub:
        results_collection = defaultdict(dict)
        po = mp.Pool(processes = threads)
        for iter in range(iters):
            fn_select = np.random.choice(fn,int(np.floor(Nsub))).tolist()
            po.apply_async(subproblem, args=([fn_select+controls,np.ones((len(fn_select)+len(controls))).tolist(),iter,UB,0],), callback=collect_results)
        po.close()
        po.join()
        process_results(results_collection)
    elif len(fn) > 0:
        results_collection = defaultdict(dict)
        po = mp.Pool(processes = threads)
        for iter in range(1):
            po.apply_async(subproblem, args=([fn+controls,np.ones((len(fn)+len(controls))).tolist(),iter,UB,0],), callback=collect_results)
        po.close()
        po.join()
        process_results(results_collection)
    else:
        print('Length fn is zero: '+str(len(fn)))
        master_necessary = 0
    return

if __name__ == "__main__":
    overall_starttime = time.time()
    homefolder = os.path.expanduser("~")

    dataset = sys.argv[1]
    datafolder = sys.argv[2]
    resultfolder = sys.argv[3]
    threads = int(sys.argv[4])
    timelimit = int(sys.argv[5])
    Nsub_ = int(sys.argv[6])
    UB_ = int(sys.argv[9])
    K = int(sys.argv[7])
    M = int(sys.argv[8])


    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    global results_collection
    results_collection = defaultdict(dict)

    global X_
    global y_
    global w_
    global N_
    global P_

    (X_,y_,w_,N_,P_) = load_data(datafolder,dataset)

    if not os.path.isdir(datafolder):
        os.system('mkdir '+datafolder)
    if not os.path.isdir(datafolder+'/'+dataset):
        os.system('mkdir '+datafolder+'/'+dataset)
    if not os.path.isdir(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)):
        os.system('mkdir '+datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_))

    generate_firstPool(X_,y_,w_,N_,P_,Nsub_,sizeFirstPool,UB_)

    stop = 0
    ob_prev = N_
    count_noImprovement = 0
    while stop == 0:
        ob, yn = master(X_,y_,w_,N_,K,M,threads,timelimit)
        if ob < 0.000001:
            with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Messages.txt','a') as f:
                f.write('Objective is 0.\n')
            stop = 1
        elif ob > ob_prev:
            with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Messages.txt','a') as f:
                f.write('The objective value has increased, which should not happen.\n')
            stop = 1
        else:
            if (ob_prev - ob)/ob_prev < 0.0001:
                with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Messages.txt','a') as f:
                    f.write('Objective improvement small: '+str(np.round((ob_prev - ob)/ob_prev,6))+'.\n')
                count_noImprovement += 1
            else:
                count_noImprovement = 0
        ob_prev = ob
        if count_noImprovement > 4:
            stop = 1
        if stop == 0:
            add_rules(X_,y_,w_,N_,P_,Nsub_,addToPool,yn,UB_)
    with open(datafolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'_UB'+str(UB_)+'/Runtimes.txt','a') as f:
        f.write('Overall: '+str(time.time()-overall_starttime))
