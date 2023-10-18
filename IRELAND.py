from gurobipy import *
import numpy as np
import sys
import time
import json
from pathlib import Path
import datetime
from collections import defaultdict
import multiprocessing as mp
from copy import copy

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

    return (X,y,w,N,P,N_controls)

def master(X,y,w,N,K,M,threads,timelimit):
    global UBlist
    t = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UBlist[0]).replace('.','p')+'/t.csv',delimiter=';',dtype=float)
    for UB_ in UBlist[1:]:
        t_ = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_).replace('.','p')+'/t.csv',delimiter=';',dtype=float)
        if len(t_.shape) == 1:
            t_ = np.expand_dims(t_,axis=0)
        t = np.vstack((t,t_))

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
    m.addConstrs(yhat[n] - quicksum(t[k,n]*q[k] for k in clauses) <= 0 for n in samples) # if y[n] == 1)

    # - yhat_n + t_nk >= 0 for all k
    m.addConstrs(-yhat[n] + t[k,n]*q[k] <= 0 for k in clauses for n in samples) # if y[n] == 0)

    # sum_[k] q_k <= K
    m.addConstr(quicksum(q[k] for k in clauses) <= K)

    # - yhat_n >= 0
    m.addConstrs(-yhat[n] <= 0 for n in samples) # if y[n] == 0)

    # minimize sum_[n: y_n = 0] yhat[n] + sum_[n:y_n = 1] (1-yhat_n)
    m.setObjective(quicksum(w[n] * yhat[n] for n in samples if y[n] == 0) + quicksum(w[n] * (1-yhat[n]) for n in samples if y[n]==1), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = threads
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/ModelStatus.txt','a') as f:
            f.write('Master problem: ')
            if m.status == GRB.Status.OPTIMAL:
                f.write('OPTIMAL\n')
            elif m.status == GRB.Status.TIME_LIMIT:
                f.write('TIME_LIMIT\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/Runtimes.txt','a') as f:
            f.write('Master problem: ')
            f.write(str(time.time()-starttime)+'\n')
        for var in m.getVars():
            try:
                test = var.X
            except:
                stop = 1
                continue
            break

        # Get the optimal/best so far solution
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

        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/q.csv',qnew,delimiter=';',fmt='%i')
        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/y.csv',ynew,delimiter=';',fmt='%i')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/ObjectiveValue_master.csv','a') as f:
            f.write(str(m.getObjective().getValue())+'\n')
        obj = m.getObjective().getValue()

    return

def masterUB(UB):
    t = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UBlist[0]).replace('.','p')+'/t.csv',delimiter=';',dtype=float)
    for UB_ in UBlist[1:]:
        t_ = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_).replace('.','p')+'/t.csv',delimiter=';',dtype=float)
        if len(t_.shape) == 1:
            t_ = np.expand_dims(t_,axis=0)
        t = np.vstack((t,t_))

    if len(t.shape) > 1:
        K_ = t.shape[0]
    else:
        K_ = 1
        t = np.expand_dims(t,axis=0)
    samples = range(N_)
    clauses = range(K_)

    m = Model(env=env)

    q = m.addVars(clauses, vtype=GRB.BINARY, name='q')
    yhat = m.addVars(samples, vtype=GRB.BINARY, name='yhat')

    # yhat_n - sum_k[t_nk] <= 0
    m.addConstrs(yhat[n] - quicksum(t[k,n]*q[k] for k in clauses) <= 0 for n in samples) # if y[n] == 1)

    # - yhat_n + t_nk >= 0 for all k
    m.addConstrs(-yhat[n] + t[k,n]*q[k] <= 0 for k in clauses for n in samples) # if y[n] == 0)

    # sum_[k] q_k <= K
    m.addConstr(quicksum(q[k] for k in clauses) <= K)

    #sum_[n: y[n] ==0] t[n] <= UB
    m.addConstr(quicksum(yhat[n] for n in samples if y_[n] == 0) <= UB*N_controls_)

    # minimize sum_[n:y_n = 1] (1-yhat_n)
    m.setObjective(quicksum(1-yhat[n] for n in samples if y_[n]==1), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = threads
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/ModelStatus.txt','a') as f:
            f.write('Master problem: ')
            if m.status == GRB.Status.OPTIMAL:
                f.write('OPTIMAL\n')
            elif m.status == GRB.Status.TIME_LIMIT:
                f.write('TIME_LIMIT\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Runtimes.txt','a') as f:
            f.write('Master problem: ')
            f.write(str(time.time()-starttime)+'\n')
        for var in m.getVars():
            try:
                test = var.X
            except:
                stop = 1
                continue
            break

        # Get the optimal/best so far solution
        qnew = np.empty((K_),dtype=int)
        ynew = np.empty((N_),dtype=int)
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

        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/q.csv',qnew,delimiter=';',fmt='%i')
        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/y.csv',ynew,delimiter=';',fmt='%i')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/ObjectiveValue_master.csv','a') as f:
            f.write(str(m.getObjective().getValue())+'\n')

        result = {}
        result['obj'] = m.getObjective().getValue()
        result['ynew'] = ynew
        result['UB'] = UB
    return result

def subproblem(a):
    iter = a[0]
    UB = a[1]
    initializing = a[2]

    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
        f.write('Calling sub problem for UB '+str(UB)+'.\n')

    controls = np.genfromtxt(runfolder+'/Controls.csv',delimiter=',',dtype=int)
    selection = np.genfromtxt(runfolder+'/Nselect_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',delimiter=',',dtype=int)
    w = np.genfromtxt(runfolder+'/Weights_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',delimiter=',',dtype=float)

    Nselect = np.hstack((controls,selection))

    X = X_[Nselect,:]
    y = y_[Nselect]

    N = X.shape[0]
    P = P_

    if initializing == 0:
        s_prev = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UBlist[0]).replace('.','p')+'/s.csv',delimiter=';',dtype=float)
        if len(s_prev.shape) == 1:
            s_prev = np.expand_dims(s_prev,axis=0)

        for UB_ in UBlist[1:]:
            s_prev_ = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_).replace('.','p')+'/s.csv',delimiter=';',dtype=float)
            if len(s_prev_.shape) == 1:
                s_prev_ = np.expand_dims(s_prev_,axis=0)
            s_prev = np.vstack((s_prev,s_prev_))
        K_prev = s_prev.shape[0]
        clauses = range(K_prev)

    # Solve sub problem for selected samples
    samples = range(N)
    features = range(P)

    m = Model(env=env)

    s = m.addVars(features, vtype=GRB.BINARY, name='s')
    t = m.addVars(samples, vtype=GRB.BINARY, name='t')

    # sum_p[s_pk] <= M for k in K
    m.addConstr(quicksum(s[p] for p in features) <= M)
    m.addConstr(quicksum(s[p] for p in features) >= 1)

    # sum_p[(1-X_np)*s_pk + P*t_nk <= P for k in K, for n in N1
    m.addConstrs(quicksum((1-X[n,p])*s[p] for p in features) + P*t[n] <= P for n in samples if y[n] == 1)

    # sum_p[(1-X_np)*s_pk + t_nk >= 1 for k in K, for n in N1
    A = np.hstack((np.array((1-X)),np.eye(len(y))))
    m.addMConstr(np.array(A),None,'>',0.99999999999*np.ones((len(y),)))

    #sum_[n: y[n] ==0] t[n] <= UB
    m.addConstr(quicksum(t[n] for n in samples if y[n] == 0) <= UB*N_controls_)

    if initializing == 0:
        #sum_[p: s_prev[k,p]==1] s[p] + sum_[p: s_prev[k,p]==0] 1-s[p] <= K_prev-1
        m.addConstrs(quicksum(s[p] for p in features if s_prev[k,p] == 1) + quicksum(1-s[p] for p in features if s_prev[k,p] == 0) <= P-1 for k in clauses)

    #m.setObjective(quicksum(-t[n] for n in samples if y[n] == 1) + (1/N**2) * quicksum(t[n] for n in samples if y[n] == 0), GRB.MINIMIZE)
    m.setObjective(quicksum(w[n]*(1-t[n]) for n in samples if y[n] == 1), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = 4
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
        f.write('Solcount: '+str(m.solcount)+'.\n')
        f.write('Model status: '+str(m.status)+'.\n')

    stop = 0
    count = 1
    if m.SolCount == 0:
        while stop == 0:
            m.optimize()
            if m.SolCount > 0:
                stop = 1
            count += 1
            if count > 5:
                stop = 1

    if not m.SolCount > 0:
        stop = 0
        X_controls = np.squeeze(X_[np.where(y_==0),:])
        cols = 1
        while stop == 0:
            if np.squeeze(np.where(np.sum(X_controls[:,:cols],axis=1)>=cols)).shape[0]/X_controls.shape[0] <= 0.005:
                stop = 1
            else:
                cols += 1
        snew = np.zeros((P),dtype=int)
        for c in range(cols):
            snew[c] = 1
        status = 'Plan B\n'
        tnew = np.floor(np.matmul(X_,snew)/np.sum(snew)).astype(int)
        t_cases = np.squeeze(tnew[np.where(y_==1)])
        w = np.array(w_)
        w_cases = np.squeeze(w[np.where(y_==1)])[0]
        obj = w_cases*np.sum(np.ones((tnew.shape),dtype=int)-tnew)

    runtime = str(time.time()-starttime)+'\n'

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        if m.status == GRB.Status.OPTIMAL:
            status = 'OPTIMAL\n'
        elif m.status == GRB.Status.TIME_LIMIT:
            status = 'TIME_LIMIT\n'
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
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/t_temp.csv','a') as f:
            f.write(';'.join(tnew.astype(str))+'\n')
    else:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
            f.write('Other status, namely: '+str(m.status)+', ')
            f.write('or other solution count: '+str(m.solcount)+'\n')

    result = {}
    result['iter'] = iter
    result['s'] = snew
    result['t'] = tnew
    result['runtime'] = runtime
    result['status'] = status
    result['obj'] = str(obj)+'\n'
    result['UB'] = str(UB).replace('.','p')
    result['UBfloat'] = UB
    return result

def collect_results(result):
    global results_collection
    UB = result['UBfloat']
    iter = result['iter']
    del result['iter']
    del result['UBfloat']
    if not UB in results_collection.keys():
        results_collection[UB] = {}
    results_collection[UB][iter] = {}
    results_collection[UB][iter] = result
    return

def collect_results_masterUB(result):
    global objectives
    global ynew_dict
    UB_ = result['UB']
    ynew_dict[UB_] = result['ynew']
    objectives[UB_] = result['obj']
    return

def process_results(results_coll):
    global UBlist_stillrunning
    for UB_fl in UBlist_stillrunning:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_fl).replace('.','p')+'/Messages.txt','a') as f:
            f.write('Step 1 writing results\n')
        for k,v in results_coll[UB_fl].items():
            UB = v['UB']
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_fl).replace('.','p')+'/Messages.txt','a') as f:
                f.write('Step 2 writing results\n')
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/ModelStatus.txt','a') as f:
                f.write('Sub problem: ')
                f.write(v['status'])
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Runtimes.txt','a') as f:
                f.write('Sub problem: ')
                f.write(v['runtime'])
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_fl).replace('.','p')+'/Messages.txt','a') as f:
                f.write('Step 3 writing results\n')
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/t.csv','a') as f:
                f.write(';'.join(v['t'].astype(str))+'\n')
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB_fl).replace('.','p')+'/Messages.txt','a') as f:
                f.write('Step 4 writing results\n')
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/s.csv','a') as f:
                f.write(';'.join(v['s'].astype(str))+'\n')
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/ObjectiveValue_sub.csv','a') as f:
                f.write(v['obj'])

def generate_firstPool(X,y,w,N,P,Nsub,UBlist):
    global results_collection

    cases = [n for n in range(N) if y[n] == 1]
    controls = [n for n in range(N) if y[n] == 0]
    np.savetxt(runfolder+'/Controls.csv',controls,delimiter=',',fmt='%i')
    Nselect = np.random.choice(cases,Nsub).tolist()
    np.savetxt(runfolder+'/Nselect_'+str(0.005).replace('.','p')+'_'+str(0)+'.csv',Nselect,delimiter=',',fmt='%i')
    np.savetxt(runfolder+'/Weights_'+str(0.005).replace('.','p')+'_'+str(0)+'.csv',np.ones((Nsub+len(controls))).tolist(),delimiter=',',fmt='%s')
    subproblem([0,0.005,1])
    po = mp.Pool(processes = threads)
    for UB_frac in UBlist:
        Nselect = np.random.choice(cases,Nsub).tolist()
        np.savetxt(runfolder+'/Nselect_'+str(UB_frac).replace('.','p')+'_'+str(iter)+'.csv',Nselect,delimiter=',',fmt='%i')
        np.savetxt(runfolder+'/Weights_'+str(UB_frac).replace('.','p')+'_'+str(iter)+'.csv',np.ones((Nsub+len(controls))).tolist(),delimiter=',',fmt='%s')
        po.apply_async(subproblem, args=([iter,UB_frac,1],), callback=collect_results)
    po.close()
    po.join()
    starttime_ = time.time()
    process_results(results_collection)
    os.system('/bin/rm -r '+runfolder+'/Nselect_*')
    os.system('/bin/rm -r '+runfolder+'/Weights_*')
    return

def add_rules(X,y,w,N,P,Nsub):
    global results_collection

    po = mp.Pool(processes = threads)
    for UB in UBlist_stillrunning:
        ynew = ynew_dict[UB]
        fn = [n for n in range(N) if y[n] == 1 and ynew[n] == 0]
        controls = [n for n in range(N) if y[n] == 0]
        cases = [n for n in range(N) if y[n] == 1]
        if len(fn) > Nsub:
            results_collection = defaultdict(dict)
            fn_select = np.random.choice(fn,int(np.floor(Nsub))).tolist()
            np.savetxt(runfolder+'/Nselect_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',fn_select,delimiter=',',fmt='%i')
            np.savetxt(runfolder+'/Weights_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',np.ones((len(controls)+len(fn_select))),delimiter=',',fmt='%s')
            po.apply_async(subproblem, args=([iter,UB,0],), callback=collect_results)
        elif len(fn) > 0:
            results_collection = defaultdict(dict)
            np.savetxt(runfolder+'/Nselect_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',fn,delimiter=',',fmt='%i')
            np.savetxt(runfolder+'/Weights_'+str(UB).replace('.','p')+'_'+str(iter)+'.csv',np.ones((len(controls)+len(fn))),delimiter=',',fmt='%s')
            po.apply_async(subproblem, args=([iter,UB,0],), callback=collect_results)
    po.close()
    po.join()
    os.system('/bin/rm -r '+runfolder+'/Nselect_*')
    os.system('/bin/rm -r '+runfolder+'/Weights_*')
    process_results(results_collection)
    return

if __name__ == "__main__":
    overall_starttime = time.time()
    homefolder = os.path.expanduser("~")

    dataset = sys.argv[1]
    datafolder = sys.argv[2]
    resultfolder = sys.argv[3]
    runfilesfolder = sys.argv[4]
    threads = int(sys.argv[5])
    timelimit = int(sys.argv[6])
    Nsub_ = int(sys.argv[7])
    K = int(sys.argv[8])
    M = int(sys.argv[9])

    runfolder = runfilesfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)
    if not os.path.isdir(runfilesfolder+'/'+dataset):
        os.system('mkdir '+runfilesfolder+'/'+dataset)
    if not os.path.isdir(runfolder):
        os.system('mkdir '+runfolder)

    global UBlist
    UBlist = [0.005,0.01,0.02,0.03,0.04,0.05]


    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    global X_
    global y_
    global w_
    global N_
    global P_

    starttime_ = time.time()
    (X_,y_,w_,N_,P_,N_controls_) = load_data(datafolder,dataset)

    if not os.path.isdir(resultfolder):
        os.system('mkdir '+resultfolder)
    if not os.path.isdir(resultfolder+'/'+dataset):
        os.system('mkdir '+resultfolder+'/'+dataset)
    if not os.path.isdir(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)):
        os.system('mkdir '+resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_))
    for UB in UBlist:
        if not os.path.isdir(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')):
            os.system('mkdir '+resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p'))

    global results_collection
    results_collection = defaultdict(dict)

    UBlist_stillrunning = UBlist[:]
    starttime_genpool = time.time()
    generate_firstPool(X_,y_,w_,N_,P_,Nsub_,sizeFirstPool,UBlist)

    results_collection = defaultdict(dict)

    stop = 0
    objectives = {}
    count_noImprovement = {}
    global ynew_dict

    for UB in UBlist:
        count_noImprovement[UB] = 0
        objectives[UB] = N_

    i = 0
    threshold = {}
    for UB in UBlist[::-1]:
        threshold[UBlist[i]] = UB
        i += 1

    while stop == 0:
        ynew_dict = {}
        objectives_prev = {}
        for k,v in objectives.items():
            objectives_prev[k] = v
        objectives = {}

        starttime_ = time.time()
        masterUB(0.005)
        po = mp.Pool(processes = threads)
        for UB in UBlist_stillrunning:
            po.apply_async(masterUB, args=(UB,), callback=collect_results_masterUB)
        po.close()
        po.join()
        for UB in UBlist_stillrunning:
            if objectives[UB] < threshold[UB]:
                with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
                    f.write('Objective is below the threshold of '+str(threshold[UB])+'.\n')
                UBlist_stillrunning.remove(UB)
            elif objectives[UB] > objectives_prev[UB]:
                with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
                    f.write('The objective value has increased, which should not happen.\n')
                UBlist_stillrunning.remove(UB)
            else:
                if (objectives_prev[UB] - objectives[UB])/objectives_prev[UB] < 0.0001:
                    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/UB'+str(UB).replace('.','p')+'/Messages.txt','a') as f:
                        f.write('Objective improvement small: '+str(np.round((objectives_prev[UB] - objectives[UB])/objectives_prev[UB],6))+'.\n')
                    count_noImprovement[UB] += 1
                else:
                    count_noImprovement[UB] = 0
            if count_noImprovement[UB] > 1:
                UBlist_stillrunning.remove(UB)

        if len(UBlist_stillrunning) == 0:
            stop = 1
        else:
            add_rules(X_,y_,w_,N_,P_,Nsub_,addToPool)
            results_collection = defaultdict(dict)

    master(X_,y_,w_,N_,K,M,threads,timelimit)

    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'_Nsub'+str(Nsub_)+'/Runtimes.txt','a') as f:
        f.write('Overall: '+str(time.time()-overall_starttime))

    os.system("'"+dataset+"' >> Instances_done.txt")
