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
    t = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/t.csv',delimiter=';',dtype=float)

    if len(t.shape) > 1:
        K_ = t.shape[0]
    else:
        K_ = 1
        t = np.expand_dims(t,axis=0)
    samples = range(N)
    clauses = range(K_)

    m = Model(env=env)

    q = m.addVars(clauses, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='q')
    yhat = m.addVars(samples, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='yhat')

    # 1 - yhat_n - sum_k[t_nk*q_k] <= 0
    m.addConstrs(- yhat[n] - quicksum(t[k,n]*q[k] for k in clauses) <= -1 for n in samples) #if y[n] == 1)

    # sum_[k] q_k <= K
    m.addConstr(quicksum(q[k] for k in clauses) <= K)

    # minimize sum_[n: y_n = 1] w_n yhat[n] + sum_[n:y_n = 0] w_n t_nk q_k
    m.setObjective(quicksum(w[n] * yhat[n] for n in samples if y[n] == 1) + quicksum(w[n] * t[k,n] * q[k] for k in clauses for n in samples if y[n]==0), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = threads
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    # Process results
    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/ModelStatus.txt','a') as f:
            f.write('Master problem: ')
            if m.status == GRB.Status.OPTIMAL:
                f.write('OPTIMAL\n')
            elif m.status == GRB.Status.TIME_LIMIT:
                f.write('TIME_LIMIT\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Runtimes.txt','a') as f:
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

        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/q.csv',qnew,delimiter=';',fmt='%1.3f')
        np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/y.csv',1-ynew,delimiter=';',fmt='%1.3f')

        counter = 1
        pi = []
        for c in m.getConstrs():
            pi_ = c.Pi
            if counter <= N:
                pi.append(pi_)
            if counter == N:
                np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/pi1.csv',pi,delimiter=';')
            if counter > N:
                np.savetxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/pi2.csv',[pi_],delimiter=';')
            counter += 1

        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/ObjectiveValue_master.csv','a') as f:
            f.write(str(m.getObjective().getValue())+'\n')
        obj = m.getObjective().getValue()

    return obj


def subproblem(X,y,w,N,K,M,threads,timelimit,initialize):
    P = X.shape[1]

    if initialize == 0:
        s_prev = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/s.csv',delimiter=';',dtype=float)
        if len(s_prev.shape) == 1:
            s_prev = np.expand_dims(s_prev,axis=0)
        K_prev = s_prev.shape[0]
        clauses = range(K_prev)
        pi_1 = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/pi1.csv',delimiter=';',dtype=float)
        pi_2 = np.genfromtxt(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/pi2.csv',delimiter=';',dtype=float)

    features = range(P)
    samples = range(N)

    m = Model(env=env)

    s = m.addVars(features, vtype=GRB.BINARY, name='s')
    t = m.addVars(samples, vtype=GRB.BINARY, name='t')

    # sum_p[s_pk] <= M for k in K
    m.addConstr(quicksum(s[p] for p in features) <= M)
    m.addConstr(quicksum(s[p] for p in features) >= 1)

    # sum_p[(1-X_np)*s_pk + P*t_nk <= P for k in K, for n in N1
    m.addConstrs(quicksum((1-X[n,p])*s[p] for p in features) + P*t[n] <= P for n in samples) # if y[n] == 1)

    # sum_p[(1-X_np)*s_pk + t_nk >= 1 for k in K, for n in N1
    A = np.hstack((np.array((1-X)),np.eye(len(y))))
    m.addMConstr(np.array(A),None,'>',0.99999999999*np.ones((len(y),)))

    if initialize == 0:
        m.setObjective(quicksum(w[n]*t[n] for n in samples if y[n] == 0) + quicksum(pi_1[n] * t[n] for n in samples) - pi_2, GRB.MINIMIZE)
    else:
        m.setObjective(quicksum(w[n] * (1-t[n]) for n in samples if y[n] == 1) + quicksum(w[n] * t[n] for n in samples if y[n]==0), GRB.MINIMIZE)

    m.setParam('OutputFlag',0)
    m.Params.Threads = 4
    m.setParam(GRB.Param.TimeLimit,timelimit)

    starttime = time.time()

    m.optimize()

    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
        f.write('Solcount: '+str(m.solcount)+'.\n')
        f.write('Model status: '+str(m.status)+'.\n')

    runtime = str(time.time()-starttime)+'\n'

    if not m.SolCount > 0:
        print('No new column found.')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
            f.write('BRCG terminated because no new column was found.\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Runtimes.txt','a') as f:
            f.write('Sub problem: ')
            f.write(runtime)

        sys.exit()

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
        tnew = np.empty((N),dtype=int)
        for var in m.getVars():
            splitname = var.VarName.split('[')
            if ',' in splitname[1]:
                splitname = [splitname[0],splitname[1].split(',')[0],splitname[1].split(',')[1]]
                splitname = [splitname[0],splitname[1],splitname[2].split(']')[0]]
            else:
                splitname = [splitname[0],splitname[1].split(']')[0]]
            if splitname[0] == 's':
                snew[int(splitname[1])] = var.X
            elif splitname[0] == 't':
                tnew[int(splitname[1])] = var.X

        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/ModelStatus.txt','a') as f:
            f.write('Sub problem: ')
            f.write(status)
            f.write('\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Runtimes.txt','a') as f:
            f.write('Sub problem: ')
            f.write(runtime)
            f.write('\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/t.csv','a') as f:
            f.write(';'.join(tnew.astype(str))+'\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/s.csv','a') as f:
            f.write(';'.join(snew.astype(str))+'\n')
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/ObjectiveValue_sub.csv','a') as f:
            f.write(str(round(obj,3)))
            f.write('\n')
    else:
        with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
            f.write('Other status, namely: '+str(m.status)+', ')
            f.write('or other solution count: '+str(m.solcount)+'\n')

    return obj


if __name__ == "__main__":
    overall_starttime = time.time()
    homefolder = os.path.expanduser("~")

    dataset = sys.argv[1]
    datafolder = sys.argv[2]
    resultfolder = sys.argv[3]
    threads = int(sys.argv[4])
    timelimit = int(sys.argv[5])
    timelimit_overall = int(sys.argv[6])
    K = int(sys.argv[7])
    M = int(sys.argv[8])

    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    starttime_ = time.time()
    (X_,y_,w_,N_,P_,N_controls_) = load_data(datafolder,dataset)

    if not os.path.isdir(resultfolder):
        os.system('mkdir '+resultfolder)
    if not os.path.isdir(resultfolder+'/'+dataset):
        os.system('mkdir '+resultfolder+'/'+dataset)
    if not os.path.isdir(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)):
        os.system('mkdir '+resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads))

    # Initialize
    subproblem(X_,y_,w_,N_,K,M,threads,timelimit,1)

    obj_master_prev = N_
    count_no_improvement = 0

    stop = 0
    while stop == 0:
        obj_master = master(X_,y_,w_,N_,K,M,threads,timelimit)
        red_cost = subproblem(X_,y_,w_,N_,K,M,threads,timelimit,0)
        if red_cost >= 0:
            stop = 1
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
                f.write('Reduced cost non-negative, hence stop.\n')
        if time.time() - overall_starttime > timelimit_overall:
            stop = 1
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
                f.write('Time limit of '+str(timelimit_overall)+' s reached.\n')
        if obj_master_prev - obj_master < 0.0001:
            count_no_improvement += 1
        else:
            count_no_improvement = 0
        obj_master_prev = obj_master
        if count_no_improvement > 5:
            stop = 1
            with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Messages.txt','a') as f:
                f.write('Objective value not decreasing, hence stop.\n')

    with open(resultfolder+'/'+dataset+'/K'+str(K)+'_M'+str(M)+'_time'+str(timelimit)+'_threads'+str(threads)+'/Runtimes.txt','a') as f:
        f.write('Overall: ')
        f.write(str(time.time()-overall_starttime))
        f.write('\n')
