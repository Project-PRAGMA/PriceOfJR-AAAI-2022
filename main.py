


from distributions import EqualChooseDistribution,OneDDistribution,TwoDDistribution
import random
from baseProgram import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



#returns maximum support of any candidate
def maxSupport(V,m):
    #compute the number of supporters for all candidates
    counts = [len([v for v in V if c in V[v]]) for c in range(m)]
    return max(counts)

#returns most popular candidate
def mostPopular(V,m):
    #compute the number of supporters for all candidates
    counts = [len([v for v in V if c in V[v]]) for c in range(m)]
    return counts.index(max(counts))


#returns most popular candidate. input is not an integer but a list (i.e. we do not assume the set of candidates to be range(1,m))
def mostPopular2(V,candidates):
    #compute the number of supporters for all candidates
    counts = {c:len([v for v in V if c in V[v]]) for c in candidates}
    return max(counts, key=counts.get)


#Function checks for JR
def isJR(V,W,m,k,n):
    #remove all voters which already approve one candidate in W
    V_new = dict((v,V[v]) for v in V if set(V[v]).intersection(set(W)) == set())
    
    if len(V_new) == 0:
        return True
    
    if maxSupport(V_new,m) >= float(n)/float(k):
        return False
    return True

#runs GreedyCC until JR is satisfied
def GreedyCC(V,m,k):
    W = []
    n = len(V)
    while(isJR(V,W,m,k,n)==False):
        c = mostPopular(V,m)
        W = W + [c]
        #remove all candidates that approve c
        V = dict((v,V[v]) for v in V if c not in V[v])
    return W

#runs greedyCC until committee contains k candidates
#if all candidates are easily covered fills up with arbitrary candidates
def GreedyCCFull(V,m,k):
    W = []
    n = len(V)
    while(len(W) < k and len(V)>0):
        c = mostPopular(V,m)
        W = W + [c]
        #remove all candidates that approve c
        V = dict((v,V[v]) for v in V if c not in V[v])
    if(len(V)==0 and len(W)<k):
        while(len(W)<k):
            print("all voters covered, GreedyCC is now filling up with arbirtrary candidates")
            W = W + list([[c for c in range(m) if c not in W][0]])
    return W

#greedy algorithm as described in experimental section of the paper
#max_coverage_app is a greedy estimate of maximum coverage
def egalitarianGreedy(V,m,k,max_coverage_app,max_approval):
    W = GreedyCC(V,m,k)
    C_rest = [item for item in range(m) if item not in W]
    #check which approximation is currently worse
    app_score = approvalScore(V,W)
    cov_score = coverageScore(V,W)
    
    while(len(W)<k):
        app_score = approvalScore(V,W)
        cov_score = coverageScore(V,W)
        cov_dict = {c:(len([v for v in V if c in V[v] and not bool(set(W)&set(V[v]))]),
                       len([v for v in V if c in V[v]])) for c in C_rest}
        if (float(app_score)/float(max_approval) >= float(cov_score)/float(max_coverage_app)):
            #maximize for coverage
            c = sorted(cov_dict.items(),key=lambda t: (t[1][0],t[1][1]),reverse=True)[0][0]
        else:
            #maximize for approval
            c = sorted(cov_dict.items(),key=lambda t: (t[1][1],t[1][0]),reverse=True)[0][0]
        W = W + [c]
        C_rest = [item for item in C_rest if item != c]
    return W



def approvalScore(V,W):
    return sum([len(set(V[v]).intersection(set(W))) for v in V])

def coverageScore(V,W):
    return len([v for v in V if len(set(V[v]).intersection(set(W))) != 0])
        



####################################################
################# START EXPERIMENTS ################
####################################################




#Goal: Show Trade-off between Social Welfare and Coverage

#Step0 - initiate parameter 

m = 100
n = 100 
k = 10

model_list = [('IC',0.1),('1D',0.054),('2D',0.195)]
#set to false if data is already available and only plots should be created
create_data = True


if create_data:
    random.seed(200)
    for model_tuple in model_list:
        model = model_tuple[0]
        p = model_tuple[1]
        #contains data which is relevant for paretocurve 1 (cov on x-axis)
        exp1_data = pd.DataFrame(columns=['cov','z','sw','JR'],dtype=float)
        exp1_data2 = pd.DataFrame(columns=['z','sw','cov','dist_sw','dist_cov'],dtype=float)
        exp1_data3 = pd.DataFrame(columns=['z','av_app'],dtype=float)
        exp1_data4 = pd.DataFrame(columns=['cov','z','sw','JR'],dtype=float)

        #z is the number of elections we sample 
        for z in range(100):
            
            #Step1 - create elections
            if model == 'IC':
                election = EqualChooseDistribution(p).generate(range(m),n)
            if model == '1D':
                election = OneDDistribution(p).generate(range(m),n)
            if model == '2D':
                election = TwoDDistribution(p).generate(range(m),n)                

            av_app = np.mean([len(election[v]) for v in election.keys()])
            exp1_data3 = exp1_data3.append({'z':z,'av_app':av_app},ignore_index=True)

            #Step 2 - Compute maximum SW and maximum coverage without restrictions
            max_approval = compute(range(m), election, 0, n*m, 0, n, k, goal=APPROVAL_MAX, requireJR=False)[1]
            max_coverage = compute(range(m), election, 0, n*m, 0, n, k, goal=COVERAGE_MAX, requireJR=True)[1]

            #Step 3 - Compute max sw for all possible fixed coverage fractions
            for x in [i/100 for i in range(40,100)]:
                ilp_result = compute(range(m), election, 0, n*m, x*max_coverage, n, k, goal=APPROVAL_MAX, requireJR=True)
                if ilp_result[0]:
                    y = ilp_result[1]
                    #exp1_data.at[(i/max_coverage,z),'sw'] = y/max_approval
                    exp1_data = exp1_data.append({'cov':x,'z':z,'sw':y/max_approval,'JR':True},ignore_index=True)
                ilp_result2 = compute(range(m), election, 0, n*m, x*max_coverage, n, k, goal=APPROVAL_MAX, requireJR=False)
                y2 = ilp_result2[1]
                exp1_data = exp1_data.append({'cov':x,'z':z,'sw':y2/max_approval,'JR':False},ignore_index=True)

            
            #Step 4 - Compute max coverage for all possible fixed social welfare fractions
            for x in [i/200 for i in range(80,200)]:
                ilp_result = compute(range(m), election, x*max_approval, n*m, 0, n, k, goal=COVERAGE_MAX, requireJR=True)
                if ilp_result[0]:
                    y = ilp_result[1]
                    exp1_data4 = exp1_data4.append({'cov':y/max_coverage,'z':z,'sw':x,'JR':True},ignore_index=True)
                else:
                    exp1_data4 = exp1_data4.append({'cov':0,'z':z,'sw':x,'JR':True},ignore_index=True)
                ilp_result2 = compute(range(m), election, x*max_approval, n*m, 0, n, k, goal=COVERAGE_MAX, requireJR=False)
                y2 = ilp_result2[1]
                exp1_data4 = exp1_data4.append({'cov':y2/max_coverage,'z':z,'sw':x,'JR':False},ignore_index=True)

            #create data for Greedy points 
            Z = GreedyCCFull(election,m,k)
            max_coverage_approx = coverageScore(election,Z)
            W = egalitarianGreedy(election,m,k,max_coverage_approx,max_approval)
            app_frac = approvalScore(election,W)/max_approval
            cov_frac = coverageScore(election,W)/max_coverage
            best_sw = compute(range(m), election, 0, n*m, coverageScore(election,W), n, k, goal=APPROVAL_MAX, requireJR=True)[1]
            best_cov = compute(range(m), election, approvalScore(election,W), n*m, 0, n, k, goal=COVERAGE_MAX, requireJR=True)[1]
            exp1_data2 = exp1_data2.append({'z':z,'sw':app_frac,'cov':cov_frac,'dist_sw':float(best_sw)/float(max_approval) - float(app_frac),'dist_cov':float(best_cov)/float(max_coverage) - float(cov_frac)},ignore_index=True)


        #save data
        exp1_data.to_csv(model + '_cov-x-axis' + '.csv')
        exp1_data2.to_csv(model + '_greedy-points'+ '.csv')
        exp1_data4.to_csv(model + '_sw-x-axis'+ '.csv')
                        



############### Read Data and make plots ###############################
experiments = ['IC_','1D_','2D_']
y_limits1 = [[0.87,1.01],[0.74,1.01],[0.74,1.01]]
x_limits1 = [[0.8,1.01],[0.4,1.01],[0.4,1.01]]

y_limits2 = [[0.85,1.01],[0.0,1.01],[0.0,1.01]]
x_limits2 = [[0.8,1.01],[0.75,1.01],[0.74,1.01]]


for i in range(3):
    #Read Data for Pareto curve with coverage on x-axis 
    data = pd.read_csv(experiments[i] + 'cov-x-axis.csv')
    data_jr = data[data['JR'] == 1]
    data_njr = data[data['JR'] == 0]

    greedy = pd.read_csv(experiments[i] + 'greedy-points.csv')
    greedy_opt = greedy[greedy['dist_sw']==0]
    greedy_nopt = greedy[greedy['dist_sw']!=0]

    data_jr = data_jr.groupby('cov')['sw'].mean()
    data_njr = data_njr.groupby('cov')['sw'].mean()

    #Create figure of Pareto cruve with coverage on x-axis
    fig = plt.figure(figsize=(12,8), dpi= 300)
    ax = fig.add_subplot(111)
    ax.set_ylim(y_limits1[i])
    ax.set_xlim(x_limits1[i])
    plt.plot(data_njr.index,data_njr,linewidth=3.0,color='red',linestyle='dashed')
    plt.plot(data_jr.index,data_jr,linewidth=3.0)
    plt.scatter(greedy_opt['cov'],greedy_opt['sw'],color = 'C0',alpha=0.5,marker='^')
    plt.scatter(greedy_nopt['cov'],greedy_nopt['sw'],color = 'red',alpha=0.5,marker='o')

    ax.yaxis.label.set_size(28)
    ax.xaxis.label.set_size(28)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.ylabel(r'social welfare',labelpad=30)
    plt.xlabel(r'coverage',labelpad=30)
    plt.savefig(experiments[i] + 'cov-x-axis.png',bbox_inches='tight')

    #report statistics 
    print(experiments[i] + 'fraction of greedy points with optimal sw for alpha coverage: ' + str(len(greedy_opt)))
    print(experiments[i]  + 'average distance of non-optimal greedy points along sw axis: ' +  str(np.mean(greedy_nopt['dist_sw'])))
    print(' ')

    #Read Data for Pareto curve with social welfare on x-axis 
    mirr = pd.read_csv(experiments[i] +'sw-x-axis.csv')
    mirr_jr = mirr[mirr['JR'] == 1]
    mirr_njr = mirr[mirr['JR'] == 0]
    mirr = mirr_jr.groupby('sw')['cov'].mean()
    mirr_njr = mirr_njr.groupby('sw')['cov'].mean()

    greedy_opt = greedy[greedy['dist_cov']==0]
    greedy_nopt = greedy[greedy['dist_cov']!=0]

    #Create figure of Pareto cruve with social welfare on x-axis
    fig = plt.figure(figsize=(12,8), dpi= 300)
    ax = fig.add_subplot(111)
    ax.set_xlim(x_limits2[i])
    ax.set_ylim(y_limits2[i])
    plt.plot(mirr.index,mirr_njr,linewidth=3,color='red',linestyle='dashed')
    plt.plot(mirr.index,mirr,linewidth=3)
    plt.scatter(greedy_opt['sw'],greedy_opt['cov'],color = 'C0',alpha=0.5,marker='^')
    plt.scatter(greedy_nopt['sw'],greedy_nopt['cov'],color = 'red',alpha=0.5,marker='o')

    ax.yaxis.label.set_size(28)
    ax.xaxis.label.set_size(28)
    ax.tick_params(axis='both', which='major', labelsize=24)
    plt.ylabel(r'coverage',labelpad=30)
    plt.xlabel(r'social welfare',labelpad=30)
    plt.savefig(experiments[i] + 'sw-x-axis',bbox_inches='tight')

    print( experiments[i] + 'fraction of greedy points with optimal coverage for alpha sw:  ' + str(len(greedy_opt)))
    print(experiments[i] + 'average distance of non-optimal greedy points along cov axis:' + str(np.mean(greedy_nopt['dist_cov'])))
    print(' ')
