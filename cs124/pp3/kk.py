# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:01:25 2016

@author: carlos
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:03:28 2016

@author: carlos
"""

#!/usr/bin/env python

import heapq
import numpy as np
import math
import random
from scipy.stats import bernoulli
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import time
from itertools import product

def kk(vals):
    heapq.heapify(vals)
    v1 = heapq.heappop(vals)
    while len(vals) >= 1:
        v2 = heapq.heappop(vals)
        v1 = heapq.heappushpop(vals, v1-v2)
    residual = 0-v1
    return residual
    
def pp(vals, ps, n):
    d = {p : 0 for p in range(1, n+1)}
    for v, p in zip(vals, ps):
        d[p]  = d[p] + v
    return kk(d.values())
    
def resid(vals, ss, n, method):
    if method == 1:
        val = 0
        for s, v in zip(ss, vals):
            if s:
                val += v
            else:
                val -= v
        return abs(val)
    else:
        return pp(vals, ss, n)
    
def init(method, n):
    if method == 1:
        return np.random.randint(2, size=n)
    else:
        return random_ints(1, n, n) 
    
def methods(vals, n, n_iters, hill_climbing = False, sim_annealing = False, l="s", method = 1):
    if l == "h":
        hill_climbing = True
    elif l == "a":
        sim_annealing = True
    t0 = time.time()    
    best = float("inf")
    best_s = init(method, n)
    if sim_annealing:
        local_s = best_s
        local = float("inf")
        t_ = t(n_iters)
    for i in range(n_iters):
        if hill_climbing:
            new_s = rand_neighbor(n, best_s, method)
        elif sim_annealing:
            new_s = rand_neighbor(n, local_s, method)            
        else:
            new_s = init(method, n)            
        new = resid(vals, new_s, n, method)
        if sim_annealing:
            if new<local:
                local, local_s = new, new_s
            elif bernoulli.rvs(math.exp(-(new-local)/t_), 1):
                local, local_s = new, new_s               
        if new < best:
            best, best_s = new, new_s
    tf = time.time()-t0
    return best, tf
    
def random_ints(lower, upper, size):
    return [random.randrange(lower, upper+1) for i in range(size)]
    
def t(iters):
    return math.pow(10, 10) * math.pow(.9, (iters/300))
    
def rand_neighbor(n, ss, method):
    switches = np.random.choice(n, 2, replace = False)
    if method == 1:            
        if np.random.randint(2, size=1):
            ss[switches[-1]] = 1-ss[switches[-1]]
        ss[switches[0]] = 1- ss[switches[0]]
        return ss
    else:
        if switches[-1]:
            ss[switches[0]] = switches[-1]
        else:
            ss[switches[0]] = n
    return ss

def main(*args):
    sample = False
    if len(args) == 1:
        sample = True
        runs = 50
    else:
        runs = 1
        with open(args[1]) as f:
            data = f.readlines()
            vals = []
            for d in data:
                try:
                    vals.append(int(d))
                except:
                    pass
    n = 100
    verbose = True

    sums = [[]]*6
    times = [[]]*6
    kks = []
    for i in range(runs):
        if sample:
            vals = [-v for v in random_ints(1, 10 ** 12, n)]
        if verbose:
            print i+1,
        kks.append(kk(vals))         
        for i, (m, t) in enumerate(list(product([1,2],["s","h","a"]))):
            val, time = methods(vals, n, 25000, method = m, l = t)
            sums[i] = sums[i] + [val]
            times[i] = times[i] + [time]
    pd.DataFrame(kks).to_csv("KK.csv")
    pd.DataFrame(sums).to_csv("Vals.csv")
    pd.DataFrame(times).to_csv("Times.csv")    
    
def jitter(min_, max_, size):
    return (np.random.ranf(size)*(max_-min_))+min_

def graph(data_, j_size, title=None):
    fig, ax = plt.subplots()
    data = pd.read_csv(data_, index_col = 0)
    jitter_ = jitter(-j_size, j_size, len(list(data.iterrows())[0][1]))
    max_1 = 0
    max_2 = 0
    for i, d in enumerate(data.iterrows()):
        if max(list(d[1])) > max_1 and i > 2:
            max_1 = max(list(d[1]))
        elif max(list(d[1])) > max_2 and i < 3:
            max_2 = max(list(d[1]))
        plt.scatter(x = np.array([i]*len(jitter_)+jitter_), y = list(d[1]))
    plt.xlim(-0.1, 5.1) # apply the x-limits
    c1 = 1
    while max_1/(c1*10) > 1:
        c1 *= 10
    c2 = 1
    while max_2/(c2*10) > 1:
        c2 *= 10
    cb = max(c1,c2)
    cs = min(c1,c2)
    max1 = max(max_1, max_2)
    max2 = min(max_1, max_2)
    plt.ylim(-cb, (max1/cb+2)*cb)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if c2 > c1:
        axins = inset_axes(ax, 4,3, loc=5)
    else:
        axins = inset_axes(ax, 4,3, loc=6)        
        axins.yaxis.tick_right()
    for i, d in enumerate(data.iterrows()):
        plt.scatter(x = np.array([i]*len(jitter_)+jitter_), y = list(d[1]))
    if c2 > c1:
        axins.set_xlim(2.9, 5.1) # apply the x-limits
    else:
        axins.set_xlim(-0.1, 2.1)
    plt.ylim(-cs, (max2/cs+2)*cs)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.show()        
        
if __name__ == '__main__':
    import sys
    main(*sys.argv)
    #graph("Times.csv", .075)
    #graph("Vals.csv", 0.075)
    