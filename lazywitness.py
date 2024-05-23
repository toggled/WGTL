from ripser import ripser
import numpy as np
from scipy import sparse
import time
import networkx as nx
import random 
import math
from tqdm import tqdm 
from time import time
from numba import jit, prange
import numba

# def get_distancetocover(G, nodes,_L,debug = True):
#     dist_to_cover = {}
#     cover = {}
#     for u in _L:
#         dist_to_cover[u] = 0
#     for u in nodes:
#         # if u=='3':
#         #     debug = True
#         # else:
#         #     debug = False
#         if u in dist_to_cover:
#             if (debug):
#                 print(u,' previously added to dist_to_cover')
#             continue
#         Queue = [u]
#         dist = {u: 0}
#         breakflag = False
#         parent = {u:u}
#         while len(Queue):
#             v = Queue.pop(0)
#             if (debug):
#                 print('pop ',v)
#             for nbr_v in G.neighbors(v):
#                 dist_nv = dist[v] + 1
#                 if (debug):
#                     print('nbr: ',nbr_v,' dist_nv: ',dist_nv)
#                 if dist.get(nbr_v, math.inf) > dist_nv:
#                     parent[nbr_v] = parent[v]
#                     if (debug):
#                         print('condition check: ',dist.get(nbr_v, math.inf))
#                     dist[nbr_v] = dist_nv
#                     # cover[nbr_v] = v
#                     if nbr_v in _L: # The first time BFS encounters a landmark, that landmark contains it in its cover.
#                         if (debug):
#                             print('nbr_v ',nbr_v,' in _L')
#                         dist_to_cover[u] = dist[nbr_v]
#                         cover[nbr_v] = cover.get(nbr_v,[])+[u]
#                         breakflag = True
#                         # break
#                     else:
#                         Queue.append(nbr_v)
#             if breakflag:
#                 break
#         if breakflag is False: # distance to nearest neighbor in L is infinity, because disconnected.
#             dist_to_cover[u] = math.inf

#     print('cover => \n',cover)
#     return dist_to_cover, cover


def get_distancetocover(G, nodes,_L,debug = False):
    dist_to_cover = {}
    cover = {}
    # parent = {}
    front = {}
    visited = {}
    for u in nodes:
        # parent[u] = u 
        visited[u] = False 
    for u in _L:
        dist_to_cover[u] = 0
        cover[u] = [u]
        front[u] = []
        for v in G.neighbors(u):
            if not visited[v]:
                visited[v] = True 
                front[u].append(v)
                cover[u].append(v)
                dist_to_cover[v] = 1
                # parent[v] = u
 
    while True:
        some_front_nonempty = False 
        for u in _L:
            temp_front = []
            some_front_nonempty = (len(front[u])>0)
            while len(front[u]):
                v = front[u].pop(0)
                for w in G.neighbors(v):
                    dist_to_cover_w = dist_to_cover[v] + 1
                    if w not in dist_to_cover:
                        dist_to_cover[w] = dist_to_cover_w
                        # parent[w] = parent[v]
                        cover[u].append(w)
                        temp_front.append(w)
                    else:
                        if dist_to_cover_w < dist_to_cover[w]:
                            cover[u].append(w)
                            # parent[w] = parent[v]
                            dist_to_cover[w] = dist_to_cover_w
                            temp_front.append(w)
            front[u] = temp_front 
        if some_front_nonempty is False:
            break 
    if len(dist_to_cover) != len(nodes):
        for u in nodes:
            if u not in dist_to_cover:
                dist_to_cover[u] = math.inf 
    
    # for u in nodes:
    #     # if u=='3':
    #     #     debug = True
    #     # else:
    #     #     debug = False
    #     if u in dist_to_cover:
    #         if (debug):
    #             print(u,' previously added to dist_to_cover')
    #         continue
    #     Queue = [u]
    #     dist = {u: 0}
    #     breakflag = False
    #     parent = {u:u}
    #     while len(Queue):
    #         v = Queue.pop(0)
    #         if (debug):
    #             print('pop ',v)
    #         for nbr_v in G.neighbors(v):
    #             dist_nv = dist[v] + 1
    #             if (debug):
    #                 print('nbr: ',nbr_v,' dist_nv: ',dist_nv)
    #             if dist.get(nbr_v, math.inf) > dist_nv:
    #                 parent[nbr_v] = parent[v]
    #                 if (debug):
    #                     print('condition check: ',dist.get(nbr_v, math.inf))
    #                 dist[nbr_v] = dist_nv
    #                 # cover[nbr_v] = v
    #                 if nbr_v in _L: # The first time BFS encounters a landmark, that landmark contains it in its cover.
    #                     if (debug):
    #                         print('nbr_v ',nbr_v,' in _L')
    #                     dist_to_cover[u] = dist[nbr_v]
    #                     cover[nbr_v] = cover.get(nbr_v,[])+[u]
    #                     breakflag = True
    #                     # break
    #                 else:
    #                     Queue.append(nbr_v)
    #         if breakflag:
    #             break
    #     if breakflag is False: # distance to nearest neighbor in L is infinity, because disconnected.
    #         dist_to_cover[u] = math.inf

    if debug: print('cover => \n',cover)
    return dist_to_cover, cover

def getLandmarksbynumL(G, L = 2, heuristic = 'degree'):
    """ dist_nearest_nbr_inL[u] is the distance from u to its nearest nbr in L"""
    if heuristic == 'degree':
        _degreenodes = sorted([(G.degree[u],u) for u in G.nodes],reverse = True)
        _L = set([pair[1] for pair in _degreenodes[:L]])
        del _degreenodes
        # print('L = ',L,' len(_L) = ',len(_L))
        dist_to_cover,cover = get_distancetocover(G, G.nodes,_L)
        return list(_L), dist_to_cover,cover
        
    if heuristic == 'random':
        _L = random.sample(G.nodes, k=L)
        dist_to_cover = get_distancetocover(G, G.nodes,_L)
        return list(_L), dist_to_cover,cover
    
        
def getLandmarksbyeps(G, epsilon = 2, heuristic = 'epsmaxmin'):
    dist_to_cover = {}
    _L = []
    if heuristic=='epsmaxmin':
        marked = {}
        for u in G.nodes:
            dist_to_cover[u] = math.inf
            marked[u] = False
        num_marked = 0
        _N = len(G.nodes)
        while num_marked < _N:
            if num_marked == 0:
                u = list(G.nodes)[0]
            Queue = [u]
            dist = {u: 0}
            marked[u] = True
            dist_to_cover[u] = 0
            _L.append(u)
            num_marked+=1
            while len(Queue):
                v = Queue.pop(0)
                for nbr_v in G.neighbors(v):
                    dist_nv = dist[v] + 1
                    if nbr_v not in dist:
                        if not marked[nbr_v] and dist_nv <= epsilon:
                            num_marked+=1
                            marked[nbr_v] = True
                            dist_to_cover[nbr_v] = dist_nv
                        dist[nbr_v] = dist_nv
                        Queue.append(nbr_v)
            # print(len(_L),' ',num_marked)
            for v in G.nodes:
                if not marked[v]:
                    u = v
                    break
        return _L, dist_to_cover

# compute the |L| x |L| matrix => sparse distance metric on landmarks
def get_sparse_matrix(G,dist_to_cover,landmarks):
    data = {}
    all_pairSP_len = dict(nx.all_pairs_shortest_path_length(G))
    # print('all pair sp')
    for i,u in enumerate(landmarks):
        for j,v in enumerate(landmarks):
            if i<j:
                e_ij = math.inf
                for n in G.nodes: # witness node = n
                    # print(u,v,n)
                    dist_i = all_pairSP_len[u].get(n,math.inf)
                    dist_j = all_pairSP_len[v].get(n,math.inf)
                    mx = max( max(dist_i,dist_j) - dist_to_cover[n],0.0)
                    if mx < e_ij:
                        e_ij = mx
                data[(i,j)] = e_ij
            elif j==i:
                data[(i,j)] = 0.0
            else:
                data[(i,j)] = data[(j,i)]
    I,J,D,INFINITY = [],[],[],0
    for key,val in data.items():
        I.append(key[0])
        J.append(key[1])
        D.append(val)
        if val != math.inf:
            INFINITY = max(val,INFINITY)
    del data
    N = len(landmarks)
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr(), INFINITY

# compute the |L| x |L| matrix => sparse distance metric on landmarks
def get_sparse_matrix_cutoff(G,dist_to_cover,landmarks, covers = None):
    # data = {}
    debug = (covers is not None)
    try:
        cutoff = max(dist_to_cover[n] for n in dist_to_cover if dist_to_cover[n]!=math.inf)
    except:
        cutoff = 1
    # print('cutoff: ',cutoff)
    if covers is None:
        witness_candidates = G.nodes
        # enum_land = enumerate(landmarks)
    # else:
    #     print(len(landmarks),cutoff)
    #     enum_land = tqdm(enumerate(landmarks))
    # enum_land = enumerate(landmarks)
    # print('cutoff: ',cutoff)
    # all_pairSP_len = dict(nx.all_pairs_shortest_path_length(G,cutoff=cutoff))
    # print('all pair sp')
    I,J,D,INFINITY = [],[],[],0
    memory = {}
    if debug:
        s_tm = time()
    for u in landmarks:
        memory[u] = nx.single_source_shortest_path_length(G,u,cutoff=cutoff)
    if debug:
        print('sp len time: ',time()-s_tm)
    N=0
    # print('len(memory) = ',len(memory))
    # for i,u in tqdm(enumerate(landmarks)):
    for i,u in enumerate(landmarks):
        N+=1
        # u_toall = nx.single_source_shortest_path_length(G,u,cutoff=cutoff)
        u_toall = memory[u]
        for j,v in enumerate(landmarks):
            # if debug:
            #     print(i,j)
            if i>j:
                continue 
            key = (i,j)
            # if debug:
            #     print(key)
            if j==i:
                # data[(i,j)] = 0.0
                val = 0.0
                I.append(key[0])
                J.append(key[1])
                D.append(val)
            else:
                # v_toall = nx.single_source_shortest_path_length(G,v,cutoff=cutoff)
                v_toall = memory[v]
                e_ij = math.inf

                if covers is not None:
                    # witness_candidates = set(covers[u]).union(covers[v])
                    witness_candidates = set(u_toall.keys()).union(v_toall.keys())
                # witness_node = witness_candidates[0]
                for n in witness_candidates: # witness node = n
                    dist_i = u_toall.get(n,math.inf)
                    dist_j = v_toall.get(n,math.inf)
                    mx = max( max(dist_i,dist_j) - dist_to_cover[n],0.0)
                    if mx < e_ij:
                        e_ij = mx
                        # witness_node = n  
                    if e_ij <= 1: # Because the subsequent ones can't be smaller
                        break
                # data[(i,j)] = e_ij
                # if debug:
                #     print(key,' ',witness_node)
                val = e_ij 
                I.append(key[0])
                J.append(key[1])
                D.append(val)
                I.append(key[1])
                J.append(key[0])
                D.append(val)
            
            # else:
            #     val = data[(j,i)]

            if val != math.inf:
                INFINITY = max(val,INFINITY)
    # for key,val in data.items():
    #     I.append(key[0])
    #     J.append(key[1])
    #     D.append(val)
    #     if val != math.inf:
    #         INFINITY = max(val,INFINITY)
    # del data
    # N = len(landmarks)
    # if debug:
    #     print(D)
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr(), INFINITY

@jit(parallel=True, nopython=True)
def get_sparse_matrix_cutoff_numba(memory, IJ, dist_to_cover):
    INFINITY = 0
    D = [numba.float64(x) for x in range(len(IJ))]
    # I = [numba.int64(x) for x in range(0)]
    # J = [numba.int64(x) for x in range(0)]
    # D = [numba.float64(x) for x in range(0)]
    # if debug:
    #     s_tm = time()
    
    # if debug:
    #     print('sp len time: ',time()-s_tm)
    # if debug:
    #     s_tm = time()
    for iter in prange(len(IJ)): # Landmarks is a List
        i,j = IJ[iter]
        # print(i,j)
        # for j in range(N):
        #     # if debug: 
        #     #     print(i,j)
        #     if i>j:
        #         continue 
        #     key = (i,j)
        #     # if debug:
        #     #     print(key)
        # if j==i:
        #     val = 0.0
        #     I.append(key[0])
        #     J.append(key[1])
        #     D.append(val)
        # else:
        u_toall = memory[i] # memory is a List
        v_toall = memory[j]
        e_ij = math.inf

        # witness_candidates = set(covers[u]).union(covers[v])
        # witness_candidates = u_toall.keys() + v_toall.keys() #set(u_toall.keys()).union(v_toall.keys())
        for n in u_toall.keys(): # witness node = n
            dist_i = u_toall.get(n,math.inf)
            dist_j = v_toall.get(n,math.inf)
            mx = max( max(dist_i,dist_j) - dist_to_cover.get(n,0),0.0) # dist_to_cover is a dictionary
            if mx < e_ij:
                e_ij = mx
            if e_ij <= 1: # Because the subsequent ones can't be smaller
                break
        for n in v_toall.keys(): # witness node = n
            dist_i = u_toall.get(n,math.inf)
            dist_j = v_toall.get(n,math.inf)
            mx = max( max(dist_i,dist_j) - dist_to_cover.get(n,0),0.0) # dist_to_cover is a dictionary
            if mx < e_ij:
                e_ij = mx
            if e_ij <= 1: # Because the subsequent ones can't be smaller
                break
        D[iter] = e_ij
            # val = e_ij 
            # I.append(key[0])
            # J.append(key[1])
            # D.append(val)
            # I.append(key[1])
            # J.append(key[0])
            # D.append(val)
        if e_ij != math.inf:
            INFINITY = max(e_ij,INFINITY)
    # if debug:
    #     print('nested for loop time: ',time()-s_tm)
    return D,INFINITY