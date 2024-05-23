import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os,pickle
from torch_geometric.data import Data
import torch_geometric.datasets
import torch_geometric.transforms as T
import sys
from utils import load_data
from persistence_image import persistence_image
from torch_geometric.utils import to_networkx
from utils import load_npz
from lazywitness import * 
import argparse 
import scipy.sparse as sp
import utils 
import math
from tqdm import tqdm 
from time import time 
from numba import typed,uint64,float64
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate') #0.02 => Pubmed
parser.add_argument('--lm_perc',type=float,default=0.05,help='%nodes as landmarks')
parser.add_argument('--eps',type=float,default=2.0,help='epsilon for Sparse Vietoris Rips')
parser.add_argument('--resolution',type = int, default = 50, help='resolution of PI')
parser.add_argument('--dataset', type=str, default='cora', \
		    choices=['cora','citeseer','pubmed','polblogs', 'ogb-arxiv',\
	       			'snap-patents', 'fb100','actor','roman_empire'], help='dataset')
parser.add_argument('--time',type=bool,default = False)
parser.add_argument('--vr',type=bool,default = False)
parser.add_argument('--spvr',type=bool,default = False)
parser.add_argument('--poisoned',type=bool,default = True)
args = parser.parse_args()
#prefix = 'data/nettack/' # 'data/'
# attack='nettack' #'meta'
prefix = 'data/'
# prefix = '/content/drive/MyDrive/WitnesscomplexGNNs/data/nettack/' # To run from colab
attack= 'meta'
# prefix = '/content/drive/MyDrive/WitnesscomplexGNNs/data/'
#prefix = 'data/pgd/' # 'data/'
#attack='pgd' #'meta'


def computeLWfeatures_numba(G,dataset_name, landmarkPerc=0.25, heuristic = 'degree'):
	L = int(len(G.nodes)*landmarkPerc) # Take top 25% maximal degree nodes as landmarks
	t = args.time
	if t:
		total_time = 0
		s_tm = time()
	landmarks,dist_to_cover,cover = getLandmarksbynumL(G, L = L, heuristic=heuristic)
	if t:
		land_tm = time()-s_tm
		total_time += land_tm
		print('Time to get landmarks: ',land_tm)
	# print('len(landmarks) : ',len(landmarks))
	local_pd = [np.ones((1,args.resolution,args.resolution))*(10**-8) ]*len(G.nodes)
	if t:
		s_tm = time()
	for u in tqdm(cover):
		G_cv = nx.Graph()
		cv = set(cover[u])
		for v in cv:
			for w in G.neighbors(v):
				if w in cv:
					G_cv.add_edge(v,w)
		len_cv = len(cv)
		# print('|cover| (',u,') => ',len(G_cv.nodes),' |local L| = ',int(len_cv*lm_perc*20))
		if args.dataset =='ogb-arxiv':
			local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, int(len_cv*args.lm_perc*10), heuristic = heuristic)
		else:
			local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, min(int(len_cv*args.lm_perc),5), heuristic = heuristic)
		DSparse,INF = get_sparse_matrix_cutoff(G_cv,local_dist_to_cover,local_landmarks)
		resultsparse = ripser(DSparse, distance_matrix=True,maxdim=0)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF
		PD = resultsparse['dgms'][0] # H0
		PI = persistence_image(PD, resolution = [args.resolution, args.resolution]).reshape(1, args.resolution, args.resolution)
		if np.isnan(PI).any():
			PI[np.isnan(PI)] = 10**-8 
		local_pd[u] = PI 
		for v in cv:
			local_pd[v] = PI # copy topological features of landmarks to the witnesses

	PI = np.concatenate(local_pd)
	if t:
		l_tm = time()-s_tm 
		total_time += l_tm
		print('Time taken for computing local PD : ',l_tm)
	else:
		# print('local pI shape: ',PI.shape)
		np.savez_compressed(dataset_name + '_localPInew.npz', PI)
		print('saved LocalPI at ',dataset_name + '_localPInew.npz')
	del local_pd
	del PI

	print('sparse matrix construction')
	if t:
		s_tm = time()
	# DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks)
	try:
		cutoff = max(dist_to_cover[n] for n in dist_to_cover if dist_to_cover[n]!=math.inf)
	except:
		cutoff = 1
	# if t:
	# 	s_tm2 = time()
	memory = typed.List([typed.Dict.empty(uint64, float64)]*len(landmarks))
	# pool = multiprocessing.Pool(processes=8)
	# # results = pool.map(bfs_worker, (G,landmarks,cutoff))
	# results = [pool.apply_async(bfs_worker, args=(G, landmark,cutoff)) for landmark in landmarks]
	# pool.close()
	# pool.join()
	# res = [result.get() for result in results]
	# for i in range(len(landmarks)):
	# 	for key,val in res[i].items():
	# 		memory[i][key] = val

	for i in range(len(landmarks)):
		u = landmarks[i]
		for key,val in nx.single_source_shortest_path_length(G,u,cutoff=cutoff).items():
			memory[i][key] = val
	
	# if t:
	# 	print('bfs time: ',time()-s_tm2)
	numba_dist_to_cover = typed.Dict.empty(uint64, float64)
	for key,val in dist_to_cover.items():
		numba_dist_to_cover[key] = val
	N = len(landmarks)
	IJ = np.array([(i, j) for i, j in combinations(range(N), 2)])
	D, INF = get_sparse_matrix_cutoff_numba(memory, IJ, numba_dist_to_cover) # Construct sparse LxL matrix
	I, J = zip(*IJ)
	DSparse = sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()
	# print(DSparse.toarray())
	# sp.save_npz(dataset_name+".sparse.npz", DSparse)
	# print('ripser call')
	resultsparse = ripser(DSparse, distance_matrix=True,maxdim=0)
	resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
	PD = resultsparse['dgms'][0] # H0
	# print(PD)
	if t:
		gl_tm = time()-s_tm
		total_time+= gl_tm
		print('Time taken in global PD: ',gl_tm)
		print('Total time taken: ',total_time)
	else:
		with open(dataset_name+'.pdnew.pkl','wb') as f:
			pickle.dump(PD,f)
	return PD

def computeVRfeatures(G,dataset_name):
	print('vr feature.')
	if os.path.isfile(dataset_name+'.vrpd.pkl'):
		with open(dataset_name+'.vrpd.pkl','rb') as f:
			PD = pickle.load(f)
	else:
		local_pd = [None]*len(G.nodes)
		for u in tqdm(G.nodes):
			G_cv = nx.ego_graph(G, u, radius=5, center=True, undirected=True, distance=None)
			len_cv = len(G_cv.nodes)
			mapping = {old:new for old,new in zip(G_cv.nodes,range(len_cv))}
			G_cv = nx.relabel_nodes(G_cv, mapping)
			# print('|cover| (',u,') => ',len_cv)
			# local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, int(len_cv*args.lm_perc), heuristic = heuristic)
			# DSparse,INF = get_sparse_matrix(G_cv,local_dist_to_cover,local_landmarks)
			# construct sparse distance of all pair shortest path length of cv
			DSparse, INF = utils.all_pair_shortest_path_lengths_sparse(G_cv)
			resultsparse = ripser(DSparse, distance_matrix=True)
			resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF
			PD = resultsparse['dgms'][0] # H0
			PI = persistence_image(PD, resolution = [args.resolution, args.resolution]).reshape(1, args.resolution, args.resolution)
			# print('local_pd: ',len(local_pd),' ',u)
			# print(u,' has Nan: ',np.isnan(PI).any())
			if np.isnan(PI).any():
				PI[np.isnan(PI)] = 10**-8 
			local_pd[u] = PI 
		# for i,_ in enumerate(local_pd):
		# 	if local_pd[i] is None:
		# 		local_pd[i] = np.ones((1,args.resolution,args.resolution))*(10**-8) 
				# print('local_pd is none for node : ',i)
			# print(local_pd[i].shape)
		DSparse,INF = utils.all_pair_shortest_path_lengths_sparse(G) # Construct sparse LxL matrix
		print('ripser call')
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		# print('Global PD: ',PD)
		with open(dataset_name+'.vrpd.pkl','wb') as f:
			pickle.dump(PD,f)
		PI = np.concatenate(local_pd)
		# print('local pI shape: ',PI.shape)
		np.savez_compressed(dataset_name + '_localPI.vr.npz', PI)
	return PD 

def computeSparseVRfeatures(G,dataset_name):
	print('Sparse vr feature. epsilon = ',args.eps)
	if os.path.isfile(dataset_name+'.spvrpd.pkl'):
		with open(dataset_name+'.spvrpd.pkl','rb') as f:
			PD = pickle.load(f)
	else:
		local_pd = [None]*len(G.nodes)
		if not nx.is_connected(G):
			raise ValueError("Input graph must be connected for all-pair shortest path computation.")

		# Get the number of nodes in the graph
		num_nodes = len(G.nodes)
		local_pd = [None]*len(G.nodes)
		for u in tqdm(G.nodes):
			G_cv = nx.ego_graph(G, u, radius=5, center=True, undirected=True, distance=None)
			len_cv = len(G_cv.nodes)
			mapping = {old:new for old,new in zip(G_cv.nodes,range(len_cv))}
			G_cv = nx.relabel_nodes(G_cv, mapping)
			# print('|cover| (',u,') => ',len_cv)
			# local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, int(len_cv*args.lm_perc), heuristic = heuristic)
			# DSparse,INF = get_sparse_matrix(G_cv,local_dist_to_cover,local_landmarks)
			# construct sparse distance of all pair shortest path length of cv
			shortest_paths = dict(nx.all_pairs_shortest_path_length(G_cv))
			# Create a sparse distance matrix
			D = np.inf * np.ones((len_cv, len_cv), dtype=np.float32)

			# Fill in the distance matrix with the computed shortest path lengths
			for source, targets in shortest_paths.items():
				for target, length in targets.items():
					D[source][target] = length
			INF = np.max(D)
			lambdas = utils.getGreedyPerm(D)
			# Now compute the sparse distance matrix
			DSparse = utils.getApproxSparseDM(lambdas, args.eps, D)
			resultsparse = ripser(DSparse, distance_matrix=True)
			resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
			PD = resultsparse['dgms'][0] # H0
			PI = persistence_image(PD, resolution = [args.resolution, args.resolution]).reshape(1, args.resolution, args.resolution)
			# print('local_pd: ',len(local_pd),' ',u)
			# print(u,' has Nan: ',np.isnan(PI).any())
			if np.isnan(PI).any():
				PI[np.isnan(PI)] = 10**-8 
			local_pd[u] = PI 
		# Compute the shortest path lengths using NetworkX
		shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

		# Create a sparse distance matrix
		D = np.inf * np.ones((num_nodes, num_nodes), dtype=np.float32)

		# Fill in the distance matrix with the computed shortest path lengths
		for source, targets in shortest_paths.items():
			for target, length in targets.items():
				D[source][target] = length
		INF = np.max(D)
		lambdas = utils.getGreedyPerm(D)
		# Now compute the sparse distance matrix
		DSparse = utils.getApproxSparseDM(lambdas, args.eps, D)
		# Finally, compute the filtration
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
	# print('Global PD: ',PD)
	with open(dataset_name+'.spvrpd.pkl','wb') as f:
		pickle.dump(PD,f)
	PI = np.concatenate(local_pd)
	print('local pI shape: ',PI.shape)
	np.savez_compressed(dataset_name + '_localPI.spvr.npz', PI)
	return PD


if __name__=='__main__':
	if args.dataset !='ogb-arxiv':
		# load dataset
		dataset_name = args.dataset
		if args.poisoned:
			# Computing LW features for perturbed adj matrices
			adj = sp.load_npz(prefix + dataset_name + '/' + dataset_name + '_'+attack+'_adj_'+str(args.ptb_rate)+'.npz')
		else:
			# Computing LW features for unperturbed adj matrix
			adj, _, _ = utils.load_npz(prefix + dataset_name + '/' + dataset_name + '.npz')
		G = nx.from_numpy_array(adj.toarray())
		dataset_name2 = prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate)
		# PD = computeLWfeatures(G,dataset_name2, landmarkPerc=args.lm_perc, heuristic = 'degree')
	else:
		dataset_name = args.dataset 
		# Computing LW features for perturbed data.edge_index
		# data = torch.load(prefix+dataset_name+ '/' + dataset_name+"_prbcd_pygdata_"+str(args.ptb_rate)+".pt")
		
		# Computing LW features for unperturbed data.edge_index
		from ogbn import get_ogbarxiv
		data = get_ogbarxiv()
		print('converting to nx')
		G = to_networkx(data,to_undirected = True,remove_self_loops = True)
		dataset_name2 = prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate)
	if args.vr:
		PD = computeVRfeatures(G,dataset_name2)
	elif args.spvr:
		PD = computeSparseVRfeatures(G,dataset_name2)
		print(PD)
		print('epsilon = ',args.eps)
	else:
		L = int(len(G.nodes)*args.lm_perc) # Take top 25% maximal degree nodes as landmarks
		print('|L|=',L)

	PD = computeLWfeatures_numba(G,dataset_name2, landmarkPerc=args.lm_perc, heuristic = 'degree')
	# print(PD)
	# resolution_size = 50
	PI = persistence_image(PD, resolution = [args.resolution, args.resolution])
	PI[np.isnan(PI)] = 10**-8
	PI = PI.reshape(1, args.resolution, args.resolution)
	if not args.time:
		print('saved Global PI at: ',prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PInew.npz')
		if args.vr:
			np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PInew.vr.npz', PI)	
		elif args.spvr:
			np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PInew.spvr.npz', PI)	
		else:
			np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PInew.npz', PI)