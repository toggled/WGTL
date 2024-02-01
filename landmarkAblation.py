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
parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate') #0.02 => Pubmed
parser.add_argument('--lm_perc',type=float,default=0.05,help='%nodes as landmarks')
parser.add_argument('--eps',type=float,default=2.0,help='epsilon for Sparse Vietoris Rips')
parser.add_argument('--resolution',type = int, default = 50, help='resolution of PI')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer','pubmed','polblogs'], help='dataset')
parser.add_argument('--vr',type=bool,default = False)
parser.add_argument('--spvr',type=bool,default = False)
args = parser.parse_args()
#prefix = 'data/nettack/' # 'data/'
#attack='nettack' #'meta'
prefix = 'data/'
attack= 'meta'
#prefix = 'data/pgd/' # 'data/'
#attack='pgd' #'meta'
def computeLWfeatures(G,dataset_name, landmarkPerc=0.25,heuristic = 'degree'):
	"""
	 Computes LW persistence pairs / if pre-computed loads&returns them. 
	landmarkPerc = %nodes fo G to be picked for landmarks. [ Currently we are selecting landarmks by degree]
	heuristic = landmark selection heuristics ('degree'/'random') 
	Returns only H0 PD	
	"""
	if os.path.isfile(dataset_name+str(args.lm_perc)+'.pd.pkl'):
		with open(dataset_name+str(args.lm_perc)+'.pd.pkl','rb') as f:
			PD = pickle.load(f)
	else:	
		L = int(len(G.nodes)*landmarkPerc) # Take top 25% maximal degree nodes as landmarks
		s_tm = time()
		landmarks,dist_to_cover,cover = getLandmarksbynumL(G, L = L,heuristic=heuristic)
		# print('len(landmarks) : ',len(landmarks))
		local_pd = [None]*len(G.nodes)
		for u in tqdm(cover):
			G_cv = nx.Graph()
			cv = set(cover[u])
			for v in cv:
				for w in G.neighbors(v):
					if w in cv:
						G_cv.add_edge(v,w)
			len_cv = len(cv)
			# print('|cover| (',u,') => ',len_cv)
			local_landmarks, local_dist_to_cover, _ = getLandmarksbynumL(G_cv, int(len_cv*args.lm_perc), heuristic = heuristic)
			DSparse,INF = get_sparse_matrix(G_cv,local_dist_to_cover,local_landmarks)
			resultsparse = ripser(DSparse, distance_matrix=True)
			resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF
			PD = resultsparse['dgms'][0] # H0
			PI = persistence_image(PD, resolution = [args.resolution, args.resolution]).reshape(1, args.resolution, args.resolution)
			# print('local_pd: ',len(local_pd),' ',u)
			# print(u,' has Nan: ',np.isnan(PI).any())
			if np.isnan(PI).any():
				PI[np.isnan(PI)] = 10**-8 
			local_pd[u] = PI 
			for v in cv:
				local_pd[v] = PI # copy topological features of landmarks to the witnesses
		for i,_ in enumerate(local_pd):
			if local_pd[i] is None:
				local_pd[i] = np.ones((1,args.resolution,args.resolution))*(10**-8) 
				# print('local_pd is none for node : ',i)
			# print(local_pd[i].shape)
		DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks) # Construct sparse LxL matrix
		# print('ripser call')
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		e_tm = time() - s_tm 
		# print('Global PD: ',PD)
		with open(dataset_name+str(args.lm_perc)+'.pd.pkl','wb') as f:
			pickle.dump(PD,f)
		# with open(dataset_name+'_local.pi.npz','wb') as f:
		# 	pickle.dump(local_pd,f)
		# PI = np.array(local_pd)
		PI = np.concatenate(local_pd)
		# print('local pI shape: ',PI.shape)
		np.savez_compressed(dataset_name +str(args.lm_perc)+ '_localPI.npz', PI)
	return PD, e_tm 

def computeVRfeatures(G,dataset_name):
	print('vr feature.')
	if os.path.isfile(dataset_name+str(args.lm_perc)+'.vrpd.pkl'):
		with open(dataset_name+str(args.lm_perc)+'.vrpd.pkl','rb') as f:
			PD = pickle.load(f)
	else:
		s_tm = time()
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
		e_tm = time() - s_tm
		# print('Global PD: ',PD)
		with open(dataset_name+str(args.lm_perc)+'.vrpd.pkl','wb') as f:
			pickle.dump(PD,f)
		PI = np.concatenate(local_pd)
		# print('local pI shape: ',PI.shape)
		np.savez_compressed(dataset_name +str(args.lm_perc)+ '_localPI.vr.npz', PI)
	return PD , e_tm

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

# load dataset
dataset_name = args.dataset
# Computing LW features for perturbed adj matrices
#perturbed_adj = sp.load_npz(prefix + dataset_name + '/' + dataset_name + '_'+attack+'_adj_'+str(args.ptb_rate)+'.npz')
#G = nx.from_numpy_matrix(perturbed_adj.toarray())
# Computing LW features for unperturbed adj matrix
adj, _, _ = utils.load_npz(prefix + dataset_name + '/' + dataset_name + '.npz')
G = nx.from_numpy_matrix(adj.toarray())
dataset_name2 = prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate)
# PD = computeLWfeatures(G,dataset_name2, landmarkPerc=args.lm_perc, heuristic = 'degree')
if args.vr:
	PD,feat_comp_tm = computeVRfeatures(G,dataset_name2)
	with open('tm_'+args.dataset+'_vr.txt','a') as wf:
		wf.write(str(feat_comp_tm)+","+str(args.lm_perc)+'\n')
	import sys 
	sys.exit(0)
elif args.spvr:
	PD = computeSparseVRfeatures(G,dataset_name2)
	print(PD)
	print('epsilon = ',args.eps)
else:
	L = int(len(G.nodes)*args.lm_perc) # Take top 25% maximal degree nodes as landmarks
	landmarks,dist_to_cover,cover = getLandmarksbynumL(G, L = L,heuristic='degree')
	_vals = [j for j in dist_to_cover.values() if j!=math.inf]
	_coverszs = [len(j) for j in cover.values()] 
	print('|L| = ',len(landmarks))
	print('epsilon = min: ',min(_vals),' max: ',max(_vals),' mean: ',np.mean(_vals), 'std: ',np.std(_vals))
	print('cover distr = min: ',min(_coverszs),' max: ',max(_coverszs),' mean: ',np.mean(_coverszs), 'std: ',np.std(_coverszs))
	dataset_name2 = prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate)+'_'
	PD,feat_comp_tm = computeLWfeatures(G,dataset_name2, landmarkPerc=args.lm_perc, heuristic = 'degree')
	with open('tm_'+args.dataset+'.txt','a') as wf:
		wf.write(str(feat_comp_tm)+","+str(args.lm_perc)+'\n')
# resolution_size = 50
PI = persistence_image(PD, resolution = [args.resolution, args.resolution])
PI[np.isnan(PI)] = 10**-8
PI = PI.reshape(1, args.resolution, args.resolution)
if args.vr:
	np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PI.vr.npz', PI)	
elif args.spvr:
	np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) + '_PI.spvr.npz', PI)	
else:
	np.savez_compressed(prefix + dataset_name + '/' + dataset_name+"_"+str(args.ptb_rate) +'_'+str(args.lm_perc)+ '_PI.npz', PI)