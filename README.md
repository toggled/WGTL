## When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning
We propose the Witness Graph Topological Layer (WGTL), the first approach that systematically bridges adversarial graph learning with persistent homology representations of graphs. 

WGTL adopts witness complex topological features as priors and topological loss as regulariser to make GNNs adversarially robust. We demonstrate the effectiveness of WGTL on various graph datasets under targetted as well as non-targetted attacks. 


![](intro.png)
-----------
<h> Requirements: </h>
- python >= 3.9
- cython
- ripser
- numba, gensim
- numpy,scipy 
- torch
- torch_geometric
- gudhi
- vit_pytorch
- ogb
- numba 

<h> Datasets </h>

Download from here: https://drive.google.com/file/d/17YtLK2uwZYRtEIYWcqkahXvduOrtTpVt/view?usp=sharing 

<h> Generating perturbed graphs (attacks): </h>
1. python generate_attack.py --dataset citeseer --ptb_rate 0.05 --seed 15 --model A-Meta-Self # mettack
2. python generate_attack.py --dataset cora --ptb_rate 0.05 --seed 15 --model PGD # PGD attack
3. python generate_attack.py --dataset snap-patents --ptb_rate 0.05 --seed 15 --model A-Meta-Self
4. python prbcd_attack.py --dataset ogb-arxiv --ptb_rate 0.1 --seed 0 # PRBCD

<h> Generating local and global witness topological features </h>
1. python generate_features.py --dataset cora --ptb_rate 0.05 --lm_perc 0.02
2. python generate_features.py --dataset citeseer --ptb_rate 0.05 --lm_perc 0.02
3. python generate_features.py --dataset ogb-arxiv --ptb_rate 0.1 --lm_perc 0.0005 
4. python generate_features.py --dataset snap-patents --ptb_rate 0.1 --lm_perc 0.02 

To generate features computed on poisoned graph:  Append `--poisoned True`

To generate Vietoris-Rips features:  Append `--vr True`

<h> Instructions: </h>
1. test_gcnRe.py: test vanilla GCN performance under meta attack with different pertubation rates
2. witcompnn_mainRe.py: test WGTL performance under mettack and nettack with different pertubation rates
