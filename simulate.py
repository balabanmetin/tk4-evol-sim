#!/usr/bin/env python3

import sys,json
import numpy as np
from scipy.linalg import expm
import random
from os.path import join
from os import makedirs


# $1 is config file
# $2 is output directory

tran=str.maketrans('ACGT', 'TGCA')
def rc(seq):
	return seq[::-1].translate(tran)

def hamming(seq1,seq2):
	return 1- sum([x==y for x,y in zip(seq1,seq2)])/len(seq1)
with open(sys.argv[1]) as f:
	j = json.load(f)

# calculate transition probabilities

pag=j["alpha"]
pac=j["delta"]
pat=j["gamma"]
w=j["omega"]
t=j["t"]
length=j["length"]

if "scaffolds" in j:
	numscaf = j["scaffolds"]
else:
	numscaf = 20

if "seed" in j:
	seed = j["seed"]
else:
	seed=12345
random.seed(seed)

# A C G T
aux=w/(1-w)

# TK4 rate matrix
R = np.array([[-(pac+pag+pat) ,pac ,pag, pat],
	[pac*aux, -(pac*aux+pat+pag*aux), pat ,pag*aux],
	[pag*aux, pat, -(pag*aux+pat+pac*aux) ,pac*aux],
	[pat, pag, pac,-(pat+pag+pac)]])

pi = np.array([w/2, 1/2-w/2, 1/2-w/2, w/2])

# normalize the rate matrix
nrm = -np.dot(np.diag(R),pi)
Rn = R/nrm

# matrix exponentiation
Pm = expm(t*Rn)
# hamming distance
Hd = 1-(np.dot(np.diag(Pm),pi))
# jukes-cantor phylogenetic distance
Jc = -3/4*np.log(1-4/3*Hd)

# expected substitution probabilities
F = (Pm.T*pi).T

# expected number of each type of substition
# in a genome with length "length"
Flen = np.ceil(length*F)

# create two genomes as list of pairs
genome_pairs = []
for ind_i,i in enumerate(["A","C", "G","T"]):
	for ind_j,j in enumerate(["A","C", "G","T"]):
		genome_pairs +=  [(i,j)]*int(Flen[ind_i,ind_j])

# shuffle the sites
random.shuffle(genome_pairs)

# partition into equal length scaffolds
scafsize = len(genome_pairs)//numscaf
scafs = []
for i in range(numscaf-1):
	scafs.append(genome_pairs[scafsize*i:scafsize*(i+1)])
scafs.append(genome_pairs[scafsize*(numscaf-1):])

count = 0
for i in scafs:
	count += len(i)

# for each scaffold, flip g1 and g2 with %50 probability
flipped_scaf = []
for sc in scafs:
	seq1 = "".join([x[0] for x in sc])
	seq2 = "".join([x[1] for x in sc])
	flip1 = random.randint(0,1)
	flip2 = random.randint(0,1)
	if flip1:
		seq1 = rc(seq1)
	if flip2:
		seq2 = rc(seq2)
	flipped_scaf.append((seq1,seq2))

makedirs(join(sys.argv[2],"genomes"))
for i in [0,1]:
	fst = ""
	for n,sc in enumerate(flipped_scaf):
		fst += ">scaf_"+str(i)+"\n"
		fst += sc[i]+"\n"
	with open(join(sys.argv[2],"genomes/genome_%s.fasta" % str(i)), "w") as f:
		f.write(fst)

sim_param = dict()
sim_param["invocation"]=sys.argv
sim_param["jc"]=Jc
sim_param["hamming"]=Hd
sim_param["F"]=F.tolist()
sim_param["Pm"]=Pm.tolist()
sim_param["R"]=R.tolist()


with open(join(sys.argv[2],"sim_out.json"), "w") as f:
	f.write(json.dumps(sim_param, sort_keys=True, indent=4))
