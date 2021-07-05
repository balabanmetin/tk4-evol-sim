#!/usr/bin/env python3

import sys,json
import numpy as np
from scipy.linalg import expm
import random
from os.path import join
from os import makedirs
import readfq as rfq

# $1 is config file
# $2 is output directory

tran=str.maketrans('ACGT', 'TGCA')
def rc(seq):
	return seq[::-1].translate(tran)

def hamming(seq1,seq2):
	return 1- sum([x==y for x,y in zip(seq1,seq2)])/len(seq1)

def findOccurrences(scafno,string_,substring_):
    return [(scafno,i) for i, letter in enumerate(string_) if letter == substring_]


def params(w,pac,pag,pat,length,pi):
	# A C G T
	aux=w/(1-w)

	# TK4 rate matrix
	R = np.array([[-(pac+pag+pat) ,pac ,pag, pat],
		[pac*aux, -(pac*aux+pat+pag*aux), pat ,pag*aux],
		[pag*aux, pat, -(pag*aux+pat+pac*aux) ,pac*aux],
		[pat, pag, pac,-(pat+pag+pac)]])

	

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
	#print(Flen)
	return Jc, Hd, F, Pm, R, Flen

with open(sys.argv[1]) as f:
	j = json.load(f)

# calculate transition probabilities

pag=j["alpha"]
pac=j["delta"]
pat=j["gamma"]
t=j["t"]
gen = 0

if "scaffolds" in j:
	numscaf = j["scaffolds"]
else:
	numscaf = 20

if "seed" in j:
	seed = j["seed"]
else:
	seed=12345
random.seed(seed)


if "genome" in j:
	ref_genome=j["genome"]
	gen = 1
	f = open(ref_genome)
	n, slen, qlen = 0, 0, 0
	scafs = []
	idx_occurances = []				# List of tuples (scaf.no, index of letter).

	for i in range(len(["A","C", "G","T"])):
		idx_occurances.append([])

	for name, seq, qual in rfq.readfq(f):
		#l = len(seq)
		#print(seq)
		seq = seq.upper()
		if j["keepunknownchar"] == False:
			chars = set(seq)
			for i in ["A","C","G","T"]:
				chars.remove(i)
			for i in chars:
				seq = seq.replace(i,"")
			
		for ind_i,i in enumerate(["A","C", "G","T"]):
			idx_occurances[ind_i].extend(findOccurrences(n,seq,i))
		scafs.append([(i,i) for i in seq])
		n += 1
	no_bases = np.array([len(i) for i in idx_occurances])
	#print(no_bases)
	slen = sum(no_bases)
	#print(slen)
	pi = no_bases/slen
	w = 2*pi[0]
	#print(pi,w)

	length = slen
	Jc, Hd, F, Pm, R, Flen = params(w,pac,pag,pat,length,pi)
	#print(scafs)
	#print(idx_occurances)

	#shuffle the sites 
	for i in idx_occurances:
		random.shuffle(i)
	#print(idx_occurances)

	for ind_i,i in enumerate(["A","C", "G","T"]):
		start=0
		for ind_j,j in enumerate(["A","C", "G","T"]):
			if ind_i == ind_j:
				start += int(Flen[ind_i,ind_j])
				continue
			for k in range(int(Flen[ind_i,ind_j])):
				if start + k >= len(idx_occurances[ind_i]):
					continue
				# idx_occurances[ind_i][start+k][0] contains the scaffold number.
				# idx_occurances[ind_i][start+k][1] contains the index of the character in the scaffold.
				scafs[idx_occurances[ind_i][start+k][0]][idx_occurances[ind_i][start+k][1]] = (i,j)
				
			start += int(Flen[ind_i,ind_j])
			#print(i,j, "done")
		
		idx_occurances[ind_i] = []
	#print(scafs)
else:
	w=j["omega"]
	length=j["length"]
	pi = np.array([w/2, 1/2-w/2, 1/2-w/2, w/2])

	Jc, Hd, F, Pm, R, Flen = params(w,pac,pag,pat,length, pi)
	# create two genomes as list of pairs
	genome_pairs = []
	for ind_i,i in enumerate(["A","C", "G","T"]):
		for ind_j,j in enumerate(["A","C", "G","T"]):
			genome_pairs +=  [(i,j)]*int(Flen[ind_i,ind_j])

	#print(genome_pairs)

	# shuffle the sites
	random.shuffle(genome_pairs)

	# partition into equal length scaffolds
	scafsize = len(genome_pairs)//numscaf
	scafs = []
	for i in range(numscaf-1):
		scafs.append(genome_pairs[scafsize*i:scafsize*(i+1)])
	scafs.append(genome_pairs[scafsize*(numscaf-1):])

	#print(genome_pairs)
	#print("scafs: ",scafs)
count = 0
for scaf in scafs:
	count += len(scaf)
#print(count)

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