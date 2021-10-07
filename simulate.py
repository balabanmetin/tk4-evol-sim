#!/usr/bin/env python3

import sys, json
import numpy as np
from scipy.linalg import expm
import random
from os.path import join
from os import makedirs
import readfq as rfq

# $1 is config file
# $2 is output directory

tran = str.maketrans('ACGT', 'TGCA')


def rc(seq):
    return seq[::-1].translate(tran)


def hamming(seq1, seq2):
    return 1 - sum([x == y for x, y in zip(seq1, seq2)]) / len(seq1)


def findOccurrences(scafno, string_, substring_):
    return [(scafno, i) for i, letter in enumerate(string_) if letter == substring_]


def params(w, pac, pag, pat, length, pi, t):
    # A C G T
    aux = w / (1 - w)

    # TK4 rate matrix
    R = np.array([[-(pac + pag + pat), pac, pag, pat],
                  [pac * aux, -(pac * aux + pat + pag * aux), pat, pag * aux],
                  [pag * aux, pat, -(pag * aux + pat + pac * aux), pac * aux],
                  [pat, pag, pac, -(pat + pag + pac)]])

    # normalize the rate matrix
    nrm = -np.dot(np.diag(R), pi)
    Rn = R / nrm

    # matrix exponentiation
    Pm = expm(t * Rn)
    # hamming distance
    Hd = 1 - (np.dot(np.diag(Pm), pi))
    # jukes-cantor phylogenetic distance
    Jc = -3 / 4 * np.log(1 - 4 / 3 * Hd)

    # expected substitution probabilities
    F = (Pm.T * pi).T

    # expected number of each type of substition
    # in a genome with length "length"
    Flen = np.ceil(length * F)
    # print(Flen)
    return Jc, Hd, F, Pm, R, Flen


with open(sys.argv[1]) as f:
    jconfig = json.load(f)

# calculate transition probabilities

pag = jconfig["alpha"]
pac = jconfig["delta"]
pat = jconfig["gamma"]
t = jconfig["t"]
gen = 0

if "scaffolds" in jconfig:
    numscaf = jconfig["scaffolds"]
else:
    numscaf = 20

if "seed" in jconfig:
    seed = jconfig["seed"]
else:
    seed = 12345
rngs = [random.Random(), random.Random()]
for i in [0, 1]:
    rngs[i].seed(seed + i)  # default 12345 and 12346


for pairid in [0, 1]:
    # create idx_occurances
    idx_occurances = []  # List of tuples (scaf.no, index of letter).
    for i in range(len(["A", "C", "G", "T"])):
        idx_occurances.append([])
    # it can be from user input genome or random.
    if "genome" in jconfig:
        ref_genome = jconfig["genome"]
        f = open(ref_genome)
        n, slen, qlen = 0, 0, 0
        scafs = []

        for name, seq, qual in rfq.readfq(f):
            # l = len(seq)
            # print(seq)
            seq = seq.upper()
            if jconfig["keepunknownchar"] == False:
                chars = set(seq)
                for i in ["A", "C", "G", "T"]:
                    chars.remove(i)
                for i in chars:
                    seq = seq.replace(i, "")

            for ind_i, i in enumerate(["A", "C", "G", "T"]):
                idx_occurances[ind_i].extend(findOccurrences(n, seq, i))
            scafs.append([(i, i) for i in seq])
            n += 1
        no_bases = np.array([len(i) for i in idx_occurances])
        slen = sum(no_bases)
        pi = no_bases / slen
        w = pi[0] + pi[3]
        length = slen
    else:
        w = jconfig["omega"]
        length = jconfig["length"]
        pi = np.array([w / 2, 0.5 - w / 2, 0.5 - w / 2, w / 2])
        counts_for_each_base = np.ceil(length * pi)
        genome_pairs = []
        for ind_i, i in enumerate(["A", "C", "G", "T"]):
            genome_pairs += [(i, i)] * int(counts_for_each_base[ind_i])
        length = len(genome_pairs)
        # shuffle the sites
        rn0 = random.Random()  # use rng[0] to shuffle the random base genome to ensure base genome is the same
        rn0.seed(seed)
        rn0.shuffle(genome_pairs)

        # partition into equal length scaffolds
        scafsize = length // numscaf
        scafs = []
        for i in range(numscaf - 1):
            scafs.append(genome_pairs[scafsize * i:scafsize * (i + 1)])
        # last scaf is usually slightly larger due to rounding
        scafs.append(genome_pairs[scafsize * (numscaf - 1):])
        # construct idx_occurances by creating scaffold sequences.
        for scaf_no, scaf in enumerate(scafs):
            for ind_i, i in enumerate(["A", "C", "G", "T"]):
                idx_occurances[ind_i].extend(findOccurrences(scaf_no, "".join([pair[0] for pair in scaf]), i))

    Jc, Hd, F, Pm, R, Flen = params(w, pac, pag, pat, length, pi, t/2)  #note t/2. we simulate like a cherry
    # shuffle the sites
    for i in idx_occurances:
        rngs[pairid].shuffle(i)
    # print(idx_occurances)

    for ind_i, i in enumerate(["A", "C", "G", "T"]):
        start = 0
        for ind_j, j in enumerate(["A", "C", "G", "T"]):
            if ind_i == ind_j:
                start += int(Flen[ind_i, ind_j])
                continue
            for k in range(int(Flen[ind_i, ind_j])):
                if start + k >= len(idx_occurances[ind_i]):
                    continue
                # idx_occurances[ind_i][start+k][0] contains the scaffold number.
                # idx_occurances[ind_i][start+k][1] contains the index of the character in the scaffold.
                scafs[idx_occurances[ind_i][start + k][0]][idx_occurances[ind_i][start + k][1]] = (i, j)
            start += int(Flen[ind_i, ind_j])
        idx_occurances[ind_i] = []

    count = 0
    for scaf in scafs:
        count += len(scaf)
    # print(count)

    # for each scaffold, flip g1 and g2 with %50 probability
    flipped_scaf = []
    for sc in scafs:
        seq1 = "".join([x[0] for x in sc])
        seq2 = "".join([x[1] for x in sc])
        flip1 = rngs[pairid].randint(0, 1)
        flip2 = rngs[pairid].randint(0, 1)
        if flip1:
            seq1 = rc(seq1)
        if flip2:
            seq2 = rc(seq2)
        flipped_scaf.append((seq1, seq2))

    makedirs(join(sys.argv[2], "genomes"), exist_ok=True)
    fst = ""
    for n, sc in enumerate(flipped_scaf):
        fst += ">scaf_" + str(n) + "\n"
        fst += sc[1] + "\n"  # always the second genome, not the base one.
    with open(join(sys.argv[2], "genomes/genome_%s.fasta" % str(pairid)), "w") as f:
        f.write(fst)

Jc, Hd, F, Pm, R, Flen = params(w, pac, pag, pat, length, pi, t)  # this time with t for printing
sim_param = dict()
sim_param["invocation"] = sys.argv
sim_param["jc"] = Jc
sim_param["hamming"] = Hd
sim_param["F"] = F.tolist()
sim_param["Pm"] = Pm.tolist()
sim_param["R"] = R.tolist()
tk4_param = {"R": (F[0][1] + F[1][0] + F[2][3] + F[3][2]),
             "P": (F[0][2] + F[2][0] + F[1][3] + F[3][1]),
             "Q1": (F[0][3] + F[3][0]),
             "Q2": (F[1][2] + F[2][1]),
             "w": w}
sim_param["tk4_param"] = tk4_param
with open(join(sys.argv[2], "sim_out.json"), "w") as f:
    f.write(json.dumps(sim_param, sort_keys=True, indent=4))
