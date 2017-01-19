#!/bin/env python

import sys
import os
import argparse
import subprocess
import gzip
import math
import random
from path import path
from collections import defaultdict
import bedtools
import shutil
import copy
import os
import math
import heapq
import sklearn.cluster
import scipy.sparse
import resource

from Bio import SeqIO


###############################################################
###############################################################
# Input and processing
###############################################################
###############################################################

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=path, required=True, help="Output directory")
parser.add_argument("--seqs", type=path, required=True, help="Input sequences, fasta format")
parser.add_argument("--mixture", type=float, default=0.5, help="Mixture parameter determining the relative weight of facility-location relative to sum-redundancy. Default=0.5")
args = parser.parse_args()
workdir = args.outdir

assert args.mixture >= 0.0
assert args.mixture <= 1.0


if not workdir.exists():
    workdir.makedirs()

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(workdir / "stdout.txt")
fh.setLevel(logging.DEBUG) # >> this determines the file level
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)# >> this determines the output level
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)



###################
# Run psiblast
###################
if not (workdir / "db").exists():
    cmd = ["makeblastdb",
      "-in", args.seqs,
      "-input_type", "fasta",
      "-out", workdir / "db",
      "-dbtype", "prot"]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

if not (workdir / "psiblast_result.tab").exists():
    cmd = ["psiblast",
      "-query", args.seqs,
      "-db", workdir / "db",
      "-num_iterations", "6",
      "-outfmt", "6 qseqid sseqid pident length mismatch evalue bitscore",
      "-seg", "yes",
      "-out", workdir / "psiblast_result.tab"
    ]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

###################
# Read psiblast output
###################

db = {}
# db: {seq_id: {
#               "neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "in_neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "scop": {level_id: [scop_ids]},
#               "seq": ""
#             }
#         }

fasta_sequences = SeqIO.parse(open(args.seqs),'fasta')
for seq in fasta_sequences:
    seq_id = seq.id
    db[seq_id] = {"neighbors": {}, "in_neighbors": {}, "seq": seq.seq}

with open(workdir / "psiblast_result.tab", "r") as f:
    for line in f:
        if line.strip() == "": continue
        if line.startswith("Search has CONVERGED!"): continue
        line = line.split()
        seq_id1 = line[0]
        seq_id2 = line[1]
        pident = float(line[2])
        evalue = line[5]
        log10_e = math.log10(float(evalue))
        if float(evalue) <= 1e-2:
            db[seq_id2]["neighbors"][seq_id1] = {"log10_e": log10_e, "pct_identical": pident}
            db[seq_id1]["in_neighbors"][seq_id2] = {"log10_e": log10_e, "pct_identical": pident}

###############################################################
###############################################################
# Submod optimization functions
###############################################################
###############################################################


#############################
# Similarity functions
#############################

def sim_from_db(db, sim, seq_id1, seq_id2):
    d = db[seq_id1]["neighbors"][seq_id2]
    return sim_from_neighbor(sim, d)

def sim_from_neighbor(sim, d):
    return sim(d["log10_e"], d["pct_identical"])

def fraciden(log10_e, pct_identical):
    return float(pct_identical) / 100

def rankpropsim(log10_e, pct_identical):
    return numpy.exp(-numpy.power(10, log10_e) / 100.0)

def rankpropsim_loge(log10_e, pct_identical):
    return numpy.exp(-log10_e / 100.0)

def logloge(log10_e, pct_identical):
    if (-log10_e) <= 0.1:
        return 0.0
    elif (-log10_e) >= 1000:
        return 3.0
    else:
        return numpy.log10(-log10_e)

def oneprankpropsim(log10_e, pct_identical):
    return 1.0 + 1e-3 * rankpropsim_loge(log10_e, pct_identical)

def prodevaliden(log10_e, pct_identical):
    return fraciden(log10_e, pct_identical) * logloge(log10_e, pct_identical) / 3

def one(log10_e, pct_identical):
    return 1.0

def p90(log10_e, pct_identical):
    return float(pct_identical >= 0.9)


#############################
# Objective functions
# -----------------
# An objective is a dictionary
# {"eval": db, seq_ids, sim -> float, # The value of an objective function
#  "diff": db, seq_ids, sim, data -> float, # The difference in value after adding seq_id
#  "negdiff": db, seq_ids, sim, data -> float, # The difference in value after removing seq_id
#  "update": db, seq_ids, sim, data -> data, # Update the data structure to add seq_id as a representative
#  "negupdate": db, seq_ids, sim, data -> data, # Update the data structure to remove seq_id as a representative
#  "base_data": db, sim -> data, # The data structure corresponding to no representatives chosen
#  "full_data": db, sim -> data, # The data structure corresponding to all representatives chosen
#  "name": name}
# db: Database
# sim: Similiarity function
# data: function-specific data structure which may be modified over the course of an optimization algorithm
#############################

######################
# summaxacross
# AKA facility location
######################

def summaxacross_eval(db, seq_ids, sim):
    max_sim = {seq_id:0 for seq_id in db}
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["in_neighbors"].items():
            if neighbor_seq_id in max_sim:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim[neighbor_seq_id]:
                    max_sim[neighbor_seq_id] = sim_val
                #max_sim[neighbor_seq_id] = max(max_sim[neighbor_seq_id], sim(d["log10_e"], d["pct_identical"]))
            else:
                pass
                #raise Exception("Found node with neighbor not in set")
    return sum(max_sim.values())

# summaxacross data:
# Who each example is represented by
# "examples": {seq_id: (representive, val) }
# Who each represntative represents
# "representatives" {seq_id: {example: val}}

summaxacross_base_data = lambda db, sim: {"examples": {seq_id: (None, 0) for seq_id in db},
                                              "representatives": {}}

summaxacross_full_data = lambda db, sim: {"examples": {seq_id: (seq_id, sim_from_db(db, sim, seq_id, seq_id)) for seq_id in db},
                                              "representatives": {seq_id: {seq_id: sim_from_db(db, sim, seq_id, seq_id)} for seq_id in db}}

def summaxacross_diff(db, seq_id, sim, data):
    diff = 0
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                diff += sim_val - data["examples"][neighbor_seq_id][1]
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return diff

def summaxacross_update(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    data["representatives"][seq_id] = {}
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                data["examples"][neighbor_seq_id] = (seq_id, sim_val)
                data["representatives"][seq_id][neighbor_seq_id] = sim_val
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return data

# O(D^2)
def summaxacross_negdiff(db, seq_id, sim, data):
    diff = 0
    # For each neighbor_seq_id that was previously represented by seq_id
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            diff += -d
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            diff += sim_from_db(db, sim, neighbor_seq_id, best_id) - d
    return diff

# O(D^2)
def summaxacross_negupdate(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            data["examples"][neighbor_seq_id] = (None, 0)
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            data["examples"][neighbor_seq_id] = (best_id, sim_from_db(db, sim, neighbor_seq_id, best_id))
            data["representatives"][best_id][neighbor_seq_id] = sim_from_db(db, sim, neighbor_seq_id, best_id)
    del data["representatives"][seq_id]
    return data

summaxacross = {"eval": summaxacross_eval,
          "diff": summaxacross_diff,
          "negdiff": summaxacross_negdiff,
          "update": summaxacross_update,
          "negupdate": summaxacross_negupdate,
          "base_data": summaxacross_base_data,
          "full_data": summaxacross_full_data,
          "name": "summaxacross"}

######################
# minmaxacross
# Most comparable to CD-HIT
# Eval only
######################

def minmaxacross_eval(db, seq_ids, sim):
    max_sim = {seq_id:0 for seq_id in db}
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["in_neighbors"].items():
            if neighbor_seq_id in max_sim:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim[neighbor_seq_id]:
                    max_sim[neighbor_seq_id] = sim_val
                #max_sim[neighbor_seq_id] = max(max_sim[neighbor_seq_id], sim(d["log10_e"], d["pct_identical"]))
            else:
                pass
                #raise Exception("Found node with neighbor not in set")

    return min(max_sim.values())

minmaxacross = {"eval": minmaxacross_eval,
          "name": "minmaxacross"}

######################
# maxmaxwithin
# Also comparable to CD-HIT
# Eval only
######################

def maxmaxwithin_eval(db, seq_ids, sim):
    max_sim = float("-inf")
    seq_ids_set = set(seq_ids)
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["neighbors"].items():
            if neighbor_seq_id in seq_ids_set:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim:
                    max_sim = sim_val
    return -max_sim

maxmaxwithin = {"eval": maxmaxwithin_eval,
                "name": "maxmaxwithin"}

######################
# summaxwithin
# AKA negfacloc
######################

def summaxwithin_eval(db, seq_ids, sim):
    max_sim = {seq_id:0 for seq_id in seq_ids}
    for chosen_seq_id in seq_ids:
        for neighbor_seq_id, d in db[chosen_seq_id]["in_neighbors"].items():
            if neighbor_seq_id == chosen_seq_id: continue
            if neighbor_seq_id in max_sim:
                sim_val = sim(d["log10_e"], d["pct_identical"])
                if sim_val > max_sim[neighbor_seq_id]:
                    max_sim[neighbor_seq_id] = sim_val
            else:
                pass
    return -sum(max_sim.values())

# summaxwithin data:
# Who each example is represented by
# "examples": {seq_id: (representive, val) }
# Who each represntative represents
# "representatives" {seq_id: {example: val}}

summaxwithin_base_data = lambda db, sim: {"examples": {seq_id: (None, 0) for seq_id in db},
                                              "representatives": {}}

def summaxwithin_full_data(db, sim):
    data = {}
    data["examples"] = {}
    data["representatives"] = {seq_id: {} for seq_id in db}
    for seq_id in db:
        neighbors = {neighbor_seq_id: d for neighbor_seq_id,d in db[seq_id]["neighbors"].items() if neighbor_seq_id != seq_id}
        if len(neighbors) == 0:
            data["examples"][seq_id] = (None, 0)
        else:
            d = max(neighbors.items(), key=lambda d: sim_from_neighbor(sim, d[1]))
            data["examples"][seq_id] = (d[0], sim_from_neighbor(sim, d[1]))
            data["representatives"][d[0]][seq_id] = sim_from_neighbor(sim, d[1])
    return data

def summaxwithin_diff(db, seq_id, sim, data):
    diff = 0
    # Difference introduced in other representatives
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id == seq_id: continue
        if neighbor_seq_id in data["representatives"]:
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                # adding a penalty of sim_val, removing old penalty
                diff -= sim_val - data["examples"][neighbor_seq_id][1]
    # Difference from adding this representative
    neighbors = {neighbor_seq_id: d for neighbor_seq_id,d in db[seq_id]["neighbors"].items() if neighbor_seq_id != seq_id}
    if len(neighbors) == 0:
        diff -= 0
    else:
        d = max(neighbors.items(), key=lambda d: sim_from_neighbor(sim, d[1]))
        diff -= sim_from_neighbor(sim, d[1])
    return diff

def summaxwithin_update(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    # Find best repr for seq_id
    candidate_ids = (set(db[seq_id]["neighbors"].keys()) & set(data["representatives"].keys())) - set([seq_id])
    if len(candidate_ids) == 0:
        data["examples"][seq_id] = (None, 0)
    else:
        best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, seq_id, x))
        data["examples"][seq_id] = (best_id, sim_from_db(db, sim, seq_id, best_id))
        data["representatives"][best_id][seq_id] = sim_from_db(db, sim, seq_id, best_id)
    # Find ids represented by seq_id
    data["representatives"][seq_id] = {}
    for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
        if neighbor_seq_id in data["examples"]:
            if neighbor_seq_id == seq_id: continue
            sim_val = sim(d["log10_e"], d["pct_identical"])
            if sim_val > data["examples"][neighbor_seq_id][1]:
                data["examples"][neighbor_seq_id] = (seq_id, sim_val)
                data["representatives"][seq_id][neighbor_seq_id] = sim_val
        else:
            pass
            #raise Exception("Found node with neighbor not in set")
    return data

# O(D^2)
def summaxwithin_negdiff(db, seq_id, sim, data):
    diff = 0
    # Difference introduced in other representatives
    # For each neighbor_seq_id that was previously represented by seq_id
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            diff += d # removing a penalty of -d
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            # removing a penalty of d, adding a new penalty of -sim(neighbor, best)
            diff += d - sim_from_db(db, sim, neighbor_seq_id, best_id)
    # Difference from adding this representative
    diff += data["examples"][seq_id][1] # removing a penalty of -sim
    return diff

# O(D^2)
def summaxwithin_negupdate(db, seq_id, sim, data):
    data = copy.deepcopy(data)
    del data["examples"][seq_id]
    new_representatives = (set(data["representatives"].keys()) - set([seq_id]))
    for neighbor_seq_id, d in data["representatives"][seq_id].items():
        # Find the next-best representative for neighbor_seq_id
        candidate_ids = set(db[neighbor_seq_id]["neighbors"].keys()) & new_representatives
        if len(candidate_ids) == 0:
            data["examples"][neighbor_seq_id] = (None, 0)
        else:
            best_id = max(candidate_ids, key=lambda x: sim_from_db(db, sim, neighbor_seq_id, x))
            data["examples"][neighbor_seq_id] = (best_id, sim_from_db(db, sim, neighbor_seq_id, best_id))
            data["representatives"][best_id][neighbor_seq_id] = sim_from_db(db, sim, neighbor_seq_id, best_id)
    del data["representatives"][seq_id]
    return data

summaxwithin = {"eval": summaxwithin_eval,
          "diff": summaxwithin_diff,
          "negdiff": summaxwithin_negdiff,
          "update": summaxwithin_update,
          "negupdate": summaxwithin_negupdate,
          "base_data": summaxwithin_base_data,
          "full_data": summaxwithin_full_data,
          "name": "summaxwithin"}


######################
# sumsumwithin
######################

def bisim(db, sim, seq_id1, seq_id2):
    ret = 0
    if seq_id2 in db[seq_id1]["neighbors"]:
        d = db[seq_id1]["neighbors"][seq_id2]
        ret += sim(d["log10_e"], d["pct_identical"])
    if seq_id1 in db[seq_id2]["neighbors"]:
        d = db[seq_id2]["neighbors"][seq_id1]
        ret += sim(d["log10_e"], d["pct_identical"])
    return ret

def sumsumwithin_eval(db, seq_ids, sim):
    seq_ids = set(seq_ids)
    s = 0
    for chosen_id in seq_ids:
        for neighbor, d in db[chosen_id]["neighbors"].items():
            if chosen_id == neighbor: continue
            if neighbor in seq_ids:
                s += -sim(d["log10_e"], d["pct_identical"])
    return s

sumsumwithin_base_data = lambda db, sim: set()
sumsumwithin_full_data = lambda db, sim: set(db.keys())

def sumsumwithin_diff(db, seq_id, sim, data):
    diff = 0
    data = data | set([seq_id])
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += -sim_from_neighbor(sim, d)
        #neighbor_bisim = bisim(db, sim, seq_id, neighbor)
        #diff += -neighbor_bisim
    for neighbor, d in db[seq_id]["in_neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += -sim_from_neighbor(sim, d)
    return diff

def sumsumwithin_update(db, seq_id, sim, data):
    data.add(seq_id)
    return data

def sumsumwithin_negdiff(db, seq_id, sim, data):
    diff = 0
    #data = data - set([seq_id])
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        #neighbor_bisim = bisim(db, sim, seq_id, neighbor)
        #diff -= -neighbor_bisim
        diff += sim_from_neighbor(sim, d) # removing a penalty
    for neighbor, d in db[seq_id]["in_neighbors"].items():
        if seq_id == neighbor: continue
        if not (neighbor in data): continue
        diff += sim_from_neighbor(sim, d) # removing a penalty
    return diff

def sumsumwithin_negupdate(db, seq_id, sim, data):
    data.remove(seq_id)
    return data

sumsumwithin = {"eval": sumsumwithin_eval,
          "diff": sumsumwithin_diff,
          "negdiff": sumsumwithin_negdiff,
          "update": sumsumwithin_update,
          "negupdate": sumsumwithin_negupdate,
          "base_data": sumsumwithin_base_data,
          "full_data": sumsumwithin_full_data,
          "name": "sumsumwithin"}

######################
# sumsumacross
######################

def sumsumacross_eval(db, seq_ids, sim):
    seq_ids = set(seq_ids)
    s = 0
    for chosen_id in seq_ids:
        for neighbor, d in db[chosen_id]["neighbors"].items():
            s += -sim(d["log10_e"], d["pct_identical"])
    return s

sumsumacross_base_data = lambda db, sim: None
sumsumacross_full_data = lambda db, sim: None

def sumsumacross_diff(db, seq_id, sim, data):
    diff = 0
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        diff += -sim(d["log10_e"], d["pct_identical"])
    return diff

def sumsumacross_negdiff(db, seq_id, sim, data):
    diff = 0
    for neighbor, d in db[seq_id]["neighbors"].items():
        if seq_id == neighbor: continue
        diff -= -sim(d["log10_e"], d["pct_identical"])
    return diff

def sumsumacross_update(db, seq_id, sim, data):
    raise Exception("Not used")

def sumsumacross_negupdate(db, seq_id, sim, data):
    raise Exception("Not used")

sumsumacross = {"eval": sumsumacross_eval,
          "diff": sumsumacross_diff,
          "negdiff": sumsumacross_negdiff,
          "update": sumsumacross_update,
          "negupdate": sumsumacross_negupdate,
          "base_data": sumsumacross_base_data,
          "full_data": sumsumacross_full_data,
          "name": "sumsumacross"}


######################
# graphcut
######################

graphcut = {"eval": lambda *args: sumsumacross_eval(*args) + sumsumwithin_eval(*args),
          "diff": lambda *args: sumsumacross_diff(*args) + sumsumwithin_diff(*args),
          "negdiff": lambda *args: sumsumacross_negdiff(*args) + sumsumwithin_negdiff(*args),
          "update": sumsumwithin_update,
          "negupdate": sumsumwithin_negupdate,
          "base_data": sumsumwithin_base_data,
          "full_data": sumsumwithin_full_data,
          "name": "graphcut"}

######################
# MixtureObjective
# ------------------------
# Create a mixture objective with:
# MixtureObjective([summaxacross, sumsumwithin], [0.1, 1.2])
# Must be used with a sim of the form
# [sim1, sim2]
# (same number of sims as objectives)
######################

class MixtureObjective(object):
    def __init__(self, objectives, weights):
        self.objectives = objectives
        self.weights = weights
        self.name = "mix-" + "-".join(["{0}({1})".format(objective["name"], self.weights[i]) for i,objective in enumerate(self.objectives)])

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __contains__(self, item):
        all_contain = True
        for i, objective in enumerate(self.objectives):
            all_contain = all_contain and (item in objective)
        return all_contain

    def eval(self, db, seq_ids, sims):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["eval"](db, seq_ids, sims[i])
        return s

    def diff(self, db, seq_id, sims, datas):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["diff"](db, seq_id, sims[i], datas[i])
        return s

    def negdiff(self, db, seq_id, sims, datas):
        s = 0
        for i, objective in enumerate(self.objectives):
            s += self.weights[i]*objective["negdiff"](db, seq_id, sims[i], datas[i])
        return s

    def update(self, db, seq_id, sims, datas):
        new_datas = []
        for i, objective in enumerate(self.objectives):
            new_datas.append(objective["update"](db, seq_id, sims[i], datas[i]))
        return new_datas

    def negupdate(self, db, seq_id, sims, datas):
        new_datas = []
        for i, objective in enumerate(self.objectives):
            new_datas.append(objective["negupdate"](db, seq_id, sims[i], datas[i]))
        return new_datas

    def base_data(self, db, sims):
        datas = []
        for i, objective in enumerate(self.objectives):
            datas.append(objective["base_data"](db, sims[i]))
        return datas

    def full_data(self, db, sims):
        datas = []
        for i, objective in enumerate(self.objectives):
            datas.append(objective["full_data"](db, sims[i]))
        return datas













#############################
# Optimization algorithms
# -------------------------
# Each returns either a specific
# subset or an order.
#############################

# random selection
# returns an order
def random_selection(db):
    return random.sample(db.keys(), len(db.keys()))

# naive greedy selecition
# returns an order
def naive_greedy_selection(db, objective, sim):
    not_in_repset = set(db.keys())
    repset = []
    objective_data = objective["base_data"](db, sim)
    for iteration_index in range(len(db.keys())):
        if (iteration_index % 100) == 0: logger.debug("Naive Greedy iteration: %s", iteration_index)
        best_id = None
        best_diff = None
        for seq_id_index, seq_id in enumerate(not_in_repset):
            diff = objective["diff"](db, seq_id, sim, objective_data)
            if (best_diff is None) or (diff > best_diff):
                best_diff = diff
                best_id = seq_id
        repset.append(best_id)
        not_in_repset.remove(best_id)
        objective_data = objective["update"](db, best_id, sim, objective_data)
    return repset

# returns an order
def accelerated_greedy_selection(db, objective, sim):
    repset = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    while len(pq) > 1:
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_possible_diff = -pq[0][0]

        if diff >= next_possible_diff:
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
            #assert(abs(cur_objective - objective["eval"](db, repset, sim)) < 1e-3)
            if (len(repset) % 100) == 0: logger.debug("Accelerated greedy iteration: %s", len(repset))
        else:
            heapq.heappush(pq, (-diff, seq_id))

    repset.append(pq[0][1])
    return repset

# Returns a set
def threshold_selection(db, objective, sim, diff_threshold, order_by_length=True):
    repset = [] # [{"id": id, "objective": objective}]
    objective_data = objective["base_data"](db, sim)
    if order_by_length:
        seq_ids_ordered = sorted(db.keys(), key=lambda seq_id: -len(str(db[seq_id]["seq"])))
    else:
        seq_ids_ordered = random.sample(db.keys(), len(db.keys()))
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        diff = objective["diff"](db, seq_id, sim, objective_data)
        if diff >= diff_threshold:
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
    return repset

def nonmonotone_selection(db, objective, sim, modular_bonus):
    id_order = random.sample(db.keys(), len(db.keys()))
    growing_set = set()
    shrinking_set = set(copy.deepcopy(db.keys()))
    growing_set_data = objective["base_data"](db, sim)
    shrinking_set_data = objective["full_data"](db, sim)
    shrinking_set_val = objective["eval"](db, shrinking_set, sim) + modular_bonus * len(shrinking_set)
    growing_set_val = objective["eval"](db, growing_set, sim) + modular_bonus * len(growing_set)
    cur_objective = 0
    for seq_id in id_order:
        growing_imp = objective["diff"](db, seq_id, sim, growing_set_data) + modular_bonus
        shrinking_imp = objective["negdiff"](db, seq_id, sim, shrinking_set_data) - modular_bonus
        norm_growing_imp = max(0, growing_imp)
        norm_shrinking_imp = max(0, shrinking_imp)
        if (norm_growing_imp == 0) and (norm_shrinking_imp == 0): norm_growing_imp, norm_shrinking_imp = 1,1
        if numpy.random.random() < float(norm_growing_imp) / (norm_growing_imp + norm_shrinking_imp):
            growing_set.add(seq_id)
            growing_set_val += growing_imp
            growing_set_data = objective["update"](db, seq_id, sim, growing_set_data)
            #true_growing_set_val = objective["eval"](db, growing_set, sim) + modular_bonus * len(growing_set)
            #if abs(growing_set_val - true_growing_set_val) > 1e-3:
                #logger.error("Miscalculated growing_set_val! calculated: %s ; true: %s", growing_set_val, true_growing_set_val)
        else:
            shrinking_set.remove(seq_id)
            shrinking_set_val += shrinking_imp
            shrinking_set_data = objective["negupdate"](db, seq_id, sim, shrinking_set_data)
            #true_shrinking_set_val = objective["eval"](db, shrinking_set, sim) + modular_bonus * len(shrinking_set)
            #if abs(shrinking_set_val - true_shrinking_set_val) > 1e-3:
                #logger.error("Miscalculated shrinking_set_val! calculated: %s ; true: %s", shrinking_set_val, true_shrinking_set_val)
    return growing_set



# cdhit selection
# seqs: {seq_id: seq}
# Returns a subset
def cdhit_selection(db, workdir, c=0.9):
    seqs = {seq_id: str(db[seq_id]["seq"]) for seq_id in db}

    workdir = path(workdir)
    if not workdir.exists():
        workdir.makedirs()
    infile = workdir / "in.fasta"

    with open(infile, "w") as f:
        for seq_id, seq in seqs.items():
            f.write(">{seq_id}\n".format(**locals()))
            f.write("{seq}\n".format(**locals()))

    if c > 7.0: n = "5"
    elif (c > 0.6) and (c <= 0.7): n = "4"
    elif (c > 0.5) and (c <= 0.6): n = "3"
    else: n = "2"

    outfile = path(workdir) / "out.cdhit"
    subprocess.check_call(["/net/noble/vol2/home/maxwl/Code/cdhit.git/trunk/cd-hit",
                           "-i", infile,
                           "-o", outfile,
                           "-c", str(c),
                           "-n", n,
                           "-M", "7000",
                           "-d", "0",
                           "-T", "1"])

    ret = []
    with open(outfile) as f:
        for line in f:
            if line[0] == ">":
                ret.append(int(line.strip()[1:]))

    return ret

# Like CD-HIT, but using my own implementation
# rather than calling the executable
def graph_cdhit_selection(db, sim, threshold=0.9, order_by_length=True):
    repset = set()
    if order_by_length:
        seq_ids_ordered = sorted(db.keys(), key=lambda seq_id: -len(str(db[seq_id]["seq"])))
    else:
        seq_ids_ordered = random.sample(db.keys(), len(db.keys()))
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        covered = False
        for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
            if (neighbor_seq_id in repset) and (sim_from_neighbor(sim, d) >= threshold):
                covered = True
                break
        if not covered:
            repset.add(seq_id)
    return sorted(list(repset))

# Use summaxacross to get a clustering on the sequences, then pick a random
# seq from each cluster
def cluster_selection(db, sim, k):
    assert(k < len(db.keys()))

    # Use k-medioids to get a clustering
    objective = summaxacross
    cluster_centers = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    while len(cluster_centers) < k:
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_possible_diff = -pq[0][0]

        if diff >= next_possible_diff:
            cluster_centers.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
        else:
            heapq.heappush(pq, (-diff, seq_id))

    clusters = objective_data["representatives"]

    # Choose a random sample from each cluster
    repset = []
    for i in range(k):
        clust = clusters.values()[i].keys()
        if len(clust) > 0:
            repset.append(random.choice(clust))
    return repset

# Use a clustering algorithm from sklearn
def sklearn_cluster_selection(db, db_seq_ids, db_seq_indices, sim, num_clusters_param, representative_type, cluster_type):
    #logger.info("Starting sklearn_cluster_selection: representative_type {representative_type} cluster_type {cluster_type} num_clusters_param {num_clusters_param}".format(**locals()))
    # Relevant clustering methods: Affinity prop, Spectral cluster, Agglomerative clustering
    logger.info("Starting creating similarity matrix...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    sims_matrix = numpy.zeros((len(db), len(db)))
    seq_ids = db.keys()
    for seq_id_index, seq_id in enumerate(seq_ids):
        for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
            if not (neighbor_seq_id in db): continue
            neighbor_seq_id_index = db_seq_indices[neighbor_seq_id]
            s = sim_from_neighbor(sim, d)
            prev_s = sims_matrix[seq_id_index, neighbor_seq_id_index]
            if prev_s != 0:
                s = float(s + prev_s) / 2
            sims_matrix[seq_id_index, neighbor_seq_id_index] = s
            sims_matrix[neighbor_seq_id_index, seq_id_index] = s
    logger.info("Starting running clustering...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    linkage = "average" # FIXME
    if cluster_type == "agglomerative":
        num_clusters_param = int(num_clusters_param)
        model = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters_param, affinity="precomputed", linkage=linkage)
    elif cluster_type == "affinityprop":
        model = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=num_clusters_param)
    elif cluster_type == "spectral":
        num_clusters_param = int(num_clusters_param)
        model = sklearn.cluster.SpectralClustering(n_clusters=num_clusters_param, affinity="precomputed")
    else:
        raise Exception("Unrecognized cluster_type: {cluster_type}".format(**locals()))
    try:
        cluster_ids = model.fit_predict(sims_matrix)
    except ValueError:
        # Spectral clustering breaks with ValueError when you ask for more clusters than rank of the matrix supports
        return random.sample(db.keys(), num_clusters_param)
    if numpy.isnan(cluster_ids[0]): return [] # AffinityProp sometimes breaks and just returns [nan]
    logger.info("Starting choosing repset and returning...")
    logger.info("Memory usage: %s gb", float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1000000.0 )
    cluster_ids_index = {}
    for i,c in enumerate(cluster_ids):
        if not (c in cluster_ids_index): cluster_ids_index[c] = []
        cluster_ids_index[c].append(i)
    repset = []
    if representative_type == "random":
        for c in cluster_ids_index:
            repset.append(db_seq_ids[random.choice(cluster_ids_index[c])])
    elif representative_type == "center":
        for c in cluster_ids_index:
            center_scores = {}
            cluster_seq_ids = set([db_seq_ids[seq_index] for seq_index in cluster_ids_index[c]])
            for seq_id in cluster_seq_ids:
                center_scores[seq_id] = sum([sim_from_neighbor(sim, d)
                                             for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items()
                                             if (neighbor_seq_id in cluster_seq_ids)])
            best_seq_id = max(center_scores.keys(), key=lambda seq_id: center_scores[seq_id])
            repset.append(best_seq_id)
    else:
        raise Exception("Unrecognized representative_type: {representative_type}".format(**locals()))
    return repset





#########################################
# binary_search_order
# ----------
# Attempts to sample a function y = f(x) as
# uniformly as possible WRT y.
# Usage:
# low_y = f(low_x)
# high_y = f(high_x)
# next = binary_search_order(low_x, low_y, high_x, high_y)
# next_x = (high_x - low_x) / 2
# for i in range(10):
#   next_x, low_x, low_y, high_x, high_y = next(next_x, f(next_x), low_x, low_y, high_x, high_y)
#########################################
def binary_search_order(low_x, low_y, high_x, high_y):
    # pq entry: (-abs(low_y-high_y), low_x, low_y, high_x, low_y)
    pq = []

    def binary_search_order_next(mid_x, mid_y, low_x, low_y, high_x, high_y):

        # Update heap from last time
        heapq.heappush(pq, (-abs(mid_y-low_y), low_x, low_y, mid_x, mid_y))
        heapq.heappush(pq, (-abs(high_y-mid_y), mid_x, mid_y, high_x, high_y))

        # Return next x to try
        negdiff, low_x, low_y, high_x, high_y = heapq.heappop(pq)
        mid_x = float(high_x + low_x) / 2
        return mid_x, low_x, low_y, high_x, high_y

    return binary_search_order_next

def binary_parameter_search(f, low_x, high_x, num_iterations=30):
    low_y = f(low_x)
    high_y = f(high_x)
    next = binary_search_order(low_x, low_y, high_x, high_y)
    next_x = (high_x + low_x) / 2
    for i in range(num_iterations):
        next_x, low_x, low_y, high_x, high_y = next(next_x, f(next_x), low_x, low_y, high_x, high_y)


###############################################################
###############################################################
# Run optimization and output
###############################################################
###############################################################

objective = MixtureObjective([summaxacross, sumsumwithin], [args.mixture, 1.0-args.mixture])
logger.info("-----------------------")
logger.info("Starting mixture of summaxacross and sumsumwithin with weight %s...", args.mixture)
sim, sim_name = ([fraciden, fraciden], "fraciden-fraciden")
repset_order = accelerated_greedy_selection(db, objective, sim)

with open(workdir / "repset.txt", "w") as f:
    for seq_id in repset_order:
        f.write(seq_id)
        f.write("\n")


#true_rs = ["cluster_{c}_seq_0".format(**locals()) for c in range(5)]
#obs_rs = repset_order[:5]
#print "obs_rs:", obs_rs
#print "Objective of intended set:", objective["eval"](db, true_rs, sim)
#print "Objective of intended set (summaxacross-fraciden):", summaxacross["eval"](db, true_rs, fraciden)
#print "Objective of chosen set:", objective["eval"](db, obs_rs, sim)

