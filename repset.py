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
import resource
import logging

import numpy
import scipy.sparse
import sklearn.cluster
import mysql.connector
from Bio import SeqIO

if __name__ == "__main__":
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

    # Logging
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
else:
    logger = logging.getLogger('log')

############################
# run_psiblast()
# ---------------
# Runs psiblast to get a similarity matrix between sequences.
# Input:
#   - workdir: directory for working files
#   - seqs: fasta file with sequences
# Output:
#   - db with sequence and similarity information.
# db: {seq_id: {
#               "neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "in_neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "scop": {level_id: [scop_ids]},
#               "seq": ""
#             }
#         }
def run_psiblast(workdir, seqs):
    # Create psiblast db
    if not (workdir / "db").exists():
        cmd = ["makeblastdb",
          "-in", seqs,
          "-input_type", "fasta",
          "-out", workdir / "db",
          "-dbtype", "prot"]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd)
    # Run psiblast
    if not (workdir / "psiblast_result.tab").exists():
        cmd = ["psiblast",
          "-query", seqs,
          "-db", workdir / "db",
          "-num_iterations", "6",
          "-outfmt", "6 qseqid sseqid pident length mismatch evalue bitscore",
          "-seg", "yes",
          "-out", workdir / "psiblast_result.tab"
        ]
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd)
    # Read psiblast output
    db = {}
    fasta_sequences = SeqIO.parse(open(seqs),'fasta')
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
    return db


scop_level_names = {
    1: "Root",
    2: "Class",
    3: "Fold",
    4: "Superfamily",
    5: "Family",
    6: "Protein",
    7: "Species",
    8: "PDB Entry Domain"
}

#############################
# Read astral from database
# Reads astral SQL database, returns db with similarity information
# - class_id: Restrict results to a particular SCOP Class (level 2). None for no restriction.
# - max_ids: Restrict results to max_ids ids. If class_id is also specified, will restrict number
#      of ids first, so will result in many fewer than max_ids returned.
# Returns:
# astral: {seq_id: {
#               "neighbors": {neighbor_seq_id: {"log10_e": -9999.0, "pct_identical": 100.0}},
#               "scop": {level_id: [scop_ids]},
#               "seq": ""
#             }
#         }
#############################
def read_astral_database(class_id=None, max_ids=None):
    raise Exception("TODO: do not symmetrize; add in_neighbors instead")
    time_before = time.time()
    def connect_db(connect_timeout=30):
        cnx = mysql.connector.connect(host="guanine.gs.washington.edu",
                                      user="root",
                                      port=3306,
                                      connect_timeout=connect_timeout,
                                      buffered=True,
                                      get_warnings=True,
                                      raise_on_warnings=True)

        cursor = cnx.cursor(dictionary=True)
        cursor.execute("USE scop")
        return cnx, cursor

    cnx, cursor = connect_db(connect_timeout=1200)

    # is this necessary?
    query ="SET GLOBAL max_allowed_packet=33554432"
    logger.info(query)
    cursor.execute(query)

    query ="SET GLOBAL net_read_timeout=3600"
    logger.info(query)
    cursor.execute(query)

    query ="SET GLOBAL net_write_timeout=3600"
    logger.info(query)
    cursor.execute(query)

    query ="SET GLOBAL connect_timeout=3600"
    logger.info(query)
    cursor.execute(query)

    query ="SET GLOBAL innodb_lock_wait_timeout=3600"
    logger.info(query)
    cursor.execute(query)

    query ="SET GLOBAL innodb_flush_log_at_timeout=2000"
    logger.info(query)
    cursor.execute(query)

    #############################
    # Read astral ids
    #############################
    logger.info("Starting reading astral ids...")
    if not (max_ids is None):
        assert(max_ids > 0)
        query = "SELECT id AS astral_seq_id, seq FROM astral_seq ORDER BY RAND() LIMIT {max_ids}".format(**locals())
    else:
        query = "SELECT id AS astral_seq_id, seq FROM astral_seq"

    logger.info(query)
    time_before = time.time()
    cursor.execute(query)
    astral = {row["astral_seq_id"]: {"seq": row["seq"]} for row in cursor}

    logger.info("Time to read astral ids: %s", time.time()-time_before )


    #############################
    # Read SCOP classes
    #############################
    logger.info("Starting reading SCOP classes...")
    time_before = time.time()
    query = """
SELECT t1.seq_id, t1.domain_id, t1.node_id, scop_node.release_id, scop_node.level_id FROM
(
    SELECT astral_seq.id AS seq_id, astral_domain.id AS domain_id, astral_domain.node_id AS node_id
    FROM astral_seq
    LEFT JOIN astral_domain ON astral_seq.id = astral_domain.seq_id
    WHERE astral_seq.id IN ({astral_ids_str})
) as t1
LEFT JOIN scop_node ON t1.node_id = scop_node.id
""".format(astral_ids_str=",".join(map(str, astral.keys())))
    time_before_query = time.time()
    cursor.execute(query)

    # nodes_by_level: {scop_level: {seq_id: [scop_id]}}
    nodes_by_level = {scop_level: {} for scop_level in range(1,9)}
    num_level1 = 0
    num_total = 0
    for row in cursor:
        if not ((row["level_id"] is None) or (row["level_id"] == 8)):
            num_level1 += 1
            num_total += 1
            continue

        if not (row["seq_id"] in nodes_by_level[8]):
            nodes_by_level[8][row["seq_id"]] = set()
        if row["release_id"] == 15:
            nodes_by_level[8][row["seq_id"]].add(row["node_id"])
            num_total += 1
    logger.warning("Number of weird level-1-associated astral domains: %s / %s", num_level1, num_total)
    logger.info("Time to read level-8 SCOP: %s", time.time()-time_before )

    # Recursively get level 7-1 nodes
    for scop_level in range(7,0,-1):
        logger.info("Starting level %s...", scop_level)
        query_nodes = set.union(*nodes_by_level[scop_level+1].values())
        if len(query_nodes) == 0:
            nodes_by_level[scop_level] = {seq_id: set() for seq_id in nodes_by_level[scop_level+1]}
        else:
            query = """
SELECT id AS node_id, level_id, parent_node_id, description, release_id FROM
scop_node
WHERE id IN ({query_nodes_str}) AND release_id = 15
""".format(query_nodes_str=",".join(map(str,query_nodes)))
            cursor.execute(query)

            nodes_reverse_index = {} # {node_id: [seq_id]}
            for seq_id, node_list in nodes_by_level[scop_level+1].items():
                for node_id in node_list:
                    if not (node_id in nodes_reverse_index):
                        nodes_reverse_index[node_id] = []
                    nodes_reverse_index[node_id].append(seq_id)

            nodes_by_level[scop_level] = {seq_id: set() for seq_id in nodes_by_level[scop_level+1]}
            for row in cursor:
                for seq_id in nodes_reverse_index[row["node_id"]]:
                    nodes_by_level[scop_level][seq_id].add(row["parent_node_id"])

    # convert nodes_by_level to seq_id-centric structure
    num_skipped = 0
    total = 0
    for scop_level_id in nodes_by_level:
        for seq_id in nodes_by_level[scop_level_id]:
            if seq_id in astral:
                if not "scop" in astral[seq_id]:
                    astral[seq_id]["scop"] = {}
                astral[seq_id]["scop"][scop_level_id] = nodes_by_level[scop_level_id][seq_id]
            else:
                num_skipped += 1
            total += 1
    logger.info("Converting nodes_by_level to seq_id-centric form: Skipped %s/%s entries", num_skipped, num_total)
    logger.info("Time to read SCOP: %s", time.time()-time_before )

    #############################
    # Remove sequences with no release-15 level-5 class and restrict to class_id
    #############################
    logger.info("Number of sequences before removing: %s", len(astral))
    astral = {seq_id: x for seq_id, x in astral.items() if
              ("scop" in x and len(x["scop"][5]) >= 1)}

    if not (class_id is None):
        astral = {seq_id: x for seq_id, x in astral.items() if
                  class_id in x["scop"][2] }
    logger.info("Number of sequences after removing: %s", len(astral))


    #############################
    # Read BLAST graph
    #############################
    logger.info("Starting reading BLAST graph...")
    time_before = time.time()
    for seq_id in astral: astral[seq_id]["neighbors"] = {}

    # add self-edges
    for seq_id in astral:
        astral[seq_id]["neighbors"][seq_id] = {"log10_e": -9999.0, "pct_identical": 100.0}

    # For some reason, there seems to be a limit on the number of rows I can get in a query
    # I'm sure there's a right way to do this, but I can't figure it out.
    # Instead, I'm just going to break up my queries into chunks.
    chunk_size = 10
    cnx, cursor = connect_db(connect_timeout=10)
    #for chunk_start in range(0, len(astral), chunk_size):
    chunk_start = 0
    chunk_end = chunk_start + chunk_size
    while chunk_start < len(astral.keys()):
        logger.info("Starting chunk %s-%s / %s", chunk_start, chunk_end, len(astral))
        #astral_ids_chunk = astral.keys()[chunk_start:min(chunk_start+chunk_size,len(astral.keys()))]
        astral_ids_chunk = astral.keys()[chunk_start:chunk_end]

        query = """
SELECT seq1_id, seq2_id, blast_log10_e, pct_identical FROM astral_seq_blast
WHERE seq1_id IN ({astral_ids_chunk_str}) AND seq2_id IN ({astral_ids_str}) AND release_id = 15
""".format(astral_ids_chunk_str=",".join(map(str, astral_ids_chunk)),
           astral_ids_str=",".join(map(str, astral.keys())))
        #logger.debug(query)
        time_before_query = time.time()

        try:
            cursor.execute(query)
            num_blast_rows = 0
            for row_index, row in enumerate(cursor):
                seq1_id = row["seq1_id"]
                seq2_id = row["seq2_id"]
                log10_e = row["blast_log10_e"]
                pct_identical = row["pct_identical"]
                assert(seq1_id in astral)
                if seq2_id in astral:
                    if seq2_id in astral[seq1_id]["neighbors"]:
                        pass
                        #if abs(log10_e - astral[seq1_id]["neighbors"][seq2_id]["log10_e"]) > 0.1:
                            #logger.error("log10_e values to not match: %s vs. %s", log10_e, astral[seq1_id]["neighbors"][seq2_id]["log10_e"])
                        #else:
                            #logger.error("log10_e values match")
                        #if abs(pct_identical - astral[seq1_id]["neighbors"][seq2_id]["pct_identical"]) > 0.1:
                            #logger.error("pct_identical values to not match: %s vs. %s", pct_identical, astral[seq1_id]["neighbors"][seq2_id]["pct_identical"])
                        #else:
                            #logger.error("pct_identical values match")
                    else:
                        astral[seq1_id]["neighbors"][seq2_id] = {"log10_e": log10_e, "pct_identical": pct_identical}
                        astral[seq2_id]["neighbors"][seq1_id] = {"log10_e": log10_e, "pct_identical": pct_identical}
                num_blast_rows += 1
            logger.debug("Time for query: %s (%s rows)", time.time()-time_before_query, num_blast_rows)
            chunk_start = chunk_end
            chunk_end = chunk_start + chunk_size
        except mysql.connector.errors.OperationalError as e:
            logger.warning("Ignoring error and renewing connection -- mysql.connector.errors.OperationalError: %s", e)
            cnx, cursor = connect_db(connect_timeout=30)
            chunk_end = chunk_start + ((chunk_end - chunk_start) // 2) + 1
        except ValueError as e:
            logger.warning("Ignoring error: %s", e)
            cnx, cursor = connect_db(connect_timeout=30)
        except IndexError as e:
            logger.warning("Ignoring error: %s", e)
            cnx, cursor = connect_db(connect_timeout=30)
            chunk_end = chunk_start + ((chunk_end - chunk_start) // 2) + 1
    logger.info("Time to read BLAST graph: %s", time.time()-time_before )

    return astral

# Try reading the database, starting over on failure
def read_astral_database_retries(*args, **kwargs):
    try:
        read_astral_database(*args, **kwargs)
    except Exception as e:
        logger.critical("---- Ignoring error and restarting database read: %s ----", e)
        read_astral_database_retries(*args, **kwargs)


#############################
# Synthetic astral-like database
#############################
def synthetic_astral(num_examples=100, num_centers=10):
    astral = {i: {} for i in range(num_examples)}
    # Add self-edges and filler seq and scop info
    for i in astral:
        astral[i]["seq"] = "AA"
        astral[i]["neighbors"] = {i: {"log10_e": -9999.0, "pct_identical": 100.0}}
        astral[i]["in_neighbors"] = {i: {"log10_e": -9999.0, "pct_identical": 100.0}}
    for i in range(num_centers):
        astral[i]["scop"] = {l:set([str(i)]) for l in scop_level_names}
    for i in range(num_centers, num_examples):
        # Add edge to a center
        center_id = random.randrange(num_centers)
        astral[i]["scop"] = astral[center_id]["scop"]
        astral[i]["neighbors"][center_id] = {"log10_e": -9999.0, "pct_identical": 95.0}
        astral[center_id]["in_neighbors"][i] = {"log10_e": -9999.0, "pct_identical": 95.0}
        # Add a worse edge to a non-center
        noncenter_id = random.randrange(num_examples)
        astral[i]["neighbors"][noncenter_id] = {"log10_e": -9999.0, "pct_identical": 50.0}
        astral[noncenter_id]["in_neighbors"][i] = {"log10_e": -9999.0, "pct_identical": 50.0}
    return astral



#############################
# Numscop
#############################
def numscop(db, seq_ids, scop_level):
    return len({list(db[seq_id]["scop"][scop_level])[0] for seq_id in seq_ids})


###############################################################
###############################################################
# Submodular optimization functions
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
            if (neighbor_seq_id in seq_ids_set) and (neighbor_seq_id != chosen_seq_id):
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
    #raise Exception("Not used")
    return None

def sumsumacross_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

sumsumacross = {"eval": sumsumacross_eval,
          "diff": sumsumacross_diff,
          "negdiff": sumsumacross_negdiff,
          "update": sumsumacross_update,
          "negupdate": sumsumacross_negupdate,
          "base_data": sumsumacross_base_data,
          "full_data": sumsumacross_full_data,
          "name": "sumsumacross"}

######################
# lengthobj
######################

def lengthobj_eval(db, seq_ids, sim):
    s = 0
    for chosen_id in seq_ids:
        s += len(db[chosen_id]["seq"])
    return s

lengthobj_base_data = lambda db, sim: None
lengthobj_full_data = lambda db, sim: None

def lengthobj_diff(db, seq_id, sim, data):
    return len(db[seq_id]["seq"])

def lengthobj_negdiff(db, seq_id, sim, data):
    return len(db[seq_id]["seq"])

def lengthobj_update(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

def lengthobj_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

lengthobj = {"eval": lengthobj_eval,
          "diff": lengthobj_diff,
          "negdiff": lengthobj_negdiff,
          "update": lengthobj_update,
          "negupdate": lengthobj_negupdate,
          "base_data": lengthobj_base_data,
          "full_data": lengthobj_full_data,
          "name": "lengthobj"}



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
# uniform
######################
def uniform_eval(db, seq_ids, sim):
    return len(seq_ids)

uniform_base_data = lambda db, sim: None
uniform_full_data = lambda db, sim: None

def uniform_diff(db, seq_id, sim, data):
    return 1

def uniform_negdiff(db, seq_id, sim, data):
    return -1

def uniform_update(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

def uniform_negupdate(db, seq_id, sim, data):
    #raise Exception("Not used")
    return None

uniform = {"eval": uniform_eval,
          "diff": uniform_diff,
          "negdiff": uniform_negdiff,
          "update": uniform_update,
          "negupdate": uniform_negupdate,
          "base_data": uniform_base_data,
          "full_data": uniform_full_data,
          "name": "uniform"}

######################
# sumsumwithin-complement
# XXX deprecated given complement_greedy_selection
######################

if False:
    def sumsumwithincomp_eval(db, seq_ids, sim):
        seq_ids = set(seq_ids)
        complement = set(db.keys()) - seq_ids
        s = 0
        for comp_id in complement:
            for neighbor, d in db[comp_id]["neighbors"].items():
                if comp_id == neighbor: continue
                if neighbor in complement:
                    s += -sim(d["log10_e"], d["pct_identical"])
        return s

    sumsumwithincomp_base_data = lambda db, sim: set(db.keys())
    sumsumwithincomp_full_data = lambda db, sim: set()

    def sumsumwithincomp_diff(db, seq_id, sim, data):
        # data: set of ids in complement
        diff = 0
        #data = data - set([seq_id])
        for neighbor, d in db[seq_id]["neighbors"].items():
            if seq_id == neighbor: continue
            if not (neighbor in data): continue
            diff += sim_from_neighbor(sim, d)
        for neighbor, d in db[seq_id]["in_neighbors"].items():
            if seq_id == neighbor: continue
            if not (neighbor in data): continue
            diff += sim_from_neighbor(sim, d)
        return diff

    def sumsumwithincomp_update(db, seq_id, sim, data):
        data.remove(seq_id)
        return data

    def sumsumwithincomp_negdiff(db, seq_id, sim, data):
        diff = 0
        #data = data | set([seq_id])
        for neighbor, d in db[seq_id]["neighbors"].items():
            if seq_id == neighbor: continue
            if not (neighbor in data): continue
            #neighbor_bisim = bisim(db, sim, seq_id, neighbor)
            #diff -= -neighbor_bisim
            diff += -sim_from_neighbor(sim, d)
        for neighbor, d in db[seq_id]["in_neighbors"].items():
            if seq_id == neighbor: continue
            if not (neighbor in data): continue
            diff += -sim_from_neighbor(sim, d)
        return diff

    def sumsumwithincomp_negupdate(db, seq_id, sim, data):
        data.add(seq_id)
        return data

    sumsumwithincomp = {"eval": sumsumwithincomp_eval,
              "diff": sumsumwithincomp_diff,
              "negdiff": sumsumwithincomp_negdiff,
              "update": sumsumwithincomp_update,
              "negupdate": sumsumwithincomp_negupdate,
              "base_data": sumsumwithincomp_base_data,
              "full_data": sumsumwithincomp_full_data,
              "name": "sumsumwithincomp"}


######################
# Set cover objective
######################

class SetCover(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.name = "setcover-" + str(self.threshold)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def eval(self, db, seq_ids, sim):
        s = 0
        for seq_id in db:
            for neighbor_seq_id, d in db[seq_id]["neighbors"].items():
                if sim_from_neighbor(sim, d) >= self.threshold:
                    s += 1
                    break
        return s

    def diff(self, db, seq_id, sim, data):
        s = 0
        for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
            if neighbor_seq_id in data:
                if not data[neighbor_seq_id]:
                    if sim_from_neighbor(sim, d) >= self.threshold:
                        s += 1
        return s

    def negdiff(self, db, seq_id, sim, data):
        raise Exception()

    def update(self, db, seq_id, sim, data):
        for neighbor_seq_id, d in db[seq_id]["in_neighbors"].items():
            if neighbor_seq_id in data:
                if not data[neighbor_seq_id]:
                    if sim_from_neighbor(sim, d) >= self.threshold:
                        data[neighbor_seq_id] = True
        return data

    def negupdate(self, db, seq_id, sim, datas):
        raise Exception()

    def base_data(self, db, sim):
        return {seq_id: False for seq_id in db}

    def full_data(self, db, sim):
        return {seq_id: (sim_from_db(db, sim, seq_id, seq_id) >= self.threshold)
                for seq_id in db}


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
def accelerated_greedy_selection(db, objective, sim, max_evals=float("inf"), diff_approx_ratio=1.0, repset_size=float("inf"), target_obj_val=float("inf")):
    assert diff_approx_ratio <= 1.0
    repset = []
    pq = [(-float("inf"), seq_id) for seq_id in db]
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    num_evals = 0
    while (len(repset) < repset_size) and (len(pq) > 1) and (cur_objective < target_obj_val):
        possible_diff, seq_id = heapq.heappop(pq)
        diff = objective["diff"](db, seq_id, sim, objective_data)
        next_diff = -pq[0][0]
        num_evals += 1
        if (num_evals >= max_evals) or (((diff - next_diff) / (abs(diff)+0.01)) >= (diff_approx_ratio - 1.0)):
            repset.append(seq_id)
            objective_data = objective["update"](db, seq_id, sim, objective_data)
            cur_objective += diff
            #assert(abs(cur_objective - objective["eval"](db, repset, sim)) < 1e-3)
            #if (len(repset) % 100) == 0: logger.debug("Accelerated greedy iteration: %s", len(repset))
            #print len(repset), num_evals # XXX
            #print len(repset), cur_objective
            num_evals = 0
        else:
            heapq.heappush(pq, (-diff, seq_id))
    if len(pq) == 1:
        repset.append(pq[0][1])
    return repset

def complement_greedy_selection(db, objective, sim):
    complement_objective = {}
    complement = lambda seq_ids: set(db.keys()) - set(seq_ids)
    complement_objective["eval"] = lambda db, seq_ids, sim: objective["eval"](db, complement(seq_ids), sim)
    complement_objective["diff"] = lambda db, seq_id, sim, data: objective["negdiff"](db, seq_id, sim, data)
    complement_objective["negdiff"] = lambda db, seq_id, sim, data: objective["diff"](db, seq_id, sim, data)
    complement_objective["update"] = lambda db, seq_id, sim, data: objective["negupdate"](db, seq_id, sim, data)
    complement_objective["negupdate"] = lambda db, seq_id, sim, data: objective["update"](db, seq_id, sim, data)
    complement_objective["base_data"] = lambda db, sim: objective["full_data"](db, sim)
    complement_objective["full_data"] = lambda db, sim: objective["base_data"](db, sim)
    repset_order = accelerated_greedy_selection(db, complement_objective, sim)
    return repset_order[::-1]


def stochastic_greedy_selection(db, objective, sim, sample_size, repset_size=float("inf")):
    repset = []
    objective_data = objective["base_data"](db, sim)
    cur_objective = 0
    possible = {seq_id: float("inf") for seq_id in db}
    choosable_seq_ids = set(db.keys())
    sample_size = int(sample_size)
    next_log = 10
    for iter_index in range(min(repset_size, len(db))):
        if iter_index >= next_log:
            if not (logger is None):
                logger.info("stochastic_greedy_selection {} / {}".format(iter_index, min(repset_size, len(db))))
            next_log *= 1.3
        sample = random.sample(choosable_seq_ids, min(sample_size, len(choosable_seq_ids)))
        sample = sorted(sample, key=lambda seq_id: possible[seq_id], reverse=True)
        cur_best_diff = float("-inf")
        cur_best_seq_id = None
        for seq_id in sample:
            if cur_best_diff >= possible[seq_id]:
                break
            diff = objective["diff"](db, seq_id, sim, objective_data)
            possible[seq_id] = diff
            if diff > cur_best_diff:
                cur_best_diff = diff
                cur_best_seq_id = seq_id
        repset.append(cur_best_seq_id)
        choosable_seq_ids.remove(cur_best_seq_id)
        objective_data = objective["update"](db, cur_best_seq_id, sim, objective_data)
        cur_objective += cur_best_diff
    #assert len(repset) == len(db)
    return repset



# Returns a set
# Like graphcdhit_selection, but works for arbitrary objectives
# Uses objective["diff"]
def threshold_selection(db, objective, sim, diff_threshold, order_by_length=True):
    repset = [] # [{"id": id, "objective": objective}]
    objective_data = objective["base_data"](db, sim)
    if order_by_length:
        seq_ids_ordered = sorted(db.keys(), key=lambda seq_id: -len(str(db[seq_id]["seq"])))
    else:
        seq_ids_ordered = random.sample(db.keys(), len(db.keys()))
    next_log = 10
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        if iteration_index >= next_log:
            if not (logger is None):
                logger.info("threshold_selection {} / {}".format(iteration_index, len(seq_ids_ordered)))
            next_log *= 1.3
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
    next_log = 10
    for iteration_index, seq_id in enumerate(seq_ids_ordered):
        if iteration_index >= next_log:
            if not (logger is None):
                logger.info("graph_cdhit_selection {} / {}".format(iteration_index, len(seq_ids_ordered)))
            next_log *= 1.3
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
        next_diff = -pq[0][0]

        if diff >= next_diff:
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

###############################################################
###############################################################
# Utility functions
###############################################################
###############################################################


# Either randomly adds or removes elements from a repset
# to match the target size
def random_topup(db, rs, target_repset_size):
    target_repset_size = int(target_repset_size)
    if len(rs) > target_repset_size:
        return random.sample(rs, target_repset_size)
    elif len(rs) < target_repset_size:
        return list(rs) + list(random.sample(set(db.keys()) - set(rs), target_repset_size-len(rs)))
    else:
        return rs

#########################################
# binary_search_repset_size
# ----------------------------
# Performs binary search to find a repset of size
# exactly equal to target.
# Usage:
# db = input database
# f = A function such that f(param) -> repset
# target = target repset size
# param_lower = lowest value of param to check. We assume that higher param <=> larger repset
# param_upper = highest value of param to check
# tolerance = If we find a repset of target<=size<target+tolerance, return sample(repset, target)
# max_iters = maximum binary search iterations to attempt
#########################################
def binary_search_repset_size(db, f, target, param_lower, param_upper, max_iters=10, tolerance=3, final_tolerance=float("inf")):
    # find range that includes target
    # use iters in increments of 2 because we have to compute f() twice at each iter
    for iter_index in range(0,int(float(max_iters)/2),2):
        size_upper = len(f(param_upper))
        size_lower = len(f(param_lower))
        if (size_lower < target and size_upper < target) or (size_lower > target and size_upper > target):
            param_upper += param_upper - param_lower
            param_lower -= param_upper - param_lower
        else:
            break
    range_search_iters = iter_index
    # find target within range
    for iter_index in range(range_search_iters, max_iters):
        try_param = float(param_upper + param_lower) / 2
        rs = f(try_param)
        print param_lower, param_upper, try_param, len(rs), target
        if abs(len(rs) - target) <= tolerance:
            ret_rs = rs
            break
        elif len(rs) > target:
            param_upper = try_param
        elif len(rs) < target:
            param_lower = try_param
    if abs(len(rs) - final_tolerance) <= final_tolerance:
        ret_rs = rs
    else:
        len_rs = len(rs)
        raise(Exception("binary_search_repset_size failed to find a suitable param. Final parameters: param_lower={param_lower}; param_upper={param_upper}; len(rs)={len_rs}; target={target}".format(**locals())))
    ret_rs = random_topup(db, ret_rs, target)
    return {"rs": ret_rs, "iters": iter_index+1, "param": try_param}



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
    if low_y is None:
        raise Exception("low_y is None -- maybe faulty return value from f()")
    high_y = f(high_x)
    get_next = binary_search_order(low_x, low_y, high_x, high_y)
    next_x = (high_x + low_x) / 2
    for i in range(num_iterations):
        next_x, low_x, low_y, high_x, high_y = get_next(next_x, f(next_x), low_x, low_y, high_x, high_y)



###############################################################
###############################################################
# Run optimization and output
###############################################################
###############################################################

if __name__ == "__main__":
    db = run_psiblast(workdir, args.seqs)
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


