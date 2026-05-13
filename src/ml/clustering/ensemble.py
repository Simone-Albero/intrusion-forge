if __name__ == '__main__':
    from base import *
else:
    from .base import *

from time import perf_counter_ns
import os
from pathlib import Path

# TESTING
from collections import defaultdict
from collections import defaultdict
import numpy as np


def class_index_sets(arr):
    '''
    prende array e returna un dizionario dove ogni cluster id è la chiave
    e value la lista degli indici in cui appare
    '''
    groups = defaultdict(set)

    for idx, cls in enumerate(arr):
        groups[cls].add(idx)

    return groups


def compare_clusters(cl1_di_labels_list : dict[int, list[int]], cl2_ri2cid : dict[int, list[int]], cluster_mappings : dict[int, int]):
    '''
    prende cl1 e cl2 da class_index_sets e usando cluster_mappings cerca di capire
    se i cluster sono identici

    cluster_mappings ha key il cluster di cl2_ri2cid e value quello di cl1_di_labels_list
    '''
    res = True    

    for clid2 in list(cl2_ri2cid.keys())[:]:
        if cluster_mappings[clid2] == "skip this":
            print("SKIPPED SENTINEL...")
            continue 
        clid1 = cluster_mappings[clid2] # il cluster id di algoritmo_clust_usato
        indexes2 = cl2_ri2cid[clid2]
        indexes1 = cl1_di_labels_list[clid1]

        print(f"DIFFERENT LEN: algoritmo_clust_usato={len(indexes1)}, ensemble={len(indexes2)}") if len(indexes1) != len(indexes2) else None

        only_in_indexes1 = set(indexes1) - set(indexes2)
        only_in_indexes2 = set(indexes2) - set(indexes1)

        print(f"Only in indexes algoritmo_clust_usato (ens={clid2},hdb={clid1}) -> {len(only_in_indexes1)} items.") if len(only_in_indexes1) > 0 else None
        print(f"Only in indexes ensemble (ens={clid2},hdb={clid1}) -> {len(only_in_indexes2)} items.") if len(only_in_indexes2) > 0 else None

        if len(only_in_indexes1)!=0 or len(only_in_indexes2)!=0:
            res = False

        del cl2_ri2cid[clid2]
        del cl1_di_labels_list[clid1]

    if len(cl1_di_labels_list.keys())!=0:
        print(f"Rimasta roba in cl1: {cl1_di_labels_list.keys()}")
        res = False

    if len(cl2_ri2cid.keys())!=0:
        print(f"Rimasta roba in cl2: {cl2_ri2cid.keys()}")
        res = False

    return res



# END OF TESTING


def are_the_labels_all_the_same_len(labels_list: list[np.ndarray]):
    length = labels_list[0].shape[0]
    for label in labels_list:
        if length != label.shape[0]:
            return False
    return True


def split_into_ranges(range_start, range_end, amount_of_ranges_to_split_into) -> list[tuple[int, int]]:
    """
    Splits a numeric interval [range_start, range_end) into approximately equal
    contiguous sub-ranges.

    The function returns a list of tuples representing sub-ranges such as:
    [(start1, end1), (start2, end2), ...] covering the full interval without gaps.

    If the interval cannot be divided evenly, the first ranges receive one extra unit.
    """

    total_length = range_end - range_start
    base = total_length // amount_of_ranges_to_split_into
    remainder = total_length % amount_of_ranges_to_split_into

    ranges = []
    start = range_start

    for i in range(amount_of_ranges_to_split_into):
        end = start + base + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end

    return ranges


def assign_cluster_id_to_array_in_place(row_index_to_cluster_id : np.ndarray, 
                                        labels_list : list[np.ndarray], 
                                        abstained_label_per_labels : list,
                                        first_compare_index : int, 
                                        start_index : int,
                                        end_index : int, 
                                        current_cluster_id : int,
                                        threshold : float):
    '''
    Takes the row_index_to_cluster_id and modifies it in-place, assigning the current_cluster_id to all the
    successive rows to the start_index until end_index, if and only if that row has the same label 
    of label[first_compare_index].

    The belonging to a cluster is decided by majority vote 
    '''
    labels_amount = float(len(labels_list)) # The amount of clustering algorithms used, inferred by how many labels are received
    
    for comparative_index in range(start_index, end_index):
        # print(f"Idx1={first_compare_index},Cl_ID1={current_cluster_id},Idx2={comparative_index}/{end_index},", end='\r', flush=True)

        votes_for_same_cluster = 0.0 # How many clustering algorithms vote for the two rows to be in the same cluster
        # Find the vote of each clustering algorithm
        for i in range(len(labels_list)):
            label = labels_list[i]
            
            current_index_cluster_id_in_this_label = label[first_compare_index]
            comparative_index_cluster_id_in_this_label = label[comparative_index]
            # print('\n', i, '->', current_index_cluster_id_in_this_label, '==', comparative_index_cluster_id_in_this_label)
            
            # TODO ABSTAINED HANDLING: if the current labeling for the comparative index is "abstained",
            # then no vote is given at all
            if abstained_label_per_labels is not None:
                current_abstained_label = abstained_label_per_labels[i]
                if current_abstained_label is not None and comparative_index_cluster_id_in_this_label == current_abstained_label:
                    continue

            # Are they assigned to the same cluster?
            if current_index_cluster_id_in_this_label == comparative_index_cluster_id_in_this_label:
                votes_for_same_cluster += 1 # Increase the votes in favor
             
        # Once votes are given, get the final verdict
        if votes_for_same_cluster/labels_amount >= threshold:
            # print(f"Had {votes_for_same_cluster} votes with {labels_amount} labels, and threshold {threshold}")
            # Same clusters, so, fill the final array
            row_index_to_cluster_id[comparative_index] = current_cluster_id
            # print(f"Idx1={first_compare_index},Cl_ID1={current_cluster_id},Idx2={comparative_index}/{end_index},Cl_ID2=ASSIGNED", end='\r', flush=True)
        else:
            # print(f"Idx1={first_compare_index},Cl_ID1={current_cluster_id},Idx2={comparative_index}/{end_index},Cl_ID2=--------", end='\r', flush=True)
            pass


def compute_ensemble_labels(labels_list: list[np.ndarray],
                            abstained_label_per_labels: list,
                            threshold: float = 0.5) -> np.ndarray:
    '''

    Assign a cluster id to all of the rows of the dataframe, starting by labels_list which is
    a list of all the labels assigned by multiple clustering algorithms, by majority vote.

    The space complexity is O(n), n being the rows amount, while time complexity is O(clusters_num * n) = O(n).
    This is because the iteration over the rows is done once per cluster, as for each iteration an entire
    cluster is assigned to the components of it.

    If cluster_num is close to n, it becomes O(n^2) which is highly infeasible, but this scenario is very
    unreasonable to happen.

    You should always use threads_number = 1.

    TODO chiedi a simone dove vanno a finire le row senza cluster, tipo quelle che hdbscan
    mette a -1

    params:
        labels_list: list of the labels assigned by each clustering algorithm. One label per clustering algorithm.
        abstained_label_per_labels: list of the label that, for each algorithm, states that a row has no cluster assigned to it.
                                    for example, hbscan would have -1, converted to the correct type used.
                                    If no abstained label is used for an algorithm, use None in that index.
                                    If no abstained labels are used at all, use None for this parameter entirely.
        threshold: used to assign a cluster to a row. each algorithm votes, and then the threshold is compared to the mean vote

    '''

    # Check if the labels all have the same length
    if not are_the_labels_all_the_same_len(labels_list=labels_list):
        raise ValueError(f"The labels have different lengths. Cannot ensemble: {[label.shape[0] for label in labels_list]}")

    rows_amount = labels_list[0].shape[0] # Number of rows in the dataframe.
    # Cannot use -1 to fill it, because it is used by some algorithms.
    sentinel = np.iinfo(np.int32).min # The value that is used as a filler to mark a "unfilled" array cell
    row_index_to_cluster_id = np.full(rows_amount, sentinel, dtype=np.int32) # Array that maps a row index to its cluster id

    next_clust_id = np.int32(0) # The id to assign to the next cluster, if a new one is found

    start_time = perf_counter_ns() # TODO remove.
    cluster_mappings = {sentinel : "skip this"} # TODO remove. mappa clust id nuovi ai vecchi

    # Iterate over each row by index, and assign its cluster instantly
    for current_index in range(rows_amount):
        current_index_time_start = perf_counter_ns() # TODO remove.
        # print(f"Idx1={current_index},", end='\r', flush=True) # TODO remove.
        current_cluster_id = row_index_to_cluster_id[current_index] # Get the current cluster id, if it was assigned
        # If no cluster id is given, assign it the next cluster id, which initiates a new cluster
        if current_cluster_id == sentinel:
            current_cluster_id = next_clust_id # Assign the correct current cluster id for the iteration
            cluster_mappings[current_cluster_id] = labels_list[0][current_index] # TODO remove.
            row_index_to_cluster_id[current_index] = current_cluster_id # Fill the final array in the current index p
            next_clust_id += 1 # Increase the cluster id for a future new cluster
        else:
            # A cluster id has been assigned already, I can skip this row
            continue
            
        # print(f"Idx1={current_index},Cl_ID1={current_cluster_id}", end='\r', flush=True)

        # Assign the cluster of the next elements of row_index_to_cluster_id, if and only if
        # they are assigned to the same cluster by MAJORITY vote
        assign_cluster_id_to_array_in_place(
            row_index_to_cluster_id=row_index_to_cluster_id, 
            labels_list=labels_list, 
            abstained_label_per_labels=abstained_label_per_labels,
            first_compare_index=current_index,
            start_index=current_index+1, 
            end_index=rows_amount, 
            current_cluster_id=current_cluster_id,
            threshold=threshold
        )

        current_index_time_elapsed = perf_counter_ns() - current_index_time_start
        # print(f"IT TOOK {current_index_time_elapsed} FOR INDEX {current_index}/{rows_amount}.")

    for el in row_index_to_cluster_id:
        assert(el != sentinel)

    end_time_elapsed = perf_counter_ns() - start_time

    print(f"IT TOOK {end_time_elapsed} FOR THE WHOLE THINGY")
    max_clusters = 0
    for label in labels_list:
        if np.unique(label).size > max_clusters:
            max_clusters = np.unique(label).size

    print("ALGO_USATO CHECK SINGLE")
    import pandas as pd
    df = pd.DataFrame(labels_list[0], columns=["algoritmo_clust_usato"])
    print(df.value_counts())
    df = pd.DataFrame(row_index_to_cluster_id, columns=["ri2cid"])
    print(df.value_counts())
    
    if compare_clusters(cl1_di_labels_list=class_index_sets(arr=labels_list[0]), cl2_ri2cid=class_index_sets(row_index_to_cluster_id), cluster_mappings=cluster_mappings):
        print("YES THE SAME")
    else:
        print("NO DIFFERENT... DIDNT WORK MAN")
    exit("CIAONE") 
    return row_index_to_cluster_id

def make_ensemble_cluster_fn(
    cluster_fns: list[ClusterFn],
    threshold: float = 0.5,
) -> ClusterFn:
    """Compose multiple ClusterFns via consensus. Returns a single ClusterFn."""

    def _fn(X: np.ndarray) -> np.ndarray:
        labels_list = [cluster_fn(X) for cluster_fn in cluster_fns]

        # save
        filepath = Path(os.path.abspath(os.path.dirname(__file__))) / 'kmeans_gmm.npz' 
        np.savez(filepath, *labels_list)
        exit("SAVED KMEANS")
        
        labels = compute_ensemble_labels(labels_list=labels_list, threshold=threshold)
        return labels

    return _fn



if __name__ == '__main__':
    # load
    filepatha = Path(os.path.abspath(os.path.dirname(__file__))) / 'kmeans_gmm.npz' 
    loadeda = np.load(filepatha)
    labels_lista = [loadeda[key] for key in loadeda]

    # abstain labels
    hdbscan_abstained_lista = [np.int32(-1)] # None
    abstained_lista = None

    # ensemble
    labelsa = compute_ensemble_labels(labels_list=labels_lista, 
                                      abstained_label_per_labels=abstained_lista)