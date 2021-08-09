import torch
import networkx
import matplotlib.pyplot as plt
import pprint
from time import time
from itertools import combinations, combinations_with_replacement, permutations, product
from typing import Dict, Tuple, List
from solutions import SOLUTIONS
from random import shuffle
from os.path import join


def save_graph(AdjMatrix, edgeCount, problem="", dataset_name= "", save_path= "", logfile= None):
    G = networkx.DiGraph()
    for i in range(len(AdjMatrix)):
        G.add_node(i)
    
    correct_egdes = 0
    wrong_edges = 0
    tot = 0
    for row in SOLUTIONS[problem]:
        tot += row.count(1)

    for r in range(len(AdjMatrix)):
        for c in range(len(AdjMatrix)):
            if AdjMatrix[r][c]:
                if AdjMatrix[r][c] == SOLUTIONS[problem][r][c]:
                    G.add_edge(r, c, color="g")
                    correct_egdes += 1
                else:
                    G.add_edge(r, c, color="r")
                    wrong_edges += 1
    logfile.write("\n\nNumber of correct edges found => " + str(correct_egdes)+ "\n")
    logfile.write("Number of wrong edges found => " + str(wrong_edges)+ "\n")
    logfile.write("Total edges found => " + str(wrong_edges + correct_egdes)+ "\n")
    logfile.write("Total edges of the correct solutions => " + str(tot) + "\n")
    # logfile.write("\n\nScore=  number of correct Edges / total edges of the solutions => " + str(correct_egdes / tot))
    for edge in edgeCount & G.edges():
        G[edge[0]][edge[1]]["weight"] = edgeCount[edge]

    if problem == "WASTE":
        pos = networkx.spring_layout(G, k=1, iterations=20)
        
    elif problem == "LC":
        pos = networkx.spring_layout(G, k=0.5, iterations=20)
    plt.figure(figsize=(10, 10))
    networkx.draw_networkx(
        G, pos, edge_color=networkx.get_edge_attributes(G, "color").values()
    )
    labels = networkx.get_edge_attributes(G, "weight")
    networkx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.savefig(join(save_path, dataset_name + ".png"))
    
def split_equal_part(n, n_part):
    r = n%n_part
    p = int(n/n_part)
    ls = []
    for i in range(n_part):
        ls.append([p*i, p*(i+1)])
    ls[-1][1] += r
    return ls

def getAdjMat(n, xt):
  narcs = n*(n-1)
  return torch.cat([torch.cat([torch.zeros(n-1).view(-1,1),xt[:narcs].view(-1,n)],dim=-1).view(-1),torch.zeros(1)]).view(n,n).int().tolist()


def get_variables_combinations(n: int, sub_problem_cardinality: int = 3) -> Tuple[Tuple[int]]:
    if sub_problem_cardinality < 3:
        raise ValueError
    return list(combinations(range(n), sub_problem_cardinality))

def get_variables_combinations_with_replacement(n: int, sub_problem_cardinality: int = 3) -> Tuple[Tuple[int]]:
    if sub_problem_cardinality < 3:
        raise ValueError
    return list(combinations_with_replacement(range(n), sub_problem_cardinality))

def get_variables_permutations(n: int, sub_problem_cardinality: int = 3) -> Tuple[Tuple[int]]:
    if sub_problem_cardinality < 3:
        raise ValueError
    return list(permutations(range(n), sub_problem_cardinality))

def get_variable_disposition_with_repetition(n: int, sub_problem_cardinality: int = 3) -> Tuple[Tuple[int]]:
    if sub_problem_cardinality < 3:
        raise ValueError
    return list(product(range(n), repeat= sub_problem_cardinality))

def is_self_loop(edge):
    return edge[0] == edge[1]

def recompose(AdjMatrixSubproblems, N) -> Tuple[List[List[int]], Dict]:
    start_time = time()
    SolAdjMatrix = [[0 for _ in range(N)] for _ in range(N)]
    edgeCount = {}
    for key in AdjMatrixSubproblems:
        subAdj = AdjMatrixSubproblems[key]
        subAdjDim = len(subAdj)
        for row in range(subAdjDim):
            for col in range(subAdjDim):
                if subAdj[row][col]:
                    try:
                        edgeCount[(key[row], key[col])] += 1
                    except KeyError:
                        edgeCount[(key[row], key[col])] = 1
    for edge in edgeCount:
        try:
            rev_edge = edge[::-1]
            if edgeCount[edge] > edgeCount[rev_edge] and edgeCount[edge] >= 1:
                SolAdjMatrix[edge[0]][edge[1]] = 1
        except KeyError:
            if edgeCount[edge] >= 1:
                SolAdjMatrix[edge[0]][edge[1]] = 1
    time_delta = time() - start_time
    return SolAdjMatrix, edgeCount, time_delta

SPLIT_METHODS = {
    "combinations" : get_variables_combinations,
    "combinations_with_replacements": get_variables_combinations_with_replacement,
    "permutations" : get_variables_permutations,
    "disposition_with_repetitions": get_variable_disposition_with_repetition
}
