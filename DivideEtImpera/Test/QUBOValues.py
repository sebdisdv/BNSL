import math
import random
import multiprocessing
from os import pardir
from time import time

from pprint import pformat, pprint
from numba.core.types.containers import List
from numpy import array
from utils import (
    get_variables_combinations,
    get_variables_combinations_with_replacement,
    get_variables_permutations,
    split_equal_part
)

from numba.typed import List 
from numba import njit, jit

# MANAGER = multiprocessing.Manager()

def getValues(filename, alpha="1/(ri*qi)"):
    examples, n, states = getExamples(filename)
    # calculation of parentSets
    parentSets = calcParentSets(n)
    # calculation of w
    w = calcW(n, parentSets, states, examples, alpha=alpha)
    # calculation of delta
    delta = calcDelta(n, parentSets, w)
    # calculation of deltaMax
    deltaMax = calcDeltaMax(n, delta)
    # calculation of deltaTrans
    deltaTrans = calcDeltaTrans(n, delta)
    # calculation of deltaConsist
    deltaConsist = calcDeltaConsist(n, deltaTrans)
    return n, parentSets, w, deltaMax, deltaTrans, deltaConsist



def fill_examples(shared_list, examples, subprob, n):
    for ex in examples:
        shared_list.append([ex[i] for i in subprob])



def getValuesForSubproblems(
    filename,
    alpha="1/(ri*qi)",
    sub_problem_cardinality=3,
    subproblems_creation_method=get_variables_permutations,
):
 
    EXAMPLES, N, STATES = getExamples(filename)


    # random.shuffle(EXAMPLES)

    # Examples_dict = {}
    # for i in range(N):
    #   Examples_dict[i] = []
    # for i in range(len(EXAMPLES)):
    #   for j in range(len(EXAMPLES[i])):
    #    Examples_dict[j].append(EXAMPLES[i][j])
    
    subProblemsData = {}
    subProblemsColIndexes = subproblems_creation_method(N, sub_problem_cardinality)


    # subProblemsColIndexes = subProblemsColIndexes[:1] # prendo solo il primo sottoproblema per test
    # breakpoint()

    # MultiPro
    EXAMPLES_LENGHT = len(EXAMPLES)
    # parts = split_equal_part(EXAMPLES_LENGHT, 2)
    # split Examples
    # examples_split = [EXAMPLES[i[0]:i[1]] for i in parts]
    # Il problema è la lista condivisa
    for subprob in subProblemsColIndexes:
        # print(f"Doing {subprob}\n")
        subProblemsData[subprob] = {}
        subProblemsData[subprob]["n"] = sub_problem_cardinality
        subProblemsData[subprob]["states"] = array([STATES[i] for i in subprob])
        
        #multiprocess test
        # l1 = []
        # l2 = [] 
        # proc1 = multiprocessing.Process(target=fill_examples, args=[l1, examples_split[0], subprob, 0])
        # proc2 = multiprocessing.Process(target= fill_examples, args=[l2, examples_split[1], subprob, 1])
        # proc1.start()
        # proc2.start()
        # proc1.join()
        # proc2.join()
        # subProblemsData[subprob]["examples"] = list(itertools.chain(l1,l2))
        # EXAMPLES[parts[i][0]:parts[i][1]] fare in modo da calcolarlo solo una volta
       
        
        #Normal way
        
        subProblemsData[subprob]["examples"] = array(filter_example(subprob, EXAMPLES, EXAMPLES_LENGHT))
        # subProblemsData[subprob]["examples"] = []
        # for example in EXAMPLES:
        #    subProblemsData[subprob]["examples"].append([example[i] for i in subprob])
        
        #numpy test(Failed)
        # subProblemsData[subprob]["examples"] = array([Examples_dict[i] for i in subprob]).T.tolist()
        # pprint(subProblemsData[subprob]["examples"])
        # breakpoint()
        
        subProblemsData[subprob]["parentSets"] = calcParentSets(
            subProblemsData[subprob]["n"]
        )
       
       
       
        subProblemsData[subprob]["w"] = calcW(
            subProblemsData[subprob]["n"],
            subProblemsData[subprob]["parentSets"],
            subProblemsData[subprob]["states"],
            subProblemsData[subprob]["examples"],
            alpha=alpha,
        )

        
        # del subProblemsData[subprob]["examples"]
        
        subProblemsData[subprob]["delta"] = calcDelta(
            subProblemsData[subprob]["n"],
            subProblemsData[subprob]["parentSets"],
            subProblemsData[subprob]["w"],
        )

        
        subProblemsData[subprob]["deltaMax"] = calcDeltaMax(
            subProblemsData[subprob]["n"], subProblemsData[subprob]["delta"]
        )
        

        
        subProblemsData[subprob]["deltaTrans"] = calcDeltaTrans(
            subProblemsData[subprob]["n"], subProblemsData[subprob]["delta"]
        )
        

        subProblemsData[subprob]["deltaConsist"] = calcDeltaConsist(
            subProblemsData[subprob]["n"], subProblemsData[subprob]["deltaTrans"]
        )
        
       
    return subProblemsData, N

@jit(nopython= True)
def filter_example(subprob, Examples, lenght):
    res = []
    for i in range(lenght):
        sub_res = []
        for j in range(len(subprob)):
            sub_res.append(Examples[i][subprob[j]])
        res.append(sub_res)
    return res


def getExamples(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        # get info from first line
        info = lines.pop(0).split(" ")
        n = int(info[0])
        # number of states for each variable
        states = [int(si) for si in info[1 : n + 1]]
        # get examples
        # examples = []
        examples = []
        for i in range(len(lines)):
            ls = lines[i].split(" ")
            l2 = []
            for j in range(len(ls)):
                l2.append(int(ls[j]))
            examples.append(l2)
            # examples += [[int(x) for x in ls]]
    return array(examples), n, array(states)


# subsets with max size m=2
def calcSubSets(s):
    sSet = [()]
    # all individuals
    sSet += [(i,) for i in s]
    # all unordered pairs
    sSet += [(i, j) for i in s for j in s if i < j]
    return sSet


def calcParentSets(n):
    parentSets = []
    for i in range(n):
        maxSet = tuple(x for x in range(0, n) if x != i)
        parentSets.append(calcSubSets(maxSet))
    return parentSets


def calcAlpha(n, states, parentSets, N, alpha="1/(ri*qi)"):
    if alpha == "1/(ri*qi)":

        def ca(N, ri, qi):
            return 1 / (ri * qi)

    elif alpha == "1":

        def ca(N, ri, qi):
            return 1

    elif alpha == "N/(ri*qi)":

        def ca(N, ri, qi):
            return N / (ri * qi)

    elif alpha == "1/ri":

        def ca(N, ri, qi):
            return 1 / ri

    alpha = []
    for i in range(n):
        alpha.append({})
        # for each valid parent set
        for parentSet in parentSets[i]:
            # generate alpha
            alpha[i][parentSet] = []
            qi = calcQi(parentSet, states)
            for j in range(qi):
                alpha[i][parentSet].append([])
                for k in range(states[i]):
                    alpha[i][parentSet][j].append([])
                    # initialize alpha according to the choice
                    alpha[i][parentSet][j][k] = ca(N, states[i], qi)
    return alpha


def calcAlphaijSum(alpha, parentSet, i, j, states):
    sum = 0
    for alphaijk in alpha[i][parentSet][j]:
        sum += alphaijk
    return sum


@jit(nopython= True)
def calcJthState(j, parentSet, states):
    # ASSUMPTION: all the variables have the same number of states,
    # if this is false, some combinations will be ignored
    p0 = parentSet[0]
    # j = (states[p0]^1)*sp0 + (states[p1]^0)*sp1
    # j = states[p0]*sp0 + sp1
    sp0 = j // states[p0]
    sp1 = j % states[p0]
    return sp0, sp1

@jit(nopython= True)
def calcNijk(examples, parentSet, i, j, k, states):
    count = 0
    for example in examples:
        # variable i is in the k-th state
        if example[i] == k:
            # parent set is in the j-th state
            if len(parentSet) == 0:
                # empty parent set -> only has one state j=0
                if j == 0:
                    count = count + 1
            elif len(parentSet) == 1:
                # one variable -> is that variable in its j-th state?
                if example[parentSet[0]] == j:
                    count = count + 1
            else:
                # parent set has 2 variables
                sp0, sp1 = calcJthState(j, parentSet, states)
                p0 = parentSet[0]
                p1 = parentSet[1]
                if example[p0] == sp0 and example[p1] == sp1:
                    count = count + 1

    return count


def calcNijSum(examples, parentSet, i, j, states):
    sum = 0
    for k in range(states[i]):
        sum += calcNijk(examples, parentSet, i, j, k, states)
    return sum


def calcQi(parentSet, states):
    qi = 1
    for j in parentSet:
        qi *= states[j]
    return qi


def calcSi(i, parentSet, states, alpha, examples):
    qi = calcQi(parentSet, states)
    sum = 0
    for j in range(qi):
        alphaij = calcAlphaijSum(alpha, parentSet, i, j, states)
        Nij = calcNijSum(examples, parentSet, i, j, states)
        sum += math.lgamma(alphaij) - math.lgamma(alphaij + Nij)
        for k in range(states[i]):
            Nijk = calcNijk(examples, parentSet, i, j, k, states)
            sum += math.lgamma(alpha[i][parentSet][j][k] + Nijk) - math.lgamma(
                alpha[i][parentSet][j][k]
            )
    return -sum


def calcWi(i, parentSet, s):
    if parentSet == ():
        return s[i][()]
    elif len(parentSet) == 1:
        return s[i][parentSet] - s[i][()]
    elif len(parentSet) == 2:
        p0 = parentSet[0]
        p1 = parentSet[1]
        return s[i][parentSet] - s[i][(p0,)] - s[i][(p1,)] + s[i][()]


def calcS(n, states, parentSets, alpha, examples):
    s = []
    for i in range(n):
        s.append({})
        # for each valid parent set
        for parentSet in parentSets[i]:
            # calculate si
            s[i][parentSet] = calcSi(i, parentSet, states, alpha, examples)
    return s


def calcWFromS(n, parentSets, s):
    w = []
    for i in range(n):
        w.append({})
        for parentSet in parentSets[i]:
            # calculate wi
            w[i][parentSet] = calcWi(i, parentSet, s)
    return w


def calcW(n, parentSets, states, examples, alpha="1/(ri*qi)"):
    # calculation of alpha
    alpha = calcAlpha(n, states, parentSets, len(examples), alpha=alpha)
    # calculation of s
    s = calcS(n, states, parentSets, alpha, examples)
    # print()
    # calculation of w
    w = calcWFromS(n, parentSets, s)
    return w


def calcDeltaji(j, i, w, parentSets, n):
    deltaPrimeji = -w[i][(j,)]
    # for all parentSets for i, {j,k}
    for parentSet in parentSets[i]:
        if len(parentSet) == 2:
            if j in parentSet:
                deltaPrimeji -= min(0, w[i][parentSet])
    return max(0, deltaPrimeji)


def getDelta(j, i, delta):
    posI = i if i < j else i - 1
    return delta[j][posI]


def calcDelta(n, parentSets, w):
    delta = []
    for j in range(n):
        delta.append([])
        for i in range(n):
            if i != j:
                delta[j].append(calcDeltaji(j, i, w, parentSets, n))
    return delta


def calcDeltaMax(n, delta):
    deltaMax = []
    for i in range(n):
        maxDelta = 0
        for j in range(n):
            if i != j:
                maxDelta = max(maxDelta, getDelta(j, i, delta))
        # +1 to satisfy the > constraint
        deltaMax.append(maxDelta + 1)
    return deltaMax


def calcDeltaTrans(n, delta):
    # calculate max of delta
    maxDelta = 0
    for j in range(n):
        for i in range(n):
            if i != j:
                maxDelta = max(maxDelta, getDelta(j, i, delta))
    # calculate deltaTrans -> they are all the same
    # +1 to satisfy the > constraint
    deltaTrans = maxDelta + 1
    return deltaTrans


def calcDeltaConsist(n, deltaTrans):
    # need to calculate max between deltaTrans(i,j,k), but there is only one deltaTrans
    # +1 to satisfy the > constraint
    deltaConsist = (n - 2) * deltaTrans + 1
    return deltaConsist


def main():
   pass


if __name__ == "__main__":
    main()
