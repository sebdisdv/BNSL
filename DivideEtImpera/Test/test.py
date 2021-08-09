#!/home/seb/Desktop/Tesi/D-Wave/ocean/bin/python3
import cProfile, pstats, io
from pstats import SortKey


import json
import time
import datetime
import os
import torch


from QUBOValues import getValuesForSubproblems
from QUBOMatrix import calcQUBOMatrixForDivideEtImpera
from dwaveSolver import dwaveSolve
from utils import SPLIT_METHODS, getAdjMat, recompose, save_graph
from pprint import pformat

with open("BNSL/DivideEtImpera/Test/test_settings.json", "r") as js:
    SETTINGS = json.load(js)


def solve_QA():
    pass


def solve_SA(logs_path):
    print("\\" * 100)
    pr = cProfile.Profile()
    pr.enable()
    for dataset in SETTINGS["datasets"]:
        log_file = open(f"{logs_path}/{dataset}.txt", "w")
        for path in SETTINGS["datasets"][dataset]:
            log_file.write("\\" * 100 + '\n')
            log_file.write(f'alpha= {SETTINGS["alpha"]}\nsubProblemCardinality= {SETTINGS["subProblemCardinality"]}\n')
            log_file.write(f"Solving {os.path.basename(path)}\n\n")
            print(f"Solving {os.path.basename(path)}")
            start_time_sub_creation = time.time()
            subprobValues, N = getValuesForSubproblems(
                path,
                SETTINGS["alpha"],
                SETTINGS["subProblemCardinality"],
                SPLIT_METHODS["combinations"],
            )
            end_time_sub_creation = time.time()
            print("Subproblem Creation time %0.5f" % (end_time_sub_creation - start_time_sub_creation))
            log_file.write("Subproblem Creation time %0.5f\n" % (end_time_sub_creation - start_time_sub_creation))
            AdjMatrixSubproblems = {}
            
            log_file.write(f"Solving {len(subprobValues)} subproblems\n")
            start_time_sub = time.time()
            for sub in subprobValues:
             
                Q, indexQUBO, posOfIndex, n = calcQUBOMatrixForDivideEtImpera(
                    subprobValues[sub]
                )
                Q = torch.tensor(Q)
                dsName = path[path.find("/") + 1 : path.find(".")]
                label = "{} - {} reads".format(dsName, SETTINGS["nReads"])
                minXt, minY, readFound, occurrences, annealTimeRes = dwaveSolve(
                    Q,
                    indexQUBO,
                    posOfIndex,
                    label,
                    method="SA",
                    nReads=SETTINGS["nReads"],
                    annealTime=SETTINGS["annealTime"],
                )
                AdjMatrixSubproblems[sub] = getAdjMat(n, minXt)
            end_time_sub = time.time()
            print("Subproblem solving time %0.5f" % (end_time_sub - start_time_sub))
            log_file.write("Subproblem solving time %0.5f\n" % (end_time_sub - start_time_sub))
            log_file.write("\nAdjMatrixSubproblems: \n")
            log_file.write(pformat(AdjMatrixSubproblems))
            log_file.write("\n\n")
            AdjMatrixSolution, edgeCount, final_sol_time = recompose(AdjMatrixSubproblems, N)
            log_file.write(pformat(edgeCount) + "\n\n")
            log_file.write("\nFinal Solution\n")
            log_file.write(pformat(AdjMatrixSolution))
            print("Final solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
            save_graph(AdjMatrixSolution, edgeCount, dataset, os.path.basename(path)[:len(os.path.basename(path)) - 4], logs_path, log_file)
            log_file.write("\nFinal solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
        log_file.close()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def main():
    log_path = os.path.join("BNSL/DivideEtImpera/Test/Logs", str(datetime.datetime.now())[:19])
    os.mkdir(log_path)
    
    solve_SA(log_path)
    # solve_QA()


if __name__ == "__main__":
    main()
