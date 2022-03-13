import json
import time
import os
import os.path as path
import torch


from QUBOValues import getValuesForSubproblems
from QUBOMatrix import calcQUBOMatrixForDivideEtImpera
from dwaveSolver import dwaveSolve
from utils import SPLIT_METHODS, getAdjMat, recompose
# from utils import save_graph
from pprint import pformat
from solutions import SOLUTIONS

# Progress bar
from tqdm import tqdm

# Terminal Menu
from consolemenu import SelectionMenu


with open("test_settings.json", "r") as js:
    SETTINGS = json.load(js)

subproblem_ranges = {"LC" : range(3,5+1), "WASTE": range(3,9+1), "ALARM": range(3,15+1)}
methods = ["SA", "QA"]



def select_problem():
    problem_selection = SelectionMenu(
        SETTINGS["problems"],
        "Select which problem to solve",
        show_exit_option=False,
        clear_screen=False,
    )

    problem_selection.show()

    problem_selection.join()

    return problem_selection.selected_option

   
def select_k(choices):
    k_selection = SelectionMenu(
        choices,
        "Select which subproblem dimension",
        show_exit_option=False,
        clear_screen=False,
    )

    k_selection.show()

    k_selection.join()

    return k_selection.selected_option

def get_datasets(pr) -> tuple:
    for _, _, imgs_path in os.walk(path.join("DatasetDEF", pr)):
        return imgs_path


def select_dataset(choices):
    dataset_selection = SelectionMenu(
        choices,
        "Select which dataset to use",
        show_exit_option=False,
        clear_screen=False,
    )

    dataset_selection.show()

    dataset_selection.join()

    return dataset_selection.selected_option

def select_method(choices):
    method_selection = SelectionMenu(
        choices,
        "Select which method to use",
        show_exit_option=False,
        clear_screen=False,
    )

    method_selection.show()

    method_selection.join()

    return method_selection.selected_option

def solve(problem, dataset_path, k, method, log_path, number_of_runs):
    print("\\" * 100)
    log_file = open(log_path, "w")
    log_file.write("\\" * 100 + '\n')
    log_file.write(f'alpha= {SETTINGS["alpha"]}\nsubProblemCardinality= {k}\n')
    log_file.write(f"Solving {os.path.basename(dataset_path)}\n\n")
    print(f"Solving {os.path.basename(dataset_path)}")
    start_time_sub_creation = time.time()
    subprobValues, N = getValuesForSubproblems(
            dataset_path,
            SETTINGS["alpha"],
            k,
            SPLIT_METHODS["combinations"],
    )
    end_time_sub_creation = time.time()
    print("Subproblem Creation time %0.5f" % (end_time_sub_creation - start_time_sub_creation))
    log_file.write("Subproblem Creation time %0.5f\n" % (end_time_sub_creation - start_time_sub_creation))
    log_file.close()
    for _ in range(number_of_runs):
        log_file = open(log_path, "a")
        AdjMatrixSubproblems = {}    
        log_file.write(f"Solving {len(subprobValues)} subproblems\n")
        start_time_sub = time.time()
        for sub in tqdm(subprobValues):
            # print(f'solving {sub}\n')
            Q, indexQUBO, posOfIndex, n = calcQUBOMatrixForDivideEtImpera(
                subprobValues[sub]
            )
            Q = torch.tensor(Q)
            # dsName = path[path.find("/") + 1 : path.find(".")]
            dsName = path.basename(log_path)
            label = "{} - {} reads".format(dsName, SETTINGS["nReads"])
            minXt, minY, readFound, occurrences, annealTimeRes = dwaveSolve(
                    SETTINGS["dwave_conf_path"],
                    Q,
                    indexQUBO,
                    posOfIndex,
                    label,
                    method,
                    nReads=SETTINGS["nReads"],
                    annealTime=SETTINGS["annealTime"],
                )
            # subprobValues[sub] = None
            del Q
            AdjMatrixSubproblems[sub] = getAdjMat(n, minXt)
        end_time_sub = time.time()
            # print("Subproblem solving time %0.5f" % (end_time_sub - start_time_sub))
        log_file.write("Subproblem solving time %0.5f\n" % (end_time_sub - start_time_sub))
        log_file.write("\nAdjMatrixSubproblems: \n")
        log_file.write(pformat(AdjMatrixSubproblems))
        log_file.write("\n\n")
        AdjMatrixSolution, edgeCount, final_sol_time, wrongEdgeCount = recompose(AdjMatrixSubproblems, N)
        log_file.write("Edge Count \n")
        log_file.write(pformat(edgeCount) + "\n\n")
        log_file.write("Wrong Edge Count \n")
        log_file.write(pformat(wrongEdgeCount) + "\n\n")
        log_file.write("\nSolution recomposed\n")
        log_file.write(pformat(AdjMatrixSolution))
        
        
        correct_egdes = 0
        wrong_edges = 0
        for r in range(len(AdjMatrixSolution)):
            for c in range(len(AdjMatrixSolution)):
                if AdjMatrixSolution[r][c]:
                    if AdjMatrixSolution[r][c] == SOLUTIONS[problem][r][c]:
                        correct_egdes += 1
                    else:
                        wrong_edges += 1
        print("\n\nNumber of correct edges found => " + str(correct_egdes))
        print("Number of wrong edges found => " + str(wrong_edges)+ "\n")
        log_file.write("\n\nNumber of correct edges found => " + str(correct_egdes)+ "\n")
        log_file.write("Number of wrong edges found => " + str(wrong_edges)+ "\n")
        
        
                # print("Final solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
                # save_graph(AdjMatrixSolution, edgeCount, dataset, os.path.basename(path)[:len(os.path.basename(path)) - 4], logs_path, log_file)
        log_file.write("\nFinal solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
        log_file.close()
            


def check_folders():
    if not path.isdir("Logs"):
        os.mkdir("Logs")
    if not path.isdir("DatasetDEF"):
         print("Dataset is missing")
         exit(1)
           
    
def main():
    check_folders()
    pr = SETTINGS["problems"][select_problem()]
    
    datasets = get_datasets(pr)
    datasetpath = path.join("DatasetDef",pr,datasets[select_dataset(datasets)])
    subproblem_range = list(subproblem_ranges[pr])
    k = subproblem_range[select_k(subproblem_range)]
    method = methods[select_method(methods)]
    log_name = input("Log name: ")
    log_path = path.join("Logs", log_name+ ".txt")
    number_of_runs = 0
    while number_of_runs <= 0:
        try:
            number_of_runs = int(input("Number of run to perform: "))
        except ValueError:
            continue
    print(number_of_runs) 
    solve(pr, datasetpath, k, method, log_path, number_of_runs)


if __name__ == "__main__":
    main()



""" 
def solve(logs_path):
    print("\\" * 100)
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
            for _ in range(1):
                AdjMatrixSubproblems = {}
            
                log_file.write(f"Solving {len(subprobValues)} subproblems\n")
                start_time_sub = time.time()
                for sub in tqdm(subprobValues):
                    # print(f'solving {sub}\n')
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
                        method=SETTINGS["method"],
                        nReads=SETTINGS["nReads"],
                        annealTime=SETTINGS["annealTime"],
                    )
                    subprobValues[sub] = None
                    del Q
                    AdjMatrixSubproblems[sub] = getAdjMat(n, minXt)
                end_time_sub = time.time()
                # print("Subproblem solving time %0.5f" % (end_time_sub - start_time_sub))
                log_file.write("Subproblem solving time %0.5f\n" % (end_time_sub - start_time_sub))
                log_file.write("\nAdjMatrixSubproblems: \n")
                log_file.write(pformat(AdjMatrixSubproblems))
                log_file.write("\n\n")
                AdjMatrixSolution, edgeCount, final_sol_time, wrongEdgeCount = recompose(AdjMatrixSubproblems, N)
                log_file.write("Edge Count \n")
                log_file.write(pformat(edgeCount) + "\n\n")
                log_file.write("Wrong Edge Count \n")
                log_file.write(pformat(wrongEdgeCount) + "\n\n")
                log_file.write("\nSolution recomposed\n")
                log_file.write(pformat(AdjMatrixSolution))
                # print("Final solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
                # save_graph(AdjMatrixSolution, edgeCount, dataset, os.path.basename(path)[:len(os.path.basename(path)) - 4], logs_path, log_file)
                log_file.write("\nFinal solution recomposition time %0.5f seconds\n%s\n" % (final_sol_time, '\\' * 100))
                log_file.close()
                log_file = open(f"{logs_path}/{dataset}.txt", "a")
        log_file.close()
"""