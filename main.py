from plotter import computePareto
from test import tester
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import numpy as np
from collections import defaultdict


START_AT_CENTER = False

WIDTH = 10
HEIGHT = 10
N_AGENTS = 3
GOAL_REWARD = 100
EMPTY_REWARD = -1
WALL_REWARD = -10
 
# ----- Individual tests

t = tester(START_AT_CENTER, WIDTH, HEIGHT, N_AGENTS, GOAL_REWARD, EMPTY_REWARD, WALL_REWARD)
log = t.runCollaborativeTest(500, [0.75,1.0],True, True, True)
log = t.runNaiveTest(500, 2000 ,True, True, True)
log = t.runUnstructuredTest(500, 2000, [64,64,64], True, True, True)


# ------ Hyperparameter optimization tests

def run_all_tests(run_single_test, param_grid): # Runs tests in parallel
    total_jobs = len(param_grid)
    completed_jobs = 0

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_test, p) for p in param_grid]
        for future in as_completed(futures):
            future.exception
            try:
                results.append(future.result())
            except Exception as e:
                print(e)
            completed_jobs += 1
            print(f"{completed_jobs} / {total_jobs} jobs done.")

    return results

def run_u_test(args): # Run unstructured test with different NN shape
    a = args
    t = tester(START_AT_CENTER, WIDTH, HEIGHT, N_AGENTS, GOAL_REWARD, EMPTY_REWARD, WALL_REWARD)
    avg, err = t.runUnstructuredTest(
        500, 2000, a, False, False, False, f"{a}"
    )
    return (a, avg, err)

def unstructuredBestNNShape(): # Run unstructured test with different NN shape
    param_grid = []
    for ele in [32, 64, 128]:
        param_grid.append([ele]*2)
        param_grid.append([ele]*3)
    
    data = run_all_tests(run_u_test, param_grid * 25)

    grouped = defaultdict(list)
    for nn, avg, err in data:
        grouped[str(nn)].append((avg,err))

    processed = []

    for nn, val in grouped.items():
        mean = np.mean([value[0] for value in val if value[0] is not None])
        err = np.mean([value[1] for value in val])
        processed.append((nn, mean, err))
    computePareto(processed)


def run_c_test(args): # Run collaborative test with different K boundaries
        v = args
        t = tester(START_AT_CENTER, WIDTH, HEIGHT, N_AGENTS, GOAL_REWARD, EMPTY_REWARD, WALL_REWARD)
        avg, err = t.runCollaborativeTest(
            500, [max(v), min(v)], False, False, False, f"{v}"
        )
        return (max(v), min(v), avg, err)

def collaborativeBestRange(): # Run collaborative test with different K boundaries
    
    param_grid = list(product([0.0, 0.25, 0.5, 0.75, 1.0], repeat=2)) * 100
    
    data = run_all_tests(run_c_test, param_grid)

    grouped = defaultdict(list)
    for high, low, avg, err in data:
        grouped[(high, low)].append((avg,err))

    processed = []

    for (high, low), val in grouped.items():
        mean = np.mean([value[0] for value in val if value[0] is not None])
        err = np.mean([value[1] for value in val])
        h = int(high) if high == int(high) else high
        l = int(low) if low == int(low) else low
        if l==h:
            processed.append((f"{l}", mean, err))
        else:
            processed.append((f"{l}-{h}", mean, err))
    computePareto(processed)


#collaborativeBestRange()
#unstructuredBestNNShape()