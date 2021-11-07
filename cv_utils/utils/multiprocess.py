import multiprocessing as mp

def run_multiprocessing(func, params, num_workers, debug=True):
    num_run_workers = min(num_workers, len(params))
    
    if debug:
        print("\n###############################################")
        print(f"The number of CPUs: {mp.cpu_count()}")
        print(f"User NUM_WORKERS input: {num_workers}")
        print(f"Number of tasks: {len(params)}")
        print(f"Number of CPUs used: {num_run_workers}")
        print("###############################################\n")
    
    if len(params) == 0:
        return None
    
    pool = mp.Pool(num_run_workers)
    
    return pool.map(func, params)