import subprocess
from multiprocessing import Pool

def run_command(task_id):
    command = f"root -l -b -q sampler.cc+ < inputfiles/{task_id}.txt > outlog_{task_id}.log 2>&1 &"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    num_cores = 20  # Number of cores to use
    num_tasks = 59  # Total number of tasks
    with Pool(num_cores) as p:
        p.map(run_command, range(0, num_tasks))