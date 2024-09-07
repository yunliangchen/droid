import subprocess
import multiprocessing
import os
import subprocess
import argparse
import time

def run_process(gpu_id):
    subprocess.run([
        'python', '/lustre/fsw/portfolios/nvr/users/lawchen/project/droid/droid/convert_droid.py',
        '--folder', "/lustre/fsw/portfolios/nvr/users/lawchen/droid_raw/droid_raw/droid_with_pcd_5k",
        '--cuda_device', str(gpu_id)
    ])

if __name__ == '__main__':
    processes = []
    for gpu_id in range(8):
        p = multiprocessing.Process(target=run_process, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
