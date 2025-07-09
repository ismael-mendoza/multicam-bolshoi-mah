#!/usr/bin/env python3
"""
Useful functions for submitting sbatch jobs.
"""
import subprocess
from pathlib import Path

import click
import numpy as np


@click.command()
@click.option("--cmd", required=True, type=str)
@click.option("--jobname", required=True, type=str)
@click.option("--jobdir", default="output/jobs", type=str, show_default=True)
@click.option("--time", default="01:00", type=str, show_default=True)
@click.option("--nodes", default=1, type=int, show_default=True)
@click.option("--ntasks", default=1, type=int, show_default=True)
@click.option("--cpus-per-task", default=1, type=int, show_default=True)
@click.option("--mem-per-cpu", default="1GB", type=str, show_default=True)
def run_sbatch_job(
    cmd,
    jobname,
    jobdir,
    time,
    nodes,
    ntasks,
    cpus_per_task,
    mem_per_cpu,
):
    # prepare files and directories
    jobseed = np.random.randint(1e7)
    jobfile_name = f"{jobname}_{jobseed}.sbatch"
    job_dir = Path(jobdir)
    jobfile = job_dir.joinpath(jobfile_name)
    if not job_dir.exists():
        job_dir.mkdir(exist_ok=True)

    with open(jobfile, "w") as f:
        f.writelines(
            "#!/bin/bash\n\n"
            f"#SBATCH --job-name={jobname}\n"
            f"#SBATCH --output={jobdir}/%j.out\n"
            f"#SBATCH --time={time}:00\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks={ntasks}\n"
            f"#SBATCH --cpus-per-task={cpus_per_task}\n"
            f"#SBATCH --mem-per-cpu={mem_per_cpu}\n"
            f"#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user=imendoza@umich.edu\n"
            f"#SBATCH --account=cavestru1\n"
            f"#SBATCH --partition=standard\n"
            f"{cmd}\n"
        )

    subprocess.run(f"sbatch {jobfile.as_posix()}", shell=True)


if __name__ == "__main__":
    run_sbatch_job()
