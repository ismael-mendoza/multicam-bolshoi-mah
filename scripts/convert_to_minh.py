#!/usr/bin/env python3
import multiprocessing as mp
import subprocess
from pathlib import Path

# NOTE: Need to check that *.config file has correct path to `name_index` and `type_index`
repo = Path("/home/imendoza/multicam/")
executable = "/home/imendoza/minnow/scripts/text_to_minh"
config = repo.joinpath("data/Bolshoi.config")
all_vars = repo.joinpath("data/all_vars.txt")
catalogs_dir = repo.joinpath("data/bolshoi_catalogs")
output_dir = repo.joinpath("data/bolshoi_catalogs_minh")


def work(cat_file):
    unzipped_file = cat_file.with_suffix("")

    # first convert using gzip (without deleting original)
    if not unzipped_file.exists():
        subprocess.run(f"gunzip -c {cat_file} >{unzipped_file}", shell=True)

    # then we run minnow to conver tit to minh.
    cmd = f"{executable} {config} {all_vars} {unzipped_file} {output_dir}"
    print(cmd)
    subprocess.run(cmd, shell=True)


cpus = 5


cat_files = []
for cat_file in catalogs_dir.iterdir():
    if cat_file.suffix == ".gz":
        output_file = output_dir.joinpath(cat_file.stem).with_suffix(".minh")
        if not output_file.exists():
            cat_files.append(cat_file)

with mp.Pool(cpus) as pool:
    pool.map(work, cat_files)
