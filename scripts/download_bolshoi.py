#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

z_map_file = Path("/home/imendoza/multicam/output/bolshoi_z_map.json")
data_dir = Path("/home/imendoza/multicam/data/bolshoi_catalogs")
with open(z_map_file, "r") as fp:
    z_map = json.load(fp)

hlist_template = "hlist_{}.list.gz"
skeleton_url = "https://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/{}"


data_dir.mkdir(exist_ok=True, parents=False)
downloads_file = data_dir.joinpath("downloads.txt")

if downloads_file.exists():
    raise IOError("Delete downloads.txt file to re-run this script.")

with open(downloads_file, "a") as f:
    for k, v in z_map.items():
        hlist_filename = hlist_template.format(v)
        hlist_file = data_dir.joinpath(hlist_filename)
        if not hlist_file.exists():
            url = skeleton_url.format(hlist_filename)
            f.write(f"{url}\n")

# then download the files using multiprocessing
os.chdir(data_dir.as_posix())
subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 20 --verbose wget", shell=True)
