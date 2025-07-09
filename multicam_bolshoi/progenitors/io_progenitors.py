"""Download ROCKSTAR tree information and process it with Consistent Trees."""
import multiprocessing
import os
import re
import subprocess
from pathlib import Path, PosixPath

url_skeletons = {"Bolshoi": "https://www.slac.stanford.edu/~behroozi/Bolshoi_Trees/tree"}


def work(task):
    return subprocess.run(task, shell=True)


def download_trees(ncubes, data_dir, catalog_name):
    """Download all the bolshoi trees from the listed url."""
    assert isinstance(type(data_dir), PosixPath)

    url_skeleton = url_skeletons[catalog_name]
    if data_dir.exists():
        raise IOError("Directory already exists! Overwriting?")

    downloads_file = data_dir.joinpath("downloads.txt")

    # create file listing all files to be downloaded one-per-line.
    for x in range(0, ncubes):
        for y in range(0, ncubes):
            for z in range(0, ncubes):
                with open(downloads_file, "a", encoding="utf-8") as f:
                    if not os.path.isfile(f"data/trees/tree_{x}_{y}_{z}.dat.gz"):
                        f.write(f"{url_skeleton}_{x}_{y}_{z}.dat.gz\n")

    # then download the files using multiprocessing
    os.chdir(data_dir.as_posix())
    subprocess.run("cat downloads.txt | xargs -n 1 --max-procs 10 --verbose wget", shell=True)


def write_main_line_progenitors(read_trees_dir, trees_dir, prefix, mcut, cpus=5):
    """Use the consistent trees package to extract main progenitor lines from downloaded trees.

    Args:
        read_trees_dir (PosixPath):
        trees_dir (PosixPath): where trees are saved (.dat files).
        prefix (PosixPath): where to save each file and it's name.
        mcut (float):
        cpus (int):
    """

    subprocess.run(f"cd {read_trees_dir}; make", shell=True)
    cmds = []
    for p in trees_dir.iterdir():
        if p.suffix == ".dat" and p.name.startswith("tree"):
            print(f"Found tree: {p.name}")
            # get numbered part.
            suffx = re.search(r"tree(_\d_\d_\d)\.dat", p.name).groups()[0]
            final = Path(f"{prefix}{suffx}.txt")
            if not final.is_file():
                cmd = f"cd {read_trees_dir}; " f"./read_tree {p} {final} {mcut}"
                cmds.append(cmd)

    pool = multiprocessing.Pool(cpus)
    pool.map(work, cmds)


def merge_progenitors(progenitor_dir, progenitor_file):
    """Merge all progenitor files in 'progenitor_dir' into one, save it as 'progenitor_file'."""
    with open(progenitor_file, "w", encoding="utf-8") as pf:
        for p in progenitor_dir.iterdir():
            assert p.name.startswith("mline")
            print(p.name)
            with open(p, "r", encoding="utf-8") as single_pf:
                # ignore headers, etc. which is accounted for in progenitors.py
                pf.write(single_pf.read())
