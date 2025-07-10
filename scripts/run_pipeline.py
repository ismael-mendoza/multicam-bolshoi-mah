#!/usr/bin/env python3
"""Pipeline to produce dark matter halo catalogs to analyze from raw ROCKSTAR catalogs."""

import json
import os
from pathlib import Path
from shutil import copyfile

import click
import numpy as np
from astropy import table
from astropy.io import ascii as astro_ascii
from tqdm import tqdm

from multicam_bolshoi.catalogs import get_id_filter, save_cat_csv
from multicam_bolshoi.minh import load_cat_minh
from multicam_bolshoi.parameters import default_params
from multicam_bolshoi.progenitors.progenitor_lines import get_next_progenitor
from multicam_bolshoi.sims import all_sims

the_root = Path(__file__).absolute().parent.parent
raw_catalogs = the_root.joinpath("catalogs")
bolshoi_minh = raw_catalogs.joinpath("Bolshoi/minh/hlist_1.00035.minh")
catname_map = {
    "Bolshoi": "bolshoi",
    "BolshoiP": "bolshoi_p",
}

NAN_INTEGER = -5555


@click.group()
@click.option("--root", default=the_root.as_posix(), type=str, show_default=True)
@click.option("--outdir", type=str, required=True)
@click.option("--minh-file", type=str, default=bolshoi_minh, show_default=True)
@click.option("--catalog-name", default="Bolshoi", type=str, show_default=True)
@click.option(
    "--all-minh-files",
    default="bolshoi_catalogs_minh",
    type=str,
    show_default=True,
    help="./data",
)
@click.pass_context
def pipeline(ctx, root, outdir, minh_file, catalog_name, all_minh_files):
    """Full pipeline for extracting halo catalog."""
    catname = catname_map[catalog_name]

    ctx.ensure_object(dict)
    params_dir = Path(root).joinpath("data/params")
    output = Path(root).joinpath(f"data/processed/{outdir}")
    ids_file = output.joinpath("ids.json")
    exist_ok = bool(ids_file.exists())
    output.mkdir(exist_ok=exist_ok, parents=False)

    progenitor_file = output.joinpath(f"{catname}_progenitors.txt")
    lookup_file = params_dir.joinpath(f"lookup_{catname}.json")
    z_map_file_global = params_dir.joinpath(f"{catname}_z_map.json")
    z_map_file = output.joinpath("z_map.json")

    # write z_map file to output if not already there.
    assert z_map_file_global.exists(), "Global z_map was deleted?!"
    if not z_map_file.exists():
        copyfile(z_map_file_global, z_map_file)

    ctx.obj.update(
        dict(
            root=Path(root),
            data=raw_catalogs,
            output=output,
            catalog_name=catalog_name,
            minh_file=minh_file,
            ids_file=ids_file,
            dm_file=output.joinpath("dm_cat.csv"),
            progenitor_file=progenitor_file,
            lookup_file=lookup_file,
            progenitor_table_file=output.joinpath("progenitor_table.csv"),
            all_minh=raw_catalogs.joinpath(all_minh_files),
            lookup_index=output.joinpath("lookup.csv"),
            z_map=z_map_file,
        )
    )


@pipeline.command()
@click.option(
    "--m-low",
    default=11.15,
    help="lower log-mass of halo considered.",
    show_default=True,
)
@click.option(
    "--m-high",
    default=11.22,
    help="high log-mass of halo considered.",
    show_default=True,
)
@click.option(
    "--n-haloes",
    default=int(1e4),
    type=int,
    help="Desired num haloes in ID file.",
    show_default=True,
)
@click.pass_context
def make_ids(ctx, m_low, m_high, n_haloes):
    """Select ids of haloes to be used in the pipeline based on mass range."""
    # create appropriate filters
    assert not ctx.obj["ids_file"].exists()
    m_low = 10**m_low
    m_high = 10**m_high
    particle_mass = all_sims[ctx.obj["catalog_name"]].particle_mass
    assert m_low > particle_mass * 1e3, f"particle mass: {particle_mass:.3g}"
    filters = {
        "mvir": lambda x: (x > m_low) & (x < m_high),
        "pid": lambda x: x == -1,
    }

    # we only need the params that appear in the filter for now. (including 'id' and 'mvir')
    params = ["id", "mvir", "pid"]

    cat = load_cat_minh(ctx.obj["minh_file"], params, filters, verbose=False)

    # do we have enough haloes? keep only N of them.
    assert len(cat) >= n_haloes, f"There are only {len(cat)} haloes satisfying filter."
    keep = np.random.choice(np.arange(len(cat)), size=n_haloes, replace=False)
    cat = cat[keep]

    # double check only host haloes are allowed.
    assert np.all(cat["pid"] == -1)

    # extract ids into a json file, first convert to int's.
    ids = sorted([int(x) for x in cat["id"]])
    assert len(ids) == n_haloes
    with open(ctx.obj["ids_file"], "w", encoding="utf-8") as fp:
        json.dump(ids, fp)


@pipeline.command()
@click.pass_context
def make_dmcat(ctx):
    """Create dark matter catalog with default halo parameters given the ids file."""
    with open(ctx.obj["ids_file"], "r", encoding="utf-8") as fp:
        ids = np.array(json.load(fp))

    assert np.all(np.sort(ids) == ids)

    id_filter = get_id_filter(ids)
    cat = load_cat_minh(ctx.obj["minh_file"], default_params, id_filter)

    assert np.all(cat["id"] == ids)
    assert np.all(cat["pid"] == -1)
    assert len(cat) == len(ids)

    save_cat_csv(cat, ctx.obj["dm_file"])


@pipeline.command()
@click.pass_context
def make_progenitors(ctx):
    """Create progenitor and lookup table for all haloes in ids file."""
    progenitor_file = ctx.obj["progenitor_file"]
    lookup_file = ctx.obj["lookup_file"]
    assert progenitor_file.exists()
    assert lookup_file.exists()
    with open(ctx.obj["ids_file"], "r", encoding="utf-8") as fp:
        root_ids = np.array(json.load(fp)).astype(int)

    with open(lookup_file, "r", encoding="utf-8") as jp:
        lookup = json.load(jp)
        lookup = {int(k): int(v) for k, v in lookup.items()}

    z_map_file = ctx.obj["z_map"]
    assert z_map_file.exists()

    # first collect all scales from existing z_map
    with open(z_map_file, "r", encoding="utf-8") as fp:
        z_map = dict(json.load(fp))
        z_map = {int(k): float(v) for k, v in z_map.items()}

    # first obtain all scales available + save lines that we want to use.
    prog_lines = []

    # iterate through the progenitor generator, obtaining the haloes that match IDs
    with open(progenitor_file, "r", encoding="utf-8") as pf:
        for id_ in tqdm(root_ids, desc="Extracting lines and building lookup table"):
            if id_ in lookup:  # only extract lines in lookup.
                pos = lookup[id_]
                pf.seek(pos, os.SEEK_SET)
                prog_line = get_next_progenitor(pf)
                prog_lines.append(prog_line)

    # ordered from early -> late
    scales = sorted(list(z_map.values()))

    mvir_names = [f"mvir_a{i}" for i in range(len(scales))]
    # ratio (m2 / m1) where m2 is second most massive co-progenitor.
    cpgr_names = [f"coprog_mvir_a{i}" for i in range(len(scales))]
    names = ("id", *mvir_names, *cpgr_names)
    values = np.zeros((len(root_ids), len(names)))
    values[:, 0] = root_ids
    values[values == 0] = np.nan

    # create an astropy table for a mainline progenitor 'lookup'
    # i.e. for a given `idx` of root_ids, where root_ids[idx] = root_id, we have
    # lookup_index[idx, s] = id of progenitor line halo at scales[s]
    lookup_names = ["id"] + [f"id_a{i}" for i in range(len(scales))]
    lookup_index = np.zeros((len(root_ids), 1 + len(scales)))
    lookup_index[:, 0] = root_ids
    # np.nan forces us to use floats when saving.
    lookup_index[lookup_index == 0] = NAN_INTEGER

    for prog_line in tqdm(prog_lines, desc="Extracting information from lines"):
        # where should I insert this line?
        idx = np.where(root_ids == prog_line.root_id)[0].item()
        for s, scale in enumerate(scales):
            if scale in prog_line.cat["scale"]:
                line_idx = np.where(prog_line.cat["scale"] == scale)[0].item()
                mvir = prog_line.cat["mvir"][line_idx]
                values[idx, 1 + s] = mvir
                cpg_mvir = prog_line.cat["coprog_mvir"][line_idx]
                cpg_mvir = 0 if cpg_mvir < 0 else cpg_mvir  # missing values -1 -> 0
                values[idx, 1 + len(scales) + s] = cpg_mvir
                lookup_index[idx, 1 + s] = prog_line.cat["halo_id"][line_idx]

    prog_table = table.Table(names=names, data=values)
    prog_table.sort("id")
    prog_table["id"] = prog_table["id"].astype(int)
    lookup_index = table.Table(names=lookup_names, data=lookup_index.astype(int))
    lookup_index.sort("id")

    # save final table and json file mapping index to scale
    astro_ascii.write(prog_table, ctx.obj["progenitor_table_file"], format="csv")
    astro_ascii.write(lookup_index, ctx.obj["lookup_index"], format="csv")


@pipeline.command()
@click.pass_context
def combine_all(ctx):
    """Combine all previous steps of the pipeline and save the final catalog."""
    # load the 3 catalogs that we will be combining
    dm_cat = astro_ascii.read(ctx.obj["dm_file"], format="csv", fast_reader=True)
    progenitor_cat = astro_ascii.read(
        ctx.obj["progenitor_table_file"], format="csv", fast_reader=True
    )

    # check all are sorted.
    assert np.array_equal(np.sort(dm_cat["id"]), dm_cat["id"])
    assert np.array_equal(np.sort(progenitor_cat["id"]), progenitor_cat["id"])

    # make sure all 3 catalog have exactly the same IDs.
    assert np.array_equal(dm_cat["id"], progenitor_cat["id"])

    fcat = table.join(dm_cat, progenitor_cat, keys=["id"], join_type="inner")
    fcat_file = ctx.obj["output"].joinpath("final_table.csv")

    # save final csv containing all the information.
    astro_ascii.write(fcat, fcat_file, format="csv")


if __name__ == "__main__":
    pipeline()  # pylint: disable=no-value-for-parameter
