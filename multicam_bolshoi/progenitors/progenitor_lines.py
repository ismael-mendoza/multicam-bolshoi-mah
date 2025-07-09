"""Module to read main progenitor lines into python objects from text file."""
import re

from astropy.table import Table


def get_next_progenitor(pf):
    # pf is a file pointer to file containing extracted main line progenitor information.
    # this function reads from the file pointer and returns the first progenitor line found.

    for line in pf:
        line = line.rstrip()  # remove trailing whitespace
        if line:  # not empty
            top_match = re.match(
                r"Order is: \(id, mvir, scale, scale_of_last_MM, coprog_id, coprog_mvir, "
                r"coprog_scale\)",
                line,
            )
            tree_root_match = re.match(r"# tree root id: (\d+) #", line)
            halo_match = re.match(
                r"(\d+),(\d+\.\d*),(\d+\.\d*),(\d+\.\d*),(\d*),(\d*.?\d*),(\d*.?\d*)",
                line,
            )
            total_halos_match = re.match(r"Number of root nodes is: (\d+)", line)
            total_root_halos_match = re.match(r"final count is: (\d+)", line)

            if tree_root_match:
                root_id = tree_root_match.groups()[0]
                prog_line = ProgenitorLine(root_id=int(root_id))

            elif halo_match:
                (
                    halo_id,
                    mvir,
                    scale,
                    scale_of_last_mm,
                    coprog_id,
                    coprog_mvir,
                    coprog_scale,
                ) = (float(x) if x != "" else -1 for x in halo_match.groups())
                prog_line.add(
                    (
                        halo_id,
                        mvir,
                        scale,
                        scale_of_last_mm,
                        coprog_id,
                        coprog_mvir,
                        coprog_scale,
                    )
                )

            elif total_halos_match or total_root_halos_match or top_match:
                continue

            else:
                print(line)
                raise ValueError("Something is wrong in this halo file.")

        elif prog_line is not None:
            prog_line.finalize()
            return prog_line

        else:
            continue


class ProgenitorLine(object):
    def __init__(self, root_id):
        """
        Class representing progenitors read from the file created by save_progenitors.py
        """
        self.root_id = root_id
        self.cat = Table()
        self.colnames = [
            "halo_id",
            "mvir",
            "scale",
            "scale_of_last_MM",
            "coprog_id",
            "coprog_mvir",
            "coprog_scale",
        ]
        self.rows = []
        self.finalized = False

    def add(self, halo_tuple):
        assert not self.finalized
        self.rows.append(halo_tuple)

    def finalize(self):
        assert not self.finalized
        self.finalized = True
        self.cat = Table(rows=self.rows, names=self.colnames)
