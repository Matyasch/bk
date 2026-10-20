from functools import partial

from algorithms.ldecc import ldecc_plus, ldecc_pp
from algorithms.load import load, load_pp
from algorithms.mb_by_mb import mb_by_mb_plus, mb_by_mb_pp
from algorithms.pc import pc, pc_pp, cl_pc, pcalg_pc
from algorithms.snap import snap, snap_pp

ALGORITHMS = {
    "ldecc_plus": ldecc_plus,
    "ldecc_pp": ldecc_pp,
    "load": load,
    "load_pp": load_pp,
    "mb_by_mb_plus": mb_by_mb_plus,
    "mb_by_mb_pp": mb_by_mb_pp,
    "pc": pc,
    "pc_pp": pc_pp,
    "cl_pc": cl_pc,
    "cl_pc_stable": partial(cl_pc, stable=True),
    "pcalg_pc": pcalg_pc,
    "pcalg_pc_stable": partial(pcalg_pc, stable=True),
    "snap": snap,
    "snap_pp": snap_pp,
}
