"""Extract the one-voxel-thick boundary of a relative-depth isosurface."""

import sys

import numpy as np
from voxcell import VoxelData


def is_background(array, val=np.nan):
    return np.isnan(array) if np.isnan(val) else (array == val)


def extract_iso_voxels(rdepth, thresh, side, bkg_val=np.nan, chunk_size=16):
    """Return foreground voxels touching the opposite side of ``thresh``.

    The original implementation computed a float64 Euclidean distance
    transform over the isocortex bounding box.  Its ``edt == 1`` test is
    exactly equivalent to asking whether a voxel has a face-adjacent neighbor
    on the opposite side of the threshold.  Evaluating that relation in small
    slabs gives the same result without the multi-gigabyte EDT allocation.
    """
    if side not in ("top", "bottom"):
        raise ValueError('side must be "top" or "bottom"')
    if rdepth.ndim != 3:
        raise ValueError("relative depth must be a three-dimensional array")

    boundary = np.zeros(rdepth.shape, dtype=np.uint8)
    shape = rdepth.shape

    if np.isnan(bkg_val):
        valid = np.isfinite
    else:
        valid = lambda array: array != bkg_val

    if side == "top":
        current_side = lambda array: array >= thresh
        opposite_side = lambda array: array < thresh
    else:
        current_side = lambda array: array <= thresh
        opposite_side = lambda array: array > thresh

    # Check both directions along each of the three axes.  Axis 0 is also
    # divided into slabs so temporary boolean arrays remain small at 10 um.
    for axis in range(3):
        for shift in (-1, 1):
            current_start = [0, 0, 0]
            current_stop = list(shape)
            if shift < 0:
                current_start[axis] = 1
            else:
                current_stop[axis] -= 1

            for slab_start in range(current_start[0], current_stop[0], chunk_size):
                slab_stop = min(slab_start + chunk_size, current_stop[0])
                current_slice = [slice(current_start[i], current_stop[i]) for i in range(3)]
                current_slice[0] = slice(slab_start, slab_stop)

                neighbor_slice = list(current_slice)
                neighbor_slice[axis] = slice(
                    current_slice[axis].start + shift,
                    current_slice[axis].stop + shift,
                )
                current_slice = tuple(current_slice)
                neighbor_slice = tuple(neighbor_slice)

                current = rdepth[current_slice]
                neighbor = rdepth[neighbor_slice]
                selected = (
                    valid(current)
                    & valid(neighbor)
                    & current_side(current)
                    & opposite_side(neighbor)
                )
                boundary[current_slice][selected] = 1

    return boundary


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 5:
        raise SystemExit(
            "usage: extract_iso_voxels.py RELATIVE_DEPTH THRESHOLD "
            "{top,bottom} OUTPUT [BACKGROUND]"
        )

    rdepth_nrrd = argv[1]
    thresh = float(argv[2])
    side = argv[3]
    output_nrrd = argv[4]
    bkg_val = np.nan if len(argv) <= 5 else np.float32(argv[5])

    vd = VoxelData.load_nrrd(rdepth_nrrd)
    boundary = extract_iso_voxels(vd.raw, thresh, side, bkg_val)
    vd.with_data(boundary).save_nrrd(output_nrrd)


if __name__ == "__main__":
    main()
