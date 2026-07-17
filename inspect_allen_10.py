#!/usr/bin/env python3
"""Inspect the Allen CCFv3 10-micron inputs without loading full volumes.

The Allen NRRDs are gzip-compressed and expand to several gigabytes each.  This
script reads one left-right plane at a time, extracts three representative
orthogonal sections, and writes a figure plus a small statistics report.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


DTYPES = {
    "unsigned short": "u2",
    "ushort": "u2",
    "uint16": "u2",
    "unsigned int": "u4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "double": "f8",
}


class NrrdPlaneReader:
    """Stream a gzip-encoded 3D NRRD as consecutive axis-2 planes."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._payload = None
        self.header: dict[str, str] = {}
        self.sizes: tuple[int, int, int]
        self.dtype: np.dtype

    def __enter__(self) -> "NrrdPlaneReader":
        self._file = self.path.open("rb")
        magic = self._file.readline().decode("ascii").strip()
        if not magic.startswith("NRRD"):
            raise ValueError(f"{self.path} is not an NRRD file")

        while True:
            raw = self._file.readline()
            if raw in (b"", b"\n", b"\r\n"):
                break
            line = raw.decode("ascii").rstrip("\r\n")
            if line and not line.startswith("#") and ": " in line:
                key, value = line.split(": ", 1)
                self.header[key] = value

        if self.header.get("dimension") != "3":
            raise ValueError(f"{self.path} is not a 3D NRRD")
        if self.header.get("encoding") not in {"gzip", "gz"}:
            raise ValueError(f"{self.path} is not gzip encoded")

        self.sizes = tuple(int(value) for value in self.header["sizes"].split())
        type_name = self.header["type"].lower()
        if type_name not in DTYPES:
            raise ValueError(f"Unsupported NRRD type {type_name!r}")
        endian = "<" if self.header.get("endian", "little") == "little" else ">"
        self.dtype = np.dtype(endian + DTYPES[type_name])
        self._payload = gzip.GzipFile(fileobj=self._file, mode="rb")
        return self

    def read_plane(self) -> np.ndarray:
        if self._payload is None:
            raise RuntimeError("Reader is not open")
        plane_values = self.sizes[0] * self.sizes[1]
        expected_bytes = plane_values * self.dtype.itemsize
        raw = self._payload.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError(
                f"Truncated payload in {self.path}: expected {expected_bytes} bytes, "
                f"read {len(raw)}"
            )
        return np.frombuffer(raw, dtype=self.dtype).reshape(
            self.sizes[:2], order="F"
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._payload is not None:
            self._payload.close()
        if self._file is not None:
            self._file.close()


def count_mask_extents(mask_path: Path) -> tuple[tuple[int, int, int], dict]:
    """Find high-area representative slices, preferring the right hemisphere."""
    with NrrdPlaneReader(mask_path) as reader:
        nx, ny, nz = reader.sizes
        counts_x = np.zeros(nx, dtype=np.int64)
        counts_y = np.zeros(ny, dtype=np.int64)
        counts_z = np.zeros(nz, dtype=np.int64)
        nonzero_values: set[int] = set()

        for z_index in range(nz):
            plane = reader.read_plane()
            active = plane != 0
            counts_x += np.count_nonzero(active, axis=1)
            counts_y += np.count_nonzero(active, axis=0)
            counts_z[z_index] = np.count_nonzero(active)
            nonzero_values.update(int(v) for v in np.unique(plane) if v != 0)

    right_start = nz // 2
    indices = (
        int(np.argmax(counts_x)),
        int(np.argmax(counts_y)),
        int(right_start + np.argmax(counts_z[right_start:])),
    )

    axis_counts = (counts_x, counts_y, counts_z)
    extents = []
    for counts in axis_counts:
        present = np.flatnonzero(counts)
        extents.append([int(present[0]), int(present[-1])])

    stats = {
        "mask_nonzero_values": sorted(nonzero_values),
        "mask_voxels": int(counts_z.sum()),
        "mask_volume_mm3": float(counts_z.sum() * 10**3 / 10**9),
        "mask_extent_indices": {
            "anterior_posterior": extents[0],
            "dorsal_ventral": extents[1],
            "left_right": extents[2],
        },
    }
    return indices, stats


def allocate_sections(dtype: np.dtype, sizes: tuple[int, int, int]) -> dict:
    nx, ny, nz = sizes
    return {
        "coronal": np.empty((ny, nz), dtype=dtype),
        "horizontal": np.empty((nx, nz), dtype=dtype),
        "sagittal": np.empty((ny, nx), dtype=dtype),
    }


def extract_template_sections(
    template_path: Path, indices: tuple[int, int, int]
) -> dict:
    x_index, y_index, z_index = indices
    with NrrdPlaneReader(template_path) as reader:
        sections = allocate_sections(reader.dtype, reader.sizes)
        for z in range(reader.sizes[2]):
            plane = reader.read_plane()
            sections["coronal"][:, z] = plane[x_index, :]
            sections["horizontal"][:, z] = plane[:, y_index]
            if z == z_index:
                sections["sagittal"][:] = plane.T
    return sections


def extract_field_sections(
    root: Path, indices: tuple[int, int, int]
) -> tuple[dict[str, dict], dict]:
    """Read mask, boundary, and Laplacian in lockstep."""
    paths = {
        "mask": root / "isocortex_mask_10.nrrd",
        "boundary": root / "isocortex_boundary_10.nrrd",
        "laplacian": root / "laplacian_10.nrrd",
    }
    readers = {name: NrrdPlaneReader(path) for name, path in paths.items()}
    opened = []
    try:
        for reader in readers.values():
            opened.append(reader.__enter__())
        sizes = readers["mask"].sizes
        if any(reader.sizes != sizes for reader in readers.values()):
            raise ValueError("Mask, boundary, and Laplacian grids do not match")

        sections = {
            name: allocate_sections(reader.dtype, sizes)
            for name, reader in readers.items()
        }
        sample_values = []
        boundary_values: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        boundary_counts: defaultdict[int, int] = defaultdict(int)
        x_index, y_index, z_index = indices

        for z in range(sizes[2]):
            planes = {name: reader.read_plane() for name, reader in readers.items()}
            for name, plane in planes.items():
                sections[name]["coronal"][:, z] = plane[x_index, :]
                sections[name]["horizontal"][:, z] = plane[:, y_index]
                if z == z_index:
                    sections[name]["sagittal"][:] = plane.T

            boundary = planes["boundary"]
            laplacian = planes["laplacian"]
            for label in (1, 3, 4):
                selected = laplacian[boundary == label]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    boundary_values[label].append(selected.copy())
                boundary_counts[label] += int(np.count_nonzero(boundary == label))

            if z % 10 == 0:
                sampled_lap = laplacian[::10, ::10]
                sampled_mask = planes["mask"][::10, ::10] != 0
                valid = sampled_mask & np.isfinite(sampled_lap)
                sample_values.append(sampled_lap[valid].copy())

        sampled = np.concatenate(sample_values)
        percentiles = (0, 1, 25, 50, 75, 99, 100)
        field_stats = {
            "laplacian_sample_count": int(sampled.size),
            "laplacian_sample_percentiles": {
                str(p): float(value)
                for p, value in zip(percentiles, np.percentile(sampled, percentiles))
            },
            "boundary_labels": {},
        }
        names = {1: "pial/top", 3: "white-matter/bottom", 4: "sides"}
        for label in (1, 3, 4):
            values = np.concatenate(boundary_values[label])
            field_stats["boundary_labels"][str(label)] = {
                "name": names[label],
                "count": int(boundary_counts[label]),
                "laplacian_min": float(values.min()),
                "laplacian_median": float(np.median(values)),
                "laplacian_max": float(values.max()),
            }
        return sections, field_stats
    finally:
        for reader in reversed(opened):
            reader.__exit__(None, None, None)


def plot_overview(
    output_path: Path,
    indices: tuple[int, int, int],
    template: dict,
    fields: dict[str, dict],
) -> None:
    planes = (
        ("coronal", f"Coronal — AP index {indices[0]} ({indices[0] / 100:.2f} mm)", "Left–right", "Dorsal–ventral"),
        ("horizontal", f"Horizontal — DV index {indices[1]} ({indices[1] / 100:.2f} mm)", "Left–right", "Anterior–posterior"),
        ("sagittal", f"Sagittal — LR index {indices[2]} ({indices[2] / 100:.2f} mm)", "Anterior–posterior", "Dorsal–ventral"),
    )
    fig, axes = plt.subplots(3, 4, figsize=(19, 14), constrained_layout=True)
    lap_image = None
    depth_image = None
    boundary_colors = {
        1: (0.95, 0.15, 0.10, 1.0),
        3: (0.10, 0.35, 0.95, 1.0),
        4: (1.00, 0.70, 0.05, 1.0),
    }

    for row, (plane, title, xlabel, ylabel) in enumerate(planes):
        anatomical = template[plane]
        mask = fields["mask"][plane] != 0
        boundary = fields["boundary"][plane]
        laplacian = fields["laplacian"][plane]
        finite_anatomy = anatomical[anatomical > 0]
        vmax = np.percentile(finite_anatomy, 99.5) if finite_anatomy.size else 1

        ax = axes[row, 0]
        ax.imshow(anatomical, cmap="gray", origin="upper", vmin=0, vmax=vmax)
        ax.contour(mask, levels=[0.5], colors=["#00ffff"], linewidths=0.7)
        ax.set_title(title + "\nTemplate with isocortex outline")

        ax = axes[row, 1]
        ax.imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1, alpha=0.35)
        rgba = np.zeros(boundary.shape + (4,), dtype=np.float32)
        for label, color in boundary_colors.items():
            rgba[boundary == label] = color
        ax.imshow(rgba, origin="upper")
        ax.set_title("Boundary conditions")

        masked_lap = np.ma.masked_where(~mask, laplacian)
        ax = axes[row, 2]
        lap_image = ax.imshow(
            masked_lap, cmap="viridis", origin="upper", vmin=0, vmax=1
        )
        ax.set_facecolor("black")
        ax.set_title("Allen Laplacian L\n0=pia, 1=white matter")

        ax = axes[row, 3]
        depth_image = ax.imshow(
            np.ma.masked_where(~mask, 1.0 - laplacian),
            cmap="magma",
            origin="upper",
            vmin=0,
            vmax=1,
        )
        ax.set_facecolor("black")
        ax.set_title("Pipeline relative depth d=1−L\n1=pia, 0=white matter")

        for ax in axes[row]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(lap_image, ax=axes[:, 2], shrink=0.75, label="Laplacian L")
    fig.colorbar(depth_image, ax=axes[:, 3], shrink=0.75, label="Relative depth d")
    legend = [
        Patch(facecolor=boundary_colors[1], label="1: pial/top"),
        Patch(facecolor=boundary_colors[3], label="3: white-matter/bottom"),
        Patch(facecolor=boundary_colors[4], label="4: sides"),
    ]
    axes[0, 1].legend(handles=legend, loc="upper right", fontsize=9)
    fig.suptitle(
        "Allen CCFv3 10 µm isocortex inputs — three orthogonal sections",
        fontsize=18,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/allen_ccf_10/input"),
        help="Directory containing the Allen 10-micron NRRDs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/allen_ccf_10/inspection"),
        help="Directory for the figure, statistics, and extracted sections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.input_dir / "isocortex_mask_10.nrrd"
    print("Finding representative isocortex sections...", flush=True)
    indices, mask_stats = count_mask_extents(mask_path)
    print(f"Using AP={indices[0]}, DV={indices[1]}, LR={indices[2]}", flush=True)

    print("Streaming mask, boundary, and Laplacian fields...", flush=True)
    fields, field_stats = extract_field_sections(args.input_dir, indices)
    print("Streaming anatomical template...", flush=True)
    template = extract_template_sections(
        args.input_dir / "average_template_10.nrrd", indices
    )

    stats = {
        "voxel_size_um": 10,
        "selected_indices": {
            "anterior_posterior": indices[0],
            "dorsal_ventral": indices[1],
            "left_right_right_hemisphere": indices[2],
        },
        **mask_stats,
        **field_stats,
    }
    stats_path = args.output_dir / "allen_inputs_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    npz_path = args.output_dir / "allen_inputs_sections.npz"
    np.savez_compressed(
        npz_path,
        indices=np.asarray(indices),
        **{
            f"{field}_{plane}": values
            for field, section_set in {"template": template, **fields}.items()
            for plane, values in section_set.items()
        },
    )

    figure_path = args.output_dir / "allen_inputs_overview.png"
    plot_overview(figure_path, indices, template, fields)
    print(f"Wrote {figure_path}", flush=True)
    print(f"Wrote {stats_path}", flush=True)
    print(f"Wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
