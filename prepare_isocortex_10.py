#!/usr/bin/env python3
"""Prepare the Allen CCFv3 10-micron isocortex flatmap inputs.

This reproduces the geometric preprocessing recipes in
``flatmap/examples/genfiles_isocortex.mk`` while streaming axis-2 planes.  The
streaming implementation avoids expanding several multi-gigabyte NRRDs in RAM.
"""

from __future__ import annotations

import argparse
import gzip
import os
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import nrrd
import numpy as np
from nrrd.writer import _write_header
from scipy import ndimage
from scipy.spatial import cKDTree


DTYPES = {
    "unsigned char": "u1",
    "uchar": "u1",
    "uint8": "u1",
    "signed char": "i1",
    "int8": "i1",
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


class NrrdPlaneWriter:
    """Atomically write a gzip-encoded NRRD one axis-2 plane at a time."""

    def __init__(
        self,
        path: Path,
        reference_header: dict,
        dtype: np.dtype,
        compression_level: int = 6,
    ):
        self.path = path
        self.reference_header = reference_header
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.compression_level = compression_level
        self._raw_file = None
        self._payload = None
        self._temporary_path: Path | None = None

    def __enter__(self) -> "NrrdPlaneWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            delete=False,
        )
        self._raw_file = temporary
        self._temporary_path = Path(temporary.name)

        header = self.reference_header.copy()
        header["type"] = {
            np.dtype("uint8"): "uint8",
            np.dtype("int8"): "int8",
            np.dtype("float32"): "float",
        }[self.dtype.newbyteorder("=")]
        header["encoding"] = "gzip"
        header["endian"] = "little"
        _write_header(self._raw_file, header)
        self._payload = gzip.GzipFile(
            fileobj=self._raw_file,
            mode="wb",
            compresslevel=self.compression_level,
        )
        return self

    def write_plane(self, plane: np.ndarray) -> None:
        if self._payload is None:
            raise RuntimeError("Writer is not open")
        data = np.asarray(plane, dtype=self.dtype, order="F")
        self._payload.write(data.tobytes(order="F"))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self._payload is not None:
                self._payload.close()
            if self._raw_file is not None:
                self._raw_file.close()
            if exc_type is None and self._temporary_path is not None:
                self._temporary_path.chmod(0o644)
                os.replace(self._temporary_path, self.path)
                self._temporary_path = None
        finally:
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)


def check_matching_grids(*headers: dict) -> None:
    keys = ("sizes", "space directions", "space origin")
    reference = headers[0]
    for header in headers[1:]:
        if any(not np.array_equal(header[key], reference[key]) for key in keys):
            raise ValueError("Input NRRD grids do not match")


def generate_hemisphere_volumes(input_dir: Path, output_dir: Path) -> None:
    mask_path = input_dir / "isocortex_mask_10.nrrd"
    reference_header = nrrd.read_header(str(mask_path))
    outputs = {
        "right": output_dir / "mask_hemi.nrrd",
        "left": output_dir / "mask_hemi_other.nrrd",
        "labels": output_dir / "hemispheres.nrrd",
    }

    with ExitStack() as stack:
        mask_reader = stack.enter_context(NrrdPlaneReader(mask_path))
        writers = {
            name: stack.enter_context(
                NrrdPlaneWriter(path, reference_header, np.dtype("uint8"))
            )
            for name, path in outputs.items()
        }
        nx, ny, nz = mask_reader.sizes
        middle = nz // 2
        for z_index in range(nz):
            mask = mask_reader.read_plane() != 0
            right_side = z_index >= middle
            right = np.full((nx, ny), right_side, dtype=np.uint8)
            left = np.full((nx, ny), not right_side, dtype=np.uint8)
            hemispheres = np.zeros((nx, ny), dtype=np.uint8)
            hemispheres[mask] = 1 if right_side else 2
            writers["right"].write_plane(right)
            writers["left"].write_plane(left)
            writers["labels"].write_plane(hemispheres)
            if z_index % 200 == 0:
                print(f"  hemisphere plane {z_index}/{nz}", flush=True)


def generate_relative_depth(input_dir: Path, output_dir: Path) -> None:
    mask_path = input_dir / "isocortex_mask_10.nrrd"
    laplacian_path = input_dir / "laplacian_10.nrrd"
    mask_header = nrrd.read_header(str(mask_path))
    laplacian_header = nrrd.read_header(str(laplacian_path))
    check_matching_grids(mask_header, laplacian_header)

    output_path = output_dir / "relative_depth.nrrd"
    with ExitStack() as stack:
        mask_reader = stack.enter_context(NrrdPlaneReader(mask_path))
        laplacian_reader = stack.enter_context(NrrdPlaneReader(laplacian_path))
        writer = stack.enter_context(
            NrrdPlaneWriter(output_path, laplacian_header, np.dtype("float32"))
        )
        nx, ny, nz = mask_reader.sizes
        middle = nz // 2
        for z_index in range(nz):
            mask = mask_reader.read_plane() != 0
            laplacian = laplacian_reader.read_plane()
            relative_depth = np.full((nx, ny), np.nan, dtype=np.float32)
            if z_index >= middle:
                relative_depth[mask] = np.float32(1.0) - laplacian[mask]
            writer.write_plane(relative_depth)
            if z_index % 200 == 0:
                print(f"  relative-depth plane {z_index}/{nz}", flush=True)


def generate_labeled_mask(input_dir: Path, output_dir: Path) -> None:
    isocortex_path = input_dir / "isocortex_mask_10.nrrd"
    boundary_path = input_dir / "isocortex_boundary_10.nrrd"
    isocortex_header = nrrd.read_header(str(isocortex_path))
    boundary_header = nrrd.read_header(str(boundary_path))
    check_matching_grids(isocortex_header, boundary_header)

    with ExitStack() as stack:
        isocortex_reader = stack.enter_context(NrrdPlaneReader(isocortex_path))
        boundary_reader = stack.enter_context(NrrdPlaneReader(boundary_path))
        writer = stack.enter_context(
            NrrdPlaneWriter(
                output_dir / "mask.nrrd", boundary_header, np.dtype("int8")
            )
        )
        nx, ny, nz = isocortex_reader.sizes
        middle = nz // 2
        for z_index in range(nz):
            isocortex = isocortex_reader.read_plane() != 0
            boundary = boundary_reader.read_plane()
            result = np.zeros((nx, ny), dtype=np.int8)
            if z_index >= middle:
                result[(boundary == 0) & isocortex] = 1
                result[boundary == 4] = 2
                result[boundary == 3] = 3
                result[boundary == 1] = 4
            writer.write_plane(result)
            if z_index % 200 == 0:
                print(f"  labeled-mask plane {z_index}/{nz}", flush=True)


def _dilate_label_in_window(
    window: list[tuple[int, np.ndarray, np.ndarray]],
    label: int,
    radius: int,
    middle: int,
) -> np.ndarray:
    """Return the center plane of a 3D Chebyshev-radius dilation."""
    center = len(window) // 2
    combined = np.zeros(window[center][1].shape, dtype=bool)
    for offset in range(-radius, radius + 1):
        z_index, boundary, _ = window[center + offset]
        if z_index >= middle:
            combined |= boundary == label
    return ndimage.maximum_filter(
        combined,
        size=(2 * radius + 1, 2 * radius + 1),
        mode="constant",
    )


def _coordinates(mask: np.ndarray, z_index: int) -> np.ndarray:
    xy = np.argwhere(mask).astype(np.int32, copy=False)
    if xy.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    z = np.full((len(xy), 1), z_index, dtype=np.int32)
    return np.concatenate((xy, z), axis=1)


def collect_extension_coordinates(
    input_dir: Path,
) -> tuple[dict[str, np.ndarray], tuple[int, int, int]]:
    """Collect the sparse voxel sets used by the original depth extension."""
    isocortex_path = input_dir / "isocortex_mask_10.nrrd"
    boundary_path = input_dir / "isocortex_boundary_10.nrrd"
    coordinate_lists: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    with ExitStack() as stack:
        isocortex_reader = stack.enter_context(NrrdPlaneReader(isocortex_path))
        boundary_reader = stack.enter_context(NrrdPlaneReader(boundary_path))
        if isocortex_reader.sizes != boundary_reader.sizes:
            raise ValueError("Isocortex and boundary grids do not match")
        nx, ny, nz = isocortex_reader.sizes
        middle = nz // 2
        zero_boundary = np.zeros((nx, ny), dtype=boundary_reader.dtype)
        zero_mask = np.zeros((nx, ny), dtype=isocortex_reader.dtype)
        window: list[tuple[int, np.ndarray, np.ndarray]] = [
            (-2, zero_boundary, zero_mask),
            (-1, zero_boundary, zero_mask),
        ]

        for input_z in range(nz + 2):
            if input_z < nz:
                boundary = boundary_reader.read_plane()
                isocortex = isocortex_reader.read_plane()
            else:
                boundary = zero_boundary
                isocortex = zero_mask
            window.append((input_z, boundary, isocortex))
            if len(window) < 5:
                continue

            center_z, _, center_isocortex = window[2]
            if 0 <= center_z < nz:
                inside = (center_isocortex != 0) & (center_z >= middle)
                outside = ~inside
                top = _dilate_label_in_window(window, 1, 1, middle) & outside
                bottom = _dilate_label_in_window(window, 3, 1, middle) & outside
                side_destination = (
                    _dilate_label_in_window(window, 4, 1, middle) & outside
                )
                side_source = (
                    _dilate_label_in_window(window, 4, 2, middle) & inside
                )
                coordinate_lists["top"].append(_coordinates(top, center_z))
                coordinate_lists["bottom"].append(_coordinates(bottom, center_z))
                coordinate_lists["side_destination"].append(
                    _coordinates(side_destination, center_z)
                )
                coordinate_lists["side_source"].append(
                    _coordinates(side_source, center_z)
                )
                if center_z % 100 == 0:
                    print(f"  extension masks plane {center_z}/{nz}", flush=True)
            window.pop(0)

    coordinates = {
        name: np.concatenate(parts, axis=0)
        for name, parts in coordinate_lists.items()
    }
    return coordinates, (nx, ny, nz)


def _values_at_coordinates(path: Path, coordinates: np.ndarray) -> np.ndarray:
    values = np.empty(len(coordinates), dtype=np.float32)
    with NrrdPlaneReader(path) as reader:
        for z_index in range(reader.sizes[2]):
            start = np.searchsorted(coordinates[:, 2], z_index, side="left")
            stop = np.searchsorted(coordinates[:, 2], z_index, side="right")
            plane = reader.read_plane()
            if stop > start:
                xy = coordinates[start:stop, :2]
                values[start:stop] = plane[xy[:, 0], xy[:, 1]]
    return values


def _indices_at_z(coordinates: np.ndarray, z_index: int) -> tuple[int, int]:
    return (
        int(np.searchsorted(coordinates[:, 2], z_index, side="left")),
        int(np.searchsorted(coordinates[:, 2], z_index, side="right")),
    )


def generate_extended_depth(input_dir: Path, output_dir: Path) -> dict:
    coordinates, sizes = collect_extension_coordinates(input_dir)
    counts = {name: int(len(values)) for name, values in coordinates.items()}
    print(f"  extension coordinate counts: {counts}", flush=True)

    side_source_values = _values_at_coordinates(
        output_dir / "relative_depth.nrrd", coordinates["side_source"]
    )
    if not np.all(np.isfinite(side_source_values)):
        raise ValueError("Side-extension source contains non-finite depth values")

    tree = cKDTree(coordinates["side_source"])
    distances, nearest = tree.query(
        coordinates["side_destination"], k=1, workers=-1
    )
    side_destination_values = side_source_values[nearest]
    nearest_stats = {
        "minimum": float(distances.min()),
        "median": float(np.median(distances)),
        "maximum": float(distances.max()),
    }
    print(f"  nearest side-source distances: {nearest_stats}", flush=True)

    relative_depth_path = output_dir / "relative_depth.nrrd"
    relative_depth_header = nrrd.read_header(str(relative_depth_path))
    with ExitStack() as stack:
        reader = stack.enter_context(NrrdPlaneReader(relative_depth_path))
        writer = stack.enter_context(
            NrrdPlaneWriter(
                output_dir / "extension.nrrd",
                relative_depth_header,
                np.dtype("float32"),
            )
        )
        for z_index in range(sizes[2]):
            relative_depth = reader.read_plane()
            extended = np.nan_to_num(relative_depth, nan=0.0).astype(
                np.float32, copy=False
            )
            for name, value in (("top", 1.01), ("bottom", -0.01)):
                start, stop = _indices_at_z(coordinates[name], z_index)
                xy = coordinates[name][start:stop, :2]
                extended[xy[:, 0], xy[:, 1]] = np.float32(value)
            start, stop = _indices_at_z(coordinates["side_destination"], z_index)
            xy = coordinates["side_destination"][start:stop, :2]
            extended[xy[:, 0], xy[:, 1]] = side_destination_values[start:stop]
            writer.write_plane(extended)
            if z_index % 200 == 0:
                print(f"  extended-depth plane {z_index}/{sizes[2]}", flush=True)

    return {"coordinate_counts": counts, "nearest_distances": nearest_stats}


def generate_orientations(input_dir: Path, output_dir: Path) -> None:
    extension_path = output_dir / "extension.nrrd"
    isocortex_path = input_dir / "isocortex_mask_10.nrrd"
    extension_header = nrrd.read_header(str(extension_path))
    isocortex_header = nrrd.read_header(str(isocortex_path))
    check_matching_grids(extension_header, isocortex_header)

    output_paths = {
        "x": output_dir / "orientation_x.nrrd",
        "y": output_dir / "orientation_y.nrrd",
        "z": output_dir / "orientation_z.nrrd",
    }
    with ExitStack() as stack:
        extension_reader = stack.enter_context(NrrdPlaneReader(extension_path))
        isocortex_reader = stack.enter_context(NrrdPlaneReader(isocortex_path))
        writers = {
            name: stack.enter_context(
                NrrdPlaneWriter(path, extension_header, np.dtype("float32"))
            )
            for name, path in output_paths.items()
        }
        _, _, nz = extension_reader.sizes
        middle = nz // 2
        previous = None
        current = extension_reader.read_plane()
        current_mask = isocortex_reader.read_plane()
        following = extension_reader.read_plane()
        following_mask = isocortex_reader.read_plane()

        for z_index in range(nz):
            orientation_x = np.gradient(current, axis=0)
            orientation_y = np.gradient(current, axis=1)
            if z_index == 0:
                orientation_z = following - current
            elif z_index == nz - 1:
                orientation_z = current - previous
            else:
                orientation_z = (following - previous) / np.float32(2.0)

            inside = (current_mask != 0) & (z_index >= middle)
            orientation_x[~inside] = 0
            orientation_y[~inside] = 0
            orientation_z[~inside] = 0
            writers["x"].write_plane(orientation_x)
            writers["y"].write_plane(orientation_y)
            writers["z"].write_plane(orientation_z)

            if z_index % 200 == 0:
                print(f"  orientation plane {z_index}/{nz}", flush=True)
            if z_index < nz - 1:
                previous = current
                current = following
                current_mask = following_mask
                if z_index + 2 < nz:
                    following = extension_reader.read_plane()
                    following_mask = isocortex_reader.read_plane()


def ensure_outputs_absent(
    output_dir: Path, expected: tuple[str, ...], force: bool
) -> None:
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/allen_ccf_10/input"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/allen_ccf_10/output"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing outputs",
    )
    parser.add_argument(
        "--stage",
        choices=("initial", "remaining", "all"),
        default="all",
        help="Select the preprocessing checkpoint to generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_outputs = (
        "mask_hemi.nrrd",
        "mask_hemi_other.nrrd",
        "hemispheres.nrrd",
        "relative_depth.nrrd",
    )
    remaining_outputs = (
        "mask.nrrd",
        "extension.nrrd",
        "orientation_x.nrrd",
        "orientation_y.nrrd",
        "orientation_z.nrrd",
    )
    if args.stage in {"initial", "all"}:
        ensure_outputs_absent(args.output_dir, initial_outputs, args.force)
        print("Generating hemisphere volumes...", flush=True)
        generate_hemisphere_volumes(args.input_dir, args.output_dir)
        print("Generating relative depth...", flush=True)
        generate_relative_depth(args.input_dir, args.output_dir)
    if args.stage in {"remaining", "all"}:
        ensure_outputs_absent(args.output_dir, remaining_outputs, args.force)
        print("Generating labeled mask...", flush=True)
        generate_labeled_mask(args.input_dir, args.output_dir)
        print("Generating extended depth...", flush=True)
        generate_extended_depth(args.input_dir, args.output_dir)
        print("Generating orientation components...", flush=True)
        generate_orientations(args.input_dir, args.output_dir)
    print(f"Wrote preprocessing outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
