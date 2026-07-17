# Project Progress: Allen Mouse Isocortex Flatmap

Last updated: 2026-07-17

## Current status

We have completed the preprocessing and surface-generation portion of the Allen CCFv3 10-micron isocortex example. We now have a valid 3D middle-depth cortical mesh and a corresponding 2D square mesh.

We have not yet generated the final voxel-wise flatmap NRRD.

The overall workflow is:

```text
Allen volumes
    ↓
relative depth + orientation fields
    ↓
d = 0.5 middle-depth point cloud
    ↓
reconstructed 3D surface
    ↓
refined 3D mesh
    ↓
flattened 2D mesh
    ↓
Stage II: project voxels along streamlines
    ↓
Stage III: assign 2D coordinates to every voxel
```

## 1. Assembled and inspected the 10-micron inputs

The input directory is [`data/allen_ccf_10/input`](data/allen_ccf_10/input).

It contains:

- Allen annotation volume
- Allen average template
- Allen structure hierarchy
- Isocortex mask
- Isocortex boundary labels
- Laplacian field

We inspected anatomical slices through these volumes and confirmed their dimensions, orientation, coordinate system, and 10-micron spacing.

We also discussed resampling:

- Continuous fields such as intensity, depth, and orientation can be interpolated.
- Label volumes and masks require nearest-neighbor or topology-aware resampling.
- Downsampling existing results is useful, but it is not necessarily equivalent to solving the anatomy again at 25, 50, or 100 microns.

## 2. Installed the Python environment

The Python environment is located at [`.venv`](.venv).

It contains Voxcell, AllenSDK, NumPy, SciPy, pynrrd, Matplotlib, and the other Python packages needed by the repository.

Voxcell is responsible for reading NRRD metadata and converting between array indices and physical Allen CCF coordinates.

## 3. Converted the Allen inputs into flatmap inputs

The resulting workflow input directory is [`data/allen_ccf_10/output`](data/allen_ccf_10/output).

The important volumes are:

- [`relative_depth.nrrd`](data/allen_ccf_10/output/relative_depth.nrrd)
- [`mask.nrrd`](data/allen_ccf_10/output/mask.nrrd)
- [`orientation_x.nrrd`](data/allen_ccf_10/output/orientation_x.nrrd)
- [`orientation_y.nrrd`](data/allen_ccf_10/output/orientation_y.nrrd)
- [`orientation_z.nrrd`](data/allen_ccf_10/output/orientation_z.nrrd)
- [`hemispheres.nrrd`](data/allen_ccf_10/output/hemispheres.nrrd)
- [`extension.nrrd`](data/allen_ccf_10/output/extension.nrrd)

Conceptually:

- `relative_depth.nrrd` assigns each isocortical voxel a value from 0 at the white-matter side to 1 at the pial side.
- `extension.nrrd` extends those depth values slightly outside the mask so gradients can be calculated cleanly near boundaries.
- The three orientation volumes contain the gradient of the extended depth field.
- Together, those components form a vector pointing approximately along cortical columns toward increasing depth.
- `mask.nrrd` distinguishes interior, side, bottom, and top voxels.

We inspected the orientation vectors and confirmed that they were finite, restricted to the intended hemisphere, and aligned sensibly with the depth field.

## 4. Built the native toolchain

Because Homebrew was not writable, we created a repository-local Pixi environment:

- [`pixi.toml`](pixi.toml)
- [`pixi.lock`](pixi.lock)

It provides:

- CGAL
- Eigen
- Gmsh
- CMake
- GNU Make and utilities
- OpenMP
- GMP and MPFR

We compiled and validated:

- `Flatten_Authalic`
- `Flatten_Authalic_Iterative`
- `Nearest_KNN`
- `Reconstruct_PCAlpha`
- `Reconstruct_PCAdv`
- `Reconstruct_JetAdv`
- `flatpath`

We also corrected several repository issues:

- Modern CGAL/CMake compatibility
- A vector allocation bug in automatic square-corner selection
- A quadratic non-manifold audit that made large meshes appear to hang
- OFF-to-PLY conversion when CGAL writes blank header lines
- macOS GNU utility discovery
- Unnecessary dependency on the unavailable `cmod` generator
- Excessive memory use during isosurface extraction

Synthetic reconstruction, flattening, nearest-neighbor, Gmsh refinement, and streamline tests all passed.

## 5. Configured the isocortex run

The configuration is [`data/allen_ccf_10/output/config.mk`](data/allen_ccf_10/output/config.mk).

The important Stage I settings are:

```make
PROJECTION_SURFACE_DSTAR := 0.5
PROJECTION_SURFACE_SIDE := top
RECONSTRUCT_SURFACE_EXTRA := 12 300 4
NREFINE := 3
FLATTEN_MESH_EXTRA := 10 0
```

This means:

- Use the middle-depth surface, `d = 0.5`.
- Select voxels on the upper side of that threshold.
- Reconstruct using 12 neighbors, 300 samples, and 4 smoothing iterations.
- Refine every triangle three times.
- Run 10 iterative authalic flattening iterations with boundary offset 0.

## 6. Extracted the middle-depth point cloud

Files:

- [`isosurface_dots.nrrd`](flatmap/workflow/01_stageI/01_extract_surface/output/isosurface_dots.nrrd)
- [`isosurface_dots.xyz`](flatmap/workflow/01_stageI/01_extract_surface/output/isosurface_dots.xyz)

Results:

- 437,268 candidate surface voxels
- 39 connected components
- Largest component: 437,186 points
- Only 82 isolated points were discarded
- Every point lies exactly at the center of a 10-micron voxel

The `.xyz` file contains physical Allen CCF coordinates in micrometres, not array indices.

## 7. Reconstructed the surface mesh

File:

- [`projection_mesh.off`](flatmap/workflow/01_stageI/02_reconstruct_mesh/output/projection_mesh.off)

Results:

- 437,186 vertices
- 873,053 triangles
- One connected component
- Zero non-manifold vertices
- One boundary loop
- Euler characteristic `V - E + F = 1`, confirming disk topology
- Surface area approximately 51.652 square millimetres
- Boundary perimeter approximately 21.485 millimetres

This is the geometric middle-depth sheet of one isocortical hemisphere.

## 8. Refined the 3D mesh three times

File:

- [`refined_mesh.off`](flatmap/workflow/01_stageI/03_refine_mesh/output/refined_mesh.off)

Every refinement splits one triangle into four. Three refinements therefore multiply the triangle count by `4^3 = 64`.

Final result:

- 27,942,965 vertices
- 55,875,392 triangles
- Approximately 2.4 GiB

We also retained the two-refinement mesh for profiling:

- [`refined_mesh_n2.off`](flatmap/workflow/01_stageI/03_refine_mesh/output/refined_mesh_n2.off)

## 9. Flattened the mesh

The flattening method:

1. Finds the single boundary loop.
2. Selects four nearly equally spaced boundary points.
3. Maps those points to the corners of a unit square.
4. Maps the rest of the boundary according to arc length.
5. Solves for interior `(u, v)` coordinates.
6. Iteratively adjusts the solution to reduce area distortion.

The exact 28-million-vertex Eigen solve exceeded the workstation's 36 GiB of RAM. Even the 6.99-million-vertex solve remained unfinished after more than an hour.

We therefore created a validated workstation approximation:

1. Run the full 10-iteration authalic solve on the original 437,186-vertex mesh.
2. Apply the same three midpoint refinements to its 2D coordinates.
3. Confirm that its topology matches the full refined 3D mesh exactly.

Files:

- Base solution: [`flat_mesh_n0.off`](flatmap/workflow/01_stageI/04_flatten_mesh/output/flat_mesh_n0.off)
- Full-resolution result: [`flat_mesh.off`](flatmap/workflow/01_stageI/04_flatten_mesh/output/flat_mesh.off)

Validation:

- 27,942,965 vertices
- 55,875,392 triangles
- All triangle records exactly match the 3D refined mesh
- `u, v` are both bounded by `[0, 1]`
- Every `z = 0`
- No folded triangles
- No zero-area triangles
- Total planar area exactly 1

The visual checkpoint and numerical statistics are:

- [`stage1_flat_mesh_checkpoint.png`](data/allen_ccf_10/inspection/stage1_flat_mesh_checkpoint.png)
- [`stage1_flat_mesh_statistics.json`](data/allen_ccf_10/inspection/stage1_flat_mesh_statistics.json)

## Where we are now

Stage I is complete, with one caveat: the current 2D mesh uses the workstation approximation instead of re-solving all 28 million vertices.

The HPC task will only need to replace that one calculation:

```text
refined_mesh.off
    ↓ MKL/PARDISO authalic solve
flat_mesh_hpc.off
```

We do not need to repeat the Allen downloads, preprocessing, point extraction, reconstruction, or Gmsh refinement.

The proposed HPC sequence is:

1. Transfer `refined_mesh_n2.off` for a profiling run.
2. Build `Flatten_Authalic_Iterative` with Intel MKL/PARDISO.
3. Run the two-refinement mesh on one high-memory node and record its peak memory use.
4. Size and run the exact three-refinement job using that measurement.
5. Bring `flat_mesh_hpc.off` back and compare it numerically with the workstation approximation.

The full RHEL 8, SLURM, and Pixi refactor design is documented in
[`HPC_PIXI_IMPLEMENTATION_PLAN.md`](HPC_PIXI_IMPLEMENTATION_PLAN.md).

## Remaining flatmap stages

After the HPC solve:

1. **Stage II:** Trace a streamline through every isocortical voxel using the orientation field and intersect it with the middle-depth mesh.
2. **Stage III:** Transfer the corresponding mesh vertex's `(u, v)` coordinate back to each voxel and write the final flatmap NRRD.
3. Inspect, discretize, and evaluate the completed flatmap.
4. Use what we learned to decide whether to rebuild at 25-micron resolution and how to adapt the method to dorsal striatum and other structures.

## Storage note

At the time of this checkpoint, approximately 29 GiB of local disk space remained. The large mesh files should therefore be handled carefully and should not be regenerated or deleted casually.
