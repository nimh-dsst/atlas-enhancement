# RHEL 8 SLURM and Pixi Refactor Plan

## Purpose

This document defines a refactor that will make the atlas-enhancement flatmap
workflow reproducible on both a workstation and an x86-64 RHEL 8 SLURM
cluster. It is an implementation plan, not a claim that the current checkout
already supports every command described below.

The immediate production target is the Allen CCFv3 10-micron isocortex
workflow. The design should also support later experiments at other
resolutions and with other anatomical structures, including dorsal striatum.

## Goals

- Reproduce the software environment with Pixi and a committed lock file.
- Support Apple Silicon workstation development and RHEL 8 `linux-64` compute.
- Build the standard CGAL tools and the MKL/PARDISO flattening executable.
- Run exact high-resolution Stage I flattening on one high-memory node.
- Run Stage II streamline projection as restartable SLURM array jobs.
- Run Stage III using explicit workstation or production mesh profiles.
- Preserve provenance, logs, validation reports, and resource measurements.
- Keep multi-gigabyte atlas inputs and generated meshes outside Git.
- Avoid cluster-specific account, partition, or filesystem paths in source.

## Non-goals

- Store NRRDs, OFF meshes, or other large generated data in Git.
- Make the existing Make workflow distributed by hiding `srun` inside recipes.
- Assume that every compute node has internet access.
- Require root access, Homebrew, or a cluster-wide Conda installation.
- Generalize the anatomical shaping method before the isocortex reference run
  has passed end-to-end validation.

## Current baseline

The current workstation checkpoint has:

- Validated Allen 10-micron preprocessing inputs.
- A `d = 0.5` point cloud with 437,186 points in its largest component.
- A manifold base mesh with 437,186 vertices and 873,053 triangles.
- A three-refinement mesh with 27,942,965 vertices and 55,875,392 triangles.
- A valid workstation flat mesh made by flattening the base mesh and applying
  the same three deterministic subdivisions in 2D.
- A native Pixi environment for `osx-arm64`.
- Source fixes needed by current CGAL, Gmsh, and macOS.

See [`PROJECT_PROGRESS.md`](PROJECT_PROGRESS.md) for the detailed checkpoint.

The principal unresolved production calculation is the exact iterative
authalic solve on the three-refinement mesh. The repository's MKL source uses
Eigen's `PardisoLU`, which is a threaded, single-node sparse direct solver.

## Assumed cluster model

The first implementation will target:

- RHEL 8 or a compatible distribution.
- x86-64 compute nodes.
- SLURM scheduling.
- A shared project or scratch filesystem visible to compute nodes.
- Node-local temporary storage exposed through `$SLURM_TMPDIR` when available.
- High-memory nodes for the MKL/PARDISO solve.
- Outbound network access on a login or data-transfer node, but not necessarily
  on compute nodes.

The following values must remain site configuration, not committed defaults:

- SLURM account.
- Partition and quality of service.
- Maximum wall time and memory per partition.
- Project, scratch, and node-local storage roots.
- Module names and versions.
- Proxy or package-mirror configuration.

## Proposed repository layout

```text
hpc/
├── README.md
├── config/
│   ├── cluster.env.example
│   ├── isocortex_10.env.example
│   └── resources.env.example
├── scripts/
│   ├── capture_provenance.sh
│   ├── stage_inputs.sh
│   ├── validate_off.py
│   ├── validate_flat_mesh.py
│   ├── merge_streamline_blocks.py
│   └── summarize_sacct.sh
└── slurm/
    ├── build_native.sbatch
    ├── flatten_profile.sbatch
    ├── flatten_production.sbatch
    ├── streamlines_array.sbatch
    ├── streamlines_merge.sbatch
    ├── nearest.sbatch
    └── make_flatmap.sbatch

profiles/
├── workstation.mk
└── hpc_isocortex_10.mk
```

Scripts will accept paths through arguments or environment variables. They
will not embed a username, account, partition, or absolute site path.

## Phase 1: make the Pixi workspace cross-platform

### Manifest changes

Extend `pixi.toml` to support both the workstation and cluster:

```toml
[workspace]
platforms = [
  "osx-arm64",
  { platform = "linux-64", glibc = "2.28" },
]
```

The RHEL 8 glibc constraint will prevent locking packages that require a newer
runtime than the cluster provides.

Separate shared and platform-specific dependencies:

- Shared native dependencies: CGAL, Eigen, Boost, GMP, MPFR, GSL, CMake,
  Ninja, GNU Make, Gawk, GNU Parallel, Gmsh, and zlib.
- macOS dependency: LLVM OpenMP.
- Linux dependencies: Intel oneMKL development files, a compatible OpenMP
  runtime, TBB where required, and `linux-64` compilers.
- Python dependencies required by preprocessing and validation.

If AllenSDK's NumPy constraints cannot coexist cleanly with the newest native
stack, define separate Pixi environments:

- `native`: CGAL, Gmsh, flatpath, and build tools.
- `allen`: AllenSDK and acquisition utilities.
- `workflow`: Voxcell, pynrrd, SciPy, NumPy, Matplotlib, and validation tools.
- `hpc-mkl`: Linux-only native feature with MKL/PARDISO.

### Reproducibility policy

- Commit `pixi.toml` and `pixi.lock` together.
- Cluster jobs use `pixi run --locked` or an activation generated from the
  locked environment.
- Do not solve or update the environment from a compute job.
- Install the locked environment on a login node or in a dedicated setup job.
- Record `pixi --version`, the lock-file checksum, and `pixi list` in each run's
  provenance directory.

### Offline compute nodes

If compute nodes cannot reach package servers:

1. Install the environment on a network-enabled login node into shared storage.
2. Verify executables from a short SLURM smoke job.
3. Use the shared environment read-only from production jobs.
4. If shared-environment startup is too slow, stage the environment or required
   shared libraries to node-local storage.

### Acceptance criteria

- `pixi install --locked` succeeds on `osx-arm64` and RHEL 8 `linux-64`.
- The lock file is unchanged after installation on either platform.
- `pixi run` reports the expected CGAL, Gmsh, CMake, compiler, and MKL versions.

## Phase 2: refactor and test native builds

### Standard CGAL build

Define Pixi tasks for configure, build, and test rather than requiring users to
remember CMake flags. Example task names:

- `configure-cgal`
- `build-cgal`
- `test-cgal`
- `build-flatpath`
- `test-flatpath`

Build directories should identify the platform and solver, for example:

```text
build/surf_cgal/linux-64-eigen/
build/surf_cgal/linux-64-mkl/
```

Generated build files and binaries remain ignored by Git.

### MKL/PARDISO source work

Before production use, update `flatmap/code/surf_cgal/mkl` to match the fixes
already applied to the standard build:

- Use a modern CMake compatibility range.
- Build the standard iterative parameterizer without requiring custom convex
  border headers.
- Add the parent source directory to the include path for
  `Flatten_common.hpp`.
- Replace unsafe `reserve(4)` indexing with `resize(4)`.
- Link through the imported MKL target.
- Keep the LP64 interface unless mesh and factor indices are deliberately
  migrated and tested with 64-bit storage indices.
- Print the selected solver, MKL version, and maximum thread count at startup.

### Native test suite

Automate the synthetic tests performed during workstation setup:

1. Reconstruct a small sphere point cloud.
2. Validate the OFF header and record counts.
3. Round-trip the mesh through Gmsh refinement.
4. Flatten a curved disk to the unit square.
5. Verify finite UV coordinates, zero `z`, and no flipped triangles.
6. Run `Nearest_KNN` on known query points.
7. Run flatpath's dots, depth, and height reference cases with a documented
   floating-point tolerance rather than byte-for-byte output comparison.

Run the same geometric flattening test with Eigen and MKL. The UV coordinates
may differ within solver tolerance, but topology, bounds, and orientation must
match.

### Acceptance criteria

- All standard native tools build on macOS and Linux.
- The MKL executable builds and passes its synthetic flattening test on Linux.
- No production job begins until the synthetic MKL test succeeds on a compute
  node.

## Phase 3: data and artifact management

### Data roots

Use explicit roots:

```text
ATLAS_SOURCE_ROOT   repository checkout
ATLAS_DATA_ROOT     immutable atlas inputs
ATLAS_RUN_ROOT      outputs for one parameterized run
ATLAS_SCRATCH_ROOT  temporary and restartable intermediates
```

Each run gets a unique directory based on anatomy, resolution, configuration,
and commit, for example:

```text
runs/isocortex_10um_d050_<short-git-sha>/
```

### Input transfer

- Use `rsync --partial --info=progress2` or the site's supported transfer tool.
- Never transfer `.pixi`, `.venv`, CMake build directories, or macOS metadata.
- Never add NRRDs or OFF meshes to Git.
- Generate SHA-256 checksums for large immutable inputs after transfer.

For the exact flattening job, only these data files are initially required:

- `refined_mesh_n2.off` for profiling.
- `refined_mesh.off` for production.

The Allen NRRDs are not required until Stage II.

### Atomic outputs

Every job writes to a temporary filename in the destination filesystem, runs
validation, and renames the file only after success. Interrupted jobs must not
leave an apparently complete production filename.

### Provenance

Every run records:

- Git commit and dirty status.
- Pixi lock checksum and package list.
- SLURM job ID and job description.
- Hostname, CPU model, thread count, and memory allocation.
- Effective workflow parameters.
- Input and output checksums.
- Wall time and peak resident memory from `sacct`.
- Complete stdout and stderr logs.

## Phase 4: exact Stage I flattening on a high-memory node

### Execution model

MKL PARDISO is a shared-memory solver in this executable. Request exactly one
node and one task, with multiple CPUs assigned to that task:

```text
nodes=1
ntasks=1
cpus-per-task=<physical cores>
mem=<total node memory required>
```

Set thread controls from the SLURM allocation:

```bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Do not start multiple MPI ranks. Additional nodes will not accelerate this
particular solver implementation.

### Profiling run

Run the two-refinement mesh first:

- Input: 6,987,059 vertices and 13,968,848 triangles.
- Initial request: one node, 32 physical cores, 256 GiB, and 8 hours.
- Output: `flat_mesh_n2_mkl.off`.

Validate it and capture `MaxRSS`, elapsed time, CPU efficiency, and output size.
Use the measured factorization memory—not only the input mesh size—to select a
production node. Sparse direct-solver memory can grow faster than the vertex
count because of fill-in.

### Production run

- Input: `refined_mesh.off` with 27,942,965 vertices.
- Output: `flat_mesh_hpc.off`.
- Use 10 authalic iterations and boundary offset 0.
- Start with a 512-GiB or larger high-memory node only if the profiling result
  supports that request.
- Request additional memory rather than allowing swap on a compute node.

### Validation gate

The production flat mesh must satisfy:

- Exact expected OFF vertex and face counts.
- Exact face connectivity match with `refined_mesh.off`.
- Finite UV coordinates.
- `u` and `v` within tolerance of `[0, 1]`.
- Every `z` equal to zero within tolerance.
- No flipped or zero-area triangles.
- A generated anatomical-coordinate and area-distortion checkpoint image.

Compare `flat_mesh_hpc.off` with the workstation approximation:

- Per-vertex UV displacement percentiles.
- Normalized triangle-area distortion percentiles.
- Maximum displacement and its anatomical location.
- Boundary correspondence and corner selection.

Do not overwrite the workstation mesh until this comparison is complete.

## Phase 5: refactor Stage II into SLURM array jobs

### Why the current recipe should change

The current 10-micron input has 61,945,750 finite right-isocortex voxels. With
the example block size of 100,000, the workflow creates approximately 620
blocks. The current Make recipe is cluster-oriented but has several weaknesses:

- Block orchestration is embedded inside Make and GNU Parallel.
- Each block reloads four full fields.
- Output completion is inferred from filenames rather than validated manifests.
- A failed nested `srun` is harder to resume and audit.
- Shared-filesystem input traffic can dominate when many tasks start together.

### Proposed array design

1. Convert NRRDs once to the flatpath binary representation.
2. Generate a versioned block manifest containing:
   - block ID;
   - first finite-voxel ordinal;
   - last finite-voxel ordinal;
   - expected output path;
   - parameter hash.
3. Submit one SLURM array where each task reads one manifest row.
4. Stage the compressed fields to `$SLURM_TMPDIR` once per node when practical.
5. Run one flatpath process per allocated task.
6. Validate line count and six-column schema before atomically publishing the
   block output.
7. Submit merge and validation as a dependent job.

The array concurrency limit must be configurable so the shared filesystem is
not flooded by hundreds of simultaneous decompressions.

### Flatpath refactor candidates

Profile before selecting one of these approaches:

1. **Minimal change:** keep independent blocks, improve manifests and staging.
2. **Persistent worker:** load the four fields once and process several blocks
   sequentially in one allocation.
3. **Direct projection mode:** compute a streamline and its intersection, write
   the result, and free the path immediately instead of retaining all paths in
   memory.
4. **Cropped field profile:** crop the right-isocortex bounding box with padding,
   preserve physical offsets, and translate local indices back to global CCF
   indices during merge.

The direct projection mode is the most promising long-term improvement because
Stage II only needs the `d = 0.5` intersection, not every saved streamline.

### Numerical pilot

Before the full array, compare a fixed stratified voxel sample using integration
steps of 0.5, 2.5, and 5.0 micrometres. Record:

- Runtime per voxel.
- Fraction of complete streamlines.
- Intersection displacement relative to the 0.5-micrometre reference.
- Displacement percentiles in units of a 10-micrometre voxel.

Keep the published 0.5-micrometre setting for the reference run unless the
comparison justifies a separate workstation profile.

### Acceptance criteria

- Every expected block is present exactly once.
- Blocks can be safely rerun without affecting completed blocks.
- Merge order is deterministic.
- Voxel indices are unique and within the atlas grid.
- Intersection coordinates are finite for the accepted subset.
- Valid and rejected streamline counts are reported.

## Phase 6: refactor Stage III profiles

### Workstation profile

Use the validated base pair:

- `projection_mesh.off`
- `flat_mesh_n0.off`

This limits nearest-neighbor memory while preserving approximately one base
surface vertex per extracted 10-micron surface voxel.

### Production profile

Use the exact refined pair:

- `refined_mesh.off`
- `flat_mesh_hpc.off`

The mesh filenames must be selected by an explicit profile. Stage III must
validate matching topology before nearest-neighbor assignment.

### Nearest-neighbor refactor

The current executable loads every query point before searching. Refactor it to
stream query points and write results incrementally after building the mesh
tree. This bounds query memory and makes the single high-memory mesh load the
dominant cost.

If the full refined mesh still exceeds the selected node's memory:

- Profile a base-mesh production alternative.
- Consider a spatially partitioned surface index.
- Do not independently partition the mesh without overlap and boundary tests.

### Flatmap assembly

Refactor `make_flatmap.py` if required to write the final two-component NRRD in
planes or chunks. The current full-array approach can require several times the
raw atlas size during assembly.

### Acceptance criteria

- Every accepted voxel has one UV coordinate.
- UV values are finite and within `[0, 1]`.
- Missing voxels retain the documented sentinel value.
- Output NRRD metadata match the source atlas grid.
- Discretization at 256 pixels succeeds.
- Coverage and continuity metrics are generated.

## Phase 7: orchestration and documentation

### Submission interface

Provide a small submission wrapper that:

- Loads site configuration from a user-owned environment file.
- Validates required paths and free space.
- Submits jobs with explicit dependencies.
- Prints job IDs and expected output locations.
- Never deletes data automatically.

The dependency chain should be visible:

```text
environment/build smoke test
    ↓
Stage I profile → Stage I production → Stage I validation
    ↓
Stage II conversion → Stage II array → merge/validation
    ↓
Stage III nearest → flatmap assembly → discretization/metrics
```

### Documentation

Document:

- Installing Pixi without administrative privileges.
- Configuring a RHEL 8 cluster profile.
- Staging data.
- Running smoke tests.
- Submitting and monitoring each stage.
- Resuming failed arrays.
- Reading validation reports.
- Returning selected artifacts to a workstation.

## Resource and storage planning

Before submission, estimate and verify:

- Input and output size per stage.
- Temporary duplicate space required by atomic writes.
- Pixi environment size.
- SLURM stdout/stderr volume.
- Node-local scratch capacity.
- Project quota and inode limits.

At minimum, reserve enough shared storage for:

- Immutable Allen inputs.
- Base and refined 3D meshes.
- Workstation and exact 2D meshes.
- Stage II block outputs plus merged output.
- Stage III text intermediates and NRRDs.
- One temporary copy of the largest file being produced.

## Failure and restart policy

- Jobs use `set -euo pipefail`.
- Production output names appear only after validation.
- SLURM array tasks are idempotent.
- A parameter or input checksum change creates a new run directory.
- Failed jobs preserve logs and temporary diagnostics but are never interpreted
  as completed outputs.
- Cleanup is always an explicit command with a printed target list.
- No workflow command removes immutable inputs.

## Security and operational policy

- Do not commit passwords, access tokens, SSH keys, cluster accounts, or private
  filesystem paths.
- Keep site configuration in ignored user files derived from committed examples.
- Prefer SSH-agent forwarding or the site's approved authentication mechanism;
  do not copy private keys into the repository or compute environment.
- Pin external downloads by version and checksum where possible.

## Implementation order

1. Add `linux-64` and RHEL 8 compatibility to Pixi.
2. Port and test the MKL/PARDISO build.
3. Add native synthetic tests and Pixi tasks.
4. Add provenance and OFF/flat-mesh validation tools.
5. Run and measure the two-refinement MKL profile.
6. Run and validate exact Stage I flattening.
7. Replace nested Stage II orchestration with a SLURM array manifest.
8. Implement and benchmark flatpath direct projection mode.
9. Add explicit workstation and production Stage III profiles.
10. Stream nearest-neighbor queries and flatmap NRRD assembly if profiling
    shows they are required.
11. Run the complete 10-micron reference workflow.
12. Only then generalize the shaping workflow to 25 microns or other regions.

## Information needed from the target cluster

Before implementing site-specific submission examples, collect:

```bash
uname -m
sinfo -o "%P %c %m %l %a"
module spider oneapi 2>&1 | head -80
module spider mkl 2>&1 | head -80
module spider cgal 2>&1 | head -80
module spider cmake 2>&1 | head -80
```

Also record, without committing secrets:

- SLURM account and allowed partitions.
- High-memory node sizes.
- Shared and node-local scratch paths.
- Internet-access restrictions.
- Maximum array size and concurrency policies.
- Filesystem quotas.

## Definition of done

The refactor is complete when a clean RHEL 8 checkout can:

1. Install the locked Pixi environment without administrative privileges.
2. Build and pass standard and MKL native smoke tests.
3. Reproduce the exact Stage I flat mesh on a high-memory node.
4. Resume and complete Stage II through validated SLURM array jobs.
5. Produce continuous and discretized Stage III flatmap NRRDs.
6. Generate provenance, resource, topology, coverage, and distortion reports.
7. Reproduce the same outputs from the same Git commit, lock file, inputs, and
   configuration without manual edits inside the workflow.
