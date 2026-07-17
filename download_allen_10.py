from pathlib import Path

from allensdk.core.reference_space_cache import ReferenceSpaceCache

output = Path("data/allen_ccf_10/input")
output.mkdir(parents=True, exist_ok=True)

cache = ReferenceSpaceCache(
    resolution=10,
    reference_space_key="annotation/ccf_2017",
    manifest=str(output / "manifest.json"),
)

print("Downloading 10 µm CCFv3 annotation...")
cache.get_annotation_volume(file_name=str(output / "annotation_10.nrrd"))

print("Downloading optional anatomical template...")
cache.get_template_volume(file_name=str(output / "average_template_10.nrrd"))

print("Finished.")
