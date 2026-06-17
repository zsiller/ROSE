import h5py
import numpy as np

cgns_file = "/home/zhsiller/research/ROSE/ShockTube_mem01_2d_000060.cgns"

# CGNS HDF5 layout: each field (Density, etc.) is a *group* with a child dataset
# named " data" (leading space), not a dataset you can slice directly.


def read_cgns_field(f: h5py.File, field_group: h5py.Group) -> np.ndarray:
    """Read a CGNS solution field from its group node."""
    if " data" not in field_group:
        raise KeyError(
            f"No ' data' dataset under {field_group.name}; keys={list(field_group.keys())}"
        )
    return np.asarray(field_group[" data"][...])


with h5py.File(cgns_file, "r") as f:
    print("Datasets in the CGNS file:")
    f.visititems(
        lambda name, obj: (
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            if isinstance(obj, h5py.Dataset)
            else None
        )
    )

    density_group = f["Base/Zone_000001/Solution/Density"]
    density_data = read_cgns_field(f, density_group)

    grid_group = f["Base/Zone_000001/Solution/GridLocation"]
    grid_data = read_cgns_field(f, grid_group)

    print(grid_data)

    # This file: zone size 63x63 -> 3969 nodal values
    print("\nDensity data shape (flat):", density_data.shape)
    print("Density data (sample):", density_data.ravel()[:8])
   

    if density_data.size == 3969:
        density_2d = density_data.reshape(63, 63)
        print("Reshaped to 2D:", density_2d.shape)
        print(density_2d)
