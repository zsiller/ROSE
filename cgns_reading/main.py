import h5py
import os
import numpy as np
root_dir = "/home/zhsiller/research/ROSE/file_reading_tester"
DATA_DIR = root_dir + "/data"
from sklearn.decomposition import PCA

#print out the number of files in the data directory
len_files = len(os.listdir(DATA_DIR))

shape = (61*len_files, 256)

layout =  h5py.VirtualLayout(shape=shape, dtype=np.float32)
print(layout.shape)

row_offset = 0
for n in sorted(os.listdir(DATA_DIR)):
    if not n.endswith(".h5"):
        continue
    filename = n
    print(filename)
    vsource = h5py.VirtualSource(DATA_DIR + "/" + filename, 'rho', shape=(61, 256))
    print(vsource.shape)
    layout[row_offset:row_offset+61, :] = vsource
    row_offset += 61

print(layout[0,:])

with h5py.File('VDS.h5', 'w', libver='latest') as f:
    f.create_virtual_dataset('rho_combined', layout, fillvalue=None)
    f.flush()  # Ensure all metadata is written

#read hdf5 file
with h5py.File('VDS.h5', 'r') as f:
    rho = f['rho_combined'][:]
    print(rho.shape)
    print(rho.dtype)
    print(rho[0,:])



print(layout[0,:])

# pca = PCA(n_components=2)
# pca.fit(layout)




# for file in sorted(os.listdir(DATA_DIR)):
#     if not file.endswith(".h5"):
#         continue
#     file_path = os.path.join(DATA_DIR, file)
#     with h5py.File(file_path, "r") as f:
#         rho = f['rho'][:]
#         print(rho.shape)
#         print(rho[0,:])

