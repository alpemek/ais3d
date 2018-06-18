from pypcd import pypcd
from io import StringIO
# also can read from file handles.
pc = pypcd.PointCloud.from_path('1.pcd')
# pc.pc_data has the data as a structured array
# pc.fields, pc.count, etc have the metadata

# center the x field
pc.pc_data['x'] -= pc.pc_data['x'].mean()

# save as binary compressed
pc.save_pcd('/home/emeka/Schreibtisch/AIS/deleteme/109newpython.pcd', compression='binary')