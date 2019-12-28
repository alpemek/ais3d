import glob,os,re
from natsort import natsorted

sorted_file = open("/home/emeka/Schreibtisch/AIS/ais3d/FreiburgImages.txt", "w")
files = glob.glob('/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg/*.png')
files = natsorted(files)

for infile in  files:
    file_path = os.path.join('/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg', infile)
    sorted_file.write(str(file_path) + os.linesep)