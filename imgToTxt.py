import glob,os,re
from natsort import natsorted

a = open("/home/emeka/Schreibtisch/AIS/ais3d/FreiburgImages.txt", "w")
#for path, subdirs, files in sorted(os.walk(r'/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg')):
#   for filename in files.sort(key=os.path.getmtime):
#     f = os.path.join(path, filename)
#     a.write(str(f) + os.linesep)
files = glob.glob('/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg/*.png')
#files = sorted(os.listdir('/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg'), key=os.path.getctime)
files=natsorted(files)
#lsorted = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
#files.sort()
#files.sort(key=lambda x: os.path.getctime(x))
for infile in  files:
     f = os.path.join('/home/dllab/percepcar_video/drive_2017_11_03_09_57_01/left_pg', infile)
     a.write(str(f) + os.linesep)

   #  files.sort(key=os.path.getmtime)





 #   pattern = re.compile(r"^\D+?_\D+?_(.+?)_")

 #   def sort_on_TS(a, b):
 #       return cmp(pattern.match(a).group(1), pattern.match(b).group(1))
#
  #  for item in sorted(file_list, sort_on_TS):
 #       print item

