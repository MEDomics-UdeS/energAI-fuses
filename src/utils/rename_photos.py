import os

rootdir = 'C:/Users/simon.giard-leroux/Google Drive/Maîtrise SGL CIMA+/General/Fuses Survey Dataset 2'

for subdir, dirs, files in os.walk(rootdir):
    for i, file in enumerate(files, start=1):
        os.rename(subdir + os.sep + file, subdir + "-" + str(i) + ".JPG")
