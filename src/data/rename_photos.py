# -*- coding: utf-8 -*-
#
# import os
#  
# Function to rename multiple files
# def main():
#  
#    for count, filename in enumerate(os.listdir("xyz")): 
#        dst ="Hostel" + str(count) + ".jpg"
#        src ='xyz'+ filename 
#        dst ='xyz'+ dst 
#          
#        # rename() function will 
#        # rename all the files 
#        os.rename(src, dst) 
#  
# Driver Code
# if __name__ == '__main__':
#      
#    # Calling main() function 
#    main() 

import os

rootdir = 'C:/Users/simon.giard-leroux/Google Drive/Maîtrise SGL CIMA+/General/Fuses Survey Dataset 2'

for subdir, dirs, files in os.walk(rootdir):
    i = 1
    for file in files:
        #        terminator = subdir.index('\\')
        #        projectNo = subdir[terminator + 1:]
        #
        #        terminator = projectNo.index(' ')
        #        projectNo = projectNo[:terminator]
        #
        #        print(projectNo)
        #        print(file)

        os.rename(subdir + os.sep + file, subdir + "-" + str(i) + ".JPG")

        i += 1

#    for file in files:
#        print(os.path.join(subdir, file))
