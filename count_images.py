import os
import matplotlib.pyplot as plt

path = "C:/Users/gias2402/Google Drive/Maîtrise SGL CIMA+/General/Fuses Survey Dataset 2"

mn = 20
fuseDict = {}

folders = ([name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))])

for folder in folders:
    contents = os.listdir(os.path.join(path,folder))
    
    if len(contents) > mn:
        fuseDict[folder] = len(contents)

x = sorted(fuseDict, key=fuseDict.get, reverse=True)
y = sorted(fuseDict.values(), reverse=True)
    
plt.bar(x,y)
plt.show()
plt.xticks(rotation=90)
plt.ylabel("# of images")
plt.grid(axis='y')
plt.tight_layout()