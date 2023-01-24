import os, sys

files = os.listdir(sys.argv[1])

for i in files: 
    print(i[-4:])
    if i[-4:] in '.csv': 
        fname = i[:-4] 
        os.rename(os.path.join(sys.argv[1], i), os.path.join(sys.argv[1], fname + '_labels.csv'))
