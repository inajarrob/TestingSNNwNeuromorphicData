import sys, os, random

files = os.listdir(sys.argv[1])
print(len(files))

# how many in train dataset
train_len = round(len(files)*0.8)
print(train_len)

# split in test and train at 0.2/0.8
train = random.sample(files, k=train_len)
print(len(train))
print(len(set(train)))

#print(train)

#test = [x for x in files if x not in train]
#test = set(files).symmetric_difference(set(train))

test = []
for i in files:
    if i not in train:
        test.append(i)
print(len(test))
print(test)

with open('trials_to_train.txt', 'w') as f:
    for file in train:
        f.write("%s\n" % file)

with open('trials_to_test.txt', 'w') as f:
    for file in test:
        f.write("%s\n" % file)