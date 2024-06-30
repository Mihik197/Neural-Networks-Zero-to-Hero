import random

# Read names from names.txt
with open('names.txt', 'r') as f:
    names = f.read().splitlines()

# Shuffle names randomly
random.shuffle(names)

# Calculate split points
num_samples = len(names)
train_split = int(0.8 * num_samples)
dev_split = int(0.1 * num_samples)

# Split into train, dev, and test sets
train_names = names[:train_split]
dev_names = names[train_split:train_split + dev_split]
test_names = names[train_split + dev_split:]

# Write train set to train.txt
with open('train.txt', 'w') as f:
    for name in train_names:
        f.write(name + '\n')

# Write dev set to dev.txt
with open('dev.txt', 'w') as f:
    for name in dev_names:
        f.write(name + '\n')

# Write test set to test.txt
with open('test.txt', 'w') as f:
    for name in test_names:
        f.write(name + '\n')

print(f'Dataset split into:\nTrain set: {len(train_names)} names\nDev set: {len(dev_names)} names\nTest set: {len(test_names)} names')
