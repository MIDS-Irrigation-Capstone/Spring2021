#diff f1 f2 | grep \^\< | awk '{print $ 2}'

#First get all files from Big Earth Net
ls /hdd/BigEarthNet-v1.0 > allben

# Get all files in balanced_splits test
cat ../process_data/balanced_splits/test > test

# Append with all files in balanced_splits_expanded test
cat ../process_data/balanced_splits_expanded/test >> test

# Get the unique list of test files (there can be overlaps)
sort -u test > unique_test

# Get all files in allben that are not in unique_test and save as train
diff allben unique_test | grep \^\< | awk '{print $ 2}' > train

# Below is a check to see how many files names overlap. None should
# Also count of records in allben = train + unique_test
#TO SEE HOW MANY COMMON RECORDS. SHOULD RETURN NONE
comm -12 train unique_test
