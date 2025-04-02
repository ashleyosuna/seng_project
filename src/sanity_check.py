filenames = ['processed_3000.csv',
             'processed.csv',
             'processed3.csv']

score_sum = 0
num_samples = 0

for filename in filenames:
    f = open('data/' + filename, "r")
    for line in f:
        score_sum += float(line.split(",")[-1])
        num_samples += 1
    f.close()

sample_mean = score_sum / num_samples
print(sample_mean)