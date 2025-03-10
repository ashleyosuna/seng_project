import csv

def write_to_csv(rows, file):
    with open(file, 'w') as f:
     
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerows(rows)

def read_csv():
    with open('data.csv', newline='') as f:
        reader = csv.reader(f)
        samples = []
        labels = []
        for row in reader:
            samples.append(row[0])
            labels.append(row[1])
        return samples, labels