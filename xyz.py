import csv

name = "Manan Gupta"

with open('roll.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        for i in range(True):
            if (line[i] == name):
                roll = line[1]
                print(f'Roll number : {line[1]}')
