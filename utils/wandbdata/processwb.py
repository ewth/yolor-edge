import csv
with open('csvs/wandb_export_2021-12-13T09 19 07.858+08 00.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimeter=',')
    headers = []
    for row in csv_reader:
        i
