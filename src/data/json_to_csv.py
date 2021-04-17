import json
import csv
from os import listdir, walk
from os.path import isfile, join

dir_list = ['final jsons/']

for directory in dir_list:
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    print(directory)
    for i in onlyfiles:
        # print(directory+i)
        json_file = open(directory + i)
        json_data = json.load(json_file)
        csv_file = open('ground_truth.csv', 'a', newline='')
        csv_writer = csv.writer(csv_file)
        try:
            for x in json_data['outputs']['object']:
                path = i
                label = x['name']
                xmin = x['bndbox']['xmin']
                ymin = x['bndbox']['ymin']
                xmax = x['bndbox']['xmax']
                ymax = x['bndbox']['ymax']
                item = [path, label, xmin, ymin, xmax, ymax]
                csv_writer.writerow(item)
            csv_file.close()
        except Exception:
            split_path = json_data['path'].split('\\')
            path = split_path[-1]
            label = split_path[-2]
            item = [path, label]
            csv_writer.writerow(item)
            csv_file.close()
