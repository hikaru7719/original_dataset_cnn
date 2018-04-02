import argparse
import os

parser = argparse.ArgumentParser(description = "description goes here")
parser.add_argument("-p", type=str, help ="Write Folder Path.", required=True)
parser.add_argument("-l", type= str,help="Write Label",required=True)

args = parser.parse_args()
folder_path = args.p
label = args.l
folder_path_name = ""
folder_list = []
label_name = ""
label_list =[]
for string1 in folder_path:
    if string1 == ",":
        folder_list.append(folder_path_name)
        folder_path_name = ""
    else:
        folder_path_name += string1

for string2 in label:
    if string2 == ",":
        label_list.append(label_name)
        label_name = ""
    else:
        label_name += string2

print(folder_list,label_list)
csv_file_name = "train.csv"
csv_file = open(csv_file_name,'w')
file_list = []
for (dir_name,label) in zip(folder_list,label_list):
    file_list = os.listdir(dir_name)
    for file_name in file_list:
        file_path = dir_name + "/" + file_name
        row = file_path + "," + label +"\n"
        csv_file.write(row)
csv_file.close()
