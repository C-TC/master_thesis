import matplotlib.pyplot as plt
import itertools
import csv
import sys
import numpy as np
import pandas as pd
import os
import glob


def get_density(sizes_string):
    s = sizes_string[1:-1].split(",")
    s = [float(i) for i in s]
    if len(s) == 3:
        return 0.01 #hack to ensure compatibility with previous results
    return s[3]    
	
def get_data_many_files(node_count_list, model, density):
	data = []
	for n in node_count_list:
		data_point = {"Model" : model, "#Nodes" : n}
		searchstring = f"../data/weak_results/*{model}*_{n}_*{density}*/*{str(density)}*.csv"
		files = glob.glob(searchstring)
		data_source = "new_data"
		if len(files) == 0 and density == 0.01:
			#SEARCHING WITH OLD FOLDER FORMAT
			searchstring = f"../data/weak_results/*{model}*_{n}__gpu__weak*/*gpu*.csv"
			files = glob.glob(searchstring)
			data_source = "old_data"
		if len(files) == 0:
			#SEARCHING ON SINGLE FILE"
			searchstring = f"../data/weak_results/dace_gpu_{n}_*{density}*.csv"
			files = glob.glob(searchstring)
			data_source = "single_file"
		if len(files) == 0:
			print(f"MISSING DATA: N: {n}, model {model},")
			continue

		with open(files[0], newline='') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',')
			if data_source == "old_data":
				rows = [row for row in reader][1:10]
			else:
				rows = [row for row in reader if row["benchmark"] == model][1:]

			vertices = [row["sizes"] for row in rows]
			times = [float(row["time"]) for row in rows]
			if len(times) == 0:
				print(f"MISSING DATA: N: {n}, model {model},")
				continue

			data_point["Time mean [s]"] = sum(times) / len(times)
			data_point["Time std [s]"] = np.std(times)
			data.append(data_point)
			
	return data


def get_data_single_file(node_count_list, model, density, filename):
	data = []
	dgl_name_map = {"a_gnn" : "agnn", "c_gnn" : "gcn", "vanilla" : "va", "gat" : "gat"}
	target = dgl_name_map[model.lower()];
	for n in node_count_list:
		data_point = {"Model" : model, "#Nodes" : n}
		files = glob.glob(f"../data/{filename}")
		if len(files) == 0:
			break;
		with open(files[0], newline='') as csvfile:
			reader = csv.DictReader(csvfile,  fieldnames = ["datetime","benchmark","framework","nodes","sizes","time"], delimiter=',')
			rows = [row for row in reader if row["benchmark"] == target and int(row["nodes"]) == n and get_density(row["sizes"]) == density]
			times = [float(row["time"]) for row in rows]
			if len(times) > 0:
				data_point["Time mean [s]"] = sum(times) / len(times)
				data_point["Time std [s]"] = np.std(times)
				data.append(data_point)
	return data
