import csv
import math as m
import matplotlib.pyplot as plt

def get_col_mar_lst_mfc(algo, model):
	col = "#000000"
	mar = "*"
	lst = "-"
	mfc = "none"

	# Color, Marker
	if model == "GAT":
		col = "#009900"
		mar = "o"
	if model == "VA":
		col = "#990000"
		mar = "d"
	if model == "AGNN":
		col = "#000099"
		mar = "s"
	if model == "CGNN":
		col = "#990099"
		mar = "^"

	# Line-style
	if algo == "This Work":
		lst = "-"
	if algo == "DGL":
		lst = "none"
		mfc = col
	if algo == "DistDGL no sampling":
		lst = ":"
	if algo == "DistDGL sampling":
		lst = "--"

	return (col, mar, lst, mfc)


def to_int(x):
	try:
		return int(x)
	except:
		return float("NaN")

def to_float(x):
	try:
		return float(x)
	except:
		return float("NaN")

def read_baseline_data(filename):
	data = []	
	with open("results/" + filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		header = next(reader)
		for row in reader:
			data_point = {}
			data_point["model"] = row[header.index("model")]
			data_point["nodes"] = to_int(row[header.index("nodes")])
			data_point["vertices"] = to_int(row[header.index("vertices")])
			data_point["density"] = to_float(row[header.index("density")])
			data_point["features"] = to_int(row[header.index("features")])
			data_point["runtime"] = to_float(row[header.index("runtime")])
			data_point["algorithm"] = row[header.index("algorithm")]
			# Compute number of edges
			n = data_point["vertices"]
			d = data_point["density"]
			max_e = n * (n-1) / 2
			data_point["edges"] = d * max_e
			data.append(data_point)
	return data
	
def read_thiswork_data(filename):
	data = []	
	with open("results/" + filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		header = next(reader)
		for row in reader:
			data_point = {}
			data_point["model"] = row[header.index("model")]
			data_point["nodes"] = to_int(row[header.index("nodes")])
			data_point["vertices"] = to_int(row[header.index("vertices")])
			data_point["edges"] = to_int(row[header.index("edges")])
			data_point["features"] = to_int(row[header.index("features")])
			data_point["runtime"] = to_float(row[header.index("runtime")]) / 1e6 # us to s
			# Compute density
			n = data_point["vertices"]
			e = data_point["edges"]
			max_e = n * (n-1) / 2
			data_point["density"] = to_float(e / max_e)
			data.append(data_point)
	return data

def create_kronecker_strong_scaling_plots():
	# Read data
	data = {}
	data_baselines = read_baseline_data("new_kronecker_strong_k16_training_bs16384.csv")
	data["DistDGL no sampling"] = [x for x in data_baselines if x["algorithm"] == "DIST"]
	data["DistDGL sampling"] = [x for x in data_baselines if x["algorithm"] == "DIST_SAMPLE"]
	data["DGL"] = [x for x in data_baselines if x["algorithm"] == "NORM"]
	data["This Work"] = read_thiswork_data("new_kronecker_strong_this_work.csv")

	# Extract info from data
	models = list(set(x["model"] for algo in data for x in data[algo]))
	node_counts = sorted(list(set(x["nodes"] for algo in data for x in data[algo])))
	vertex_counts = sorted(list(set(x["vertices"] for algo in data for x in data[algo])))
	feature_counts = sorted(list(set(x["features"] for algo in data for x in data[algo])))

	# Create one plot per vertex-count x node-count combination
	for vertices in vertex_counts:
		for features in feature_counts:
			data_filtered = {}
			for algo in data:
				data_filtered[algo] = [x for x in data[algo] if x["vertices"] == vertices and x["features"] == features]
			# Create Plot
			(fig, ax) = plt.subplots(1, 1, figsize = (3,2.5))
			plt.subplots_adjust(left=0.225, right = 0.975, top = 0.95, bottom = 0.18)
			# One data-series per model-algorithm combination
			for algo in data_filtered:
				for model in models:
					dat = [x for x in data_filtered[algo] if x["model"] == model]
					dat = sorted(dat, key = lambda x : x["nodes"])
					xvals = [x["nodes"] for x in dat]
					yvals = [x["runtime"] for x in dat]
					idx = 0	
					while idx < len(xvals) - 1:
						if xvals[idx] == xvals[idx+1]:
							print("WARNING: Duplicated measurement detected -> Only considering the first entry")
							del xvals[idx+1]
							del yvals[idx+1]
						idx += 1
			
					(col, mar, lst, mfc) = get_col_mar_lst_mfc(algo, model.upper())
					lab = model.upper() + " - " + algo
					ax.plot(xvals, yvals, label = lab, color = col, marker = mar, linestyle = lst, markersize = 4, markerfacecolor = mfc, linewidth = 1)
			# Configure plot
			ax.grid(which = "major")
			ax.grid(which = "minor", linewidth = 0.5)
			# Configure x-axis
			ax.set_xscale("log", base = 2)
			ax.set_xlabel("Compute Nodes")
			ax.set_xticks([2**x for x in range(int(m.log2(min(node_counts))), int(m.log2(max(node_counts)))+1, 1)])
			# Configure y-axis
			ax.set_yscale("log", base = 10)
			ax.set_ylabel("Runtime [s]")
			# Save Plot
			plt.savefig("plots_new/kronecker_strong_scaling_n=%d_k=%d.pdf" % (vertices, features))

def create_kronecker_weak_scaling_plots():
	# Read data
	data = {}
	data_baselines = read_baseline_data("new_kronecker_strong_k16_training_bs16384.csv")
	data["DistDGL no sampling"] = [x for x in data_baselines if x["algorithm"] == "DIST"]
	data["DistDGL sampling"] = [x for x in data_baselines if x["algorithm"] == "DIST_SAMPLE"]
	data["DGL"] = [x for x in data_baselines if x["algorithm"] == "NORM"]
	data["This Work"] = read_thiswork_data("new_kronecker_strong_this_work.csv")

	# Extract info from data
	models = list(set(x["model"] for algo in data for x in data[algo]))
	node_counts = sorted(list(set(x["nodes"] for algo in data for x in data[algo])))
	densities= sorted(list(set(x["density"] for algo in data for x in data[algo])))
	feature_counts = sorted(list(set(x["features"] for algo in data for x in data[algo])))

	# Create one plot per vertex-count x node-count combination
	for density in densities:
		for features in feature_counts:
			data_filtered = {}
			for algo in data:
				data_filtered[algo] = [x for x in data[algo] if x["density"] == density and x["features"] == features]
			# Create Plot
			(fig, ax) = plt.subplots(1, 1, figsize = (3,2.5))
			plt.subplots_adjust(left=0.225, right = 0.975, top = 0.95, bottom = 0.18)
			# One data-series per model-algorithm combination
			for algo in data_filtered:
				for model in models:
					dat = [x for x in data_filtered[algo] if x["model"] == model]
					dat = sorted(dat, key = lambda x : x["nodes"])
					xvals = [x["nodes"] for x in dat]
					yvals = [x["runtime"] for x in dat]
					idx = 0	
					while idx < len(xvals) - 1:
						if xvals[idx] == xvals[idx+1]:
							print("WARNING: Duplicated measurement detected -> Only considering the first entry")
							del xvals[idx+1]
							del yvals[idx+1]
						idx += 1
						
					if len(xvals) > 0:
						(col, mar, lst, mfc) = get_col_mar_lst_mfc(algo, model.upper())
						lab = model.upper() + " - " + algo
						ax.plot(xvals, yvals, label = lab, color = col, marker = mar, linestyle = lst, markersize = 4, markerfacecolor = mfc, linewidth = 1)
			# Configure plot
			ax.grid(which = "major")
			ax.grid(which = "minor", linewidth = 0.5)
			# Configure x-axis
			ax.set_xscale("log", base = 2)
			ax.set_xlabel("Compute Nodes")
			ax.set_xticks([2**x for x in range(int(m.log2(min(node_counts))), int(m.log2(max(node_counts)))+1, 1)])
			# Configure y-axis
			ax.set_yscale("log", base = 10)
			ax.set_ylabel("Runtime [s]")
			# Save Plot
			plt.savefig("plots_new/kronecker_weak_scaling_d=%.4f_k=%d.pdf" % (density, features))

create_kronecker_strong_scaling_plots()
create_kronecker_weak_scaling_plots()


