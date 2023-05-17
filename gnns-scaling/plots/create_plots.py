import csv
import pylab
import math as m
import matplotlib.pyplot as plt

import additional_data_loader as adl

legendfontsize = 6
max_lines_in_global_legend = 0
max_lines_in_global_legend_reduced = 0


def get_lab_col_mar_ls(algo, model):
	algo_map = {"This Work" : "This Work", 
				"Norm" : "DGL", 
				"Dist" : "DistDGL no sampling",
				"DistDGL" : "DistDGL no sampling",					
				"Dist Sample" : "DistDGL sampling",
				"Training" : "Training",
				"Inference" : "Inference",
			   }
	lab = "%s - %s" % (model, algo_map[algo])
	col = "#000000"
	mar = "*"
	ls = "-"
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
		ls = "-"
	if algo in ["DGL","Norm"]:
		ls = "none"
		mfc = col
	if algo in ["DistDGL","Dist","Inference"]:
		ls = ":"
	if algo in ["Dist Sample"]:
		ls = "--"

	return (lab, col, mar, ls, mfc)

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

data_type_map = {
	# Field present in both our data and baseline data
	"Model":lambda x : str(x).upper(),
	"Setting":str,
	"#Nodes":to_int,
	"#Vertices":to_int,
	"Density":to_float,
	"#Features":to_int,

	# Fields only present in our data
	"Experiment ID":to_int,
	"Job ID":to_int,
	"#Repetitions":to_int,
	"Dataset":str,
	"Datatyp":str,
	"Scale":to_int,
	"Task":str,
	"#Edges":to_int,
	"#Layers":to_int,
	"Elapsed":str,
	"Node-Hours":str,
	"Done":str,
	"Time mean [ms]": to_float,
	"Time std [ms]": to_float,

	# Fields only present in baseline data
	"Timestamp" : str,
	"Algorithm" : str,
	"#Features_2" : to_int,	
	"Time mean [s]" : to_float,
	"What is this?" : to_int,	

	# Misc
	"nodes" : to_int,
	"model" : str,
	"vertices" : to_int,
	"features" : to_int,
	"time" : to_float,
	"exp_type" : str,
	"density" : to_float,
	}

def read_data(filename, fix_density):
	data = []	
	with open("results/" + filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		header = next(reader)
		for row in reader:
			data_point = {}
			for field_name in header:
				field_data = row[header.index(field_name)]
				field_data = field_data[:-1] if fix_density and field_name == "Density" else field_data 
				field_data = data_type_map[field_name](field_data)
				field_data = field_data / 100 if fix_density and field_name == "Density" else field_data 
				data_point[field_name] = field_data
			data.append(data_point)
	return data



def create_real_world_strong_scaling_plot(filename, features):
	# Configuration of plot
	node_count_list = [16,64,256,1024]

	# Read data
	data = read_data(filename, True)
	
	# Filter data
	data = [x for x in data if x["#Features"] == features]

	# Create Plot
	(fig, ax) = plt.subplots(1, 1, figsize = (3,3))
	plt.subplots_adjust(left=0.22, right = 0.97, top = 0.97, bottom = 0.15)
	
	# Iterate through models (data series)
	for (i, model) in enumerate(["GAT","VA","AGNN"]):
		# Iterate through tasks
		for (j, task) in enumerate(["Training", "Inference"]):
			xvals = []
			yvals = []
			# Iterate through node-counts
			for (k, nodes) in enumerate(node_count_list):
				times = [x["Time mean [ms]"] * 1e-3 for x in data if x["Model"] == model and x["Task"] == task and x["#Nodes"] == nodes]
				tmp = [x for x in data if x["Model"] == model and x["Task"] == task and x["#Nodes"] == nodes]
				# If we have multiple data points for the same configuration, compute the average
				if len(times) > 0:
					time = sum(times) / len(times)
					if len(times) > 1:
						print("Aggregating multiple measurements")
						[print(x) for x in tmp]
				else:
					time = float("NaN")
				# Store time
				if not m.isnan(time):
					xvals.append(nodes)
					yvals.append(time)
			# Plot data
			if len(xvals) > 0:
				(lab, col, mar, ls, mfc) = get_lab_col_mar_ls(task, model)
				ax.plot(xvals, yvals, label = lab,color = col,marker = mar, linestyle = ls,markersize = 4,markerfacecolor = mfc,linewidth = 1,)
	# Configure plot
	ax.grid(which = "major")
	ax.grid(which = "minor", linewidth = 0.5)
	ax.set_ylim(1,1000)
	ax.legend(fontsize = legendfontsize, ncol = 2, loc = "lower center")
	# Configure x-axis
	ax.set_xscale("log", base = 2)
	ax.set_xlabel("Compute Nodes")
	ax.set_xticks(node_count_list)
	ax.set_xticklabels(node_count_list)
	# Configure y-axis
	ax.set_yscale("log", base = 10)
	ax.set_ylabel("Runtime [s]")
	# Save Plot
	plt.savefig("plots/" + filename[:-4] + "_" + str(features) +".pdf")

def create_random_weak_scaling_plot(density):
	# Configuration of plot
	node_count_list = [1,2,4,8,16,32,64,128,256,512]

	# Create Plot
	(fig, ax) = plt.subplots(1, 1, figsize = (3,3))
	plt.subplots_adjust(left=0.22, right = 0.975, top = 0.975, bottom = 0.15)
	
	# Iterate through models (data series)
	for (i, model) in enumerate(["GAT","VA","CGNN"]):
		# Iterate through algorithms
		for (j, algo) in enumerate(["This Work", "DistDGL"]):
			xvals = []
			yvals = []
			# Iterate through node- and vertex- counts (x-axis)
			for (k, nodes)in enumerate(node_count_list):
				# Read data
				model_ = model.lower().replace("va","vanilla").replace("cgnn","c_gnn")
				filename = "uniform_dgl_res_bs8192_rerun.csv"
				if algo == "This Work":
					data = adl.get_data_many_files(node_count_list, model_, density)
				else:
					data = adl.get_data_single_file(node_count_list, model_, density, filename)
				times = [x["Time mean [s]"] for x in data if x["#Nodes"] == nodes]
				tmp = [x for x in data if x["#Nodes"] == nodes]

				# If we have multiple data points for the same configuration, compute the average
				if len(times) > 0:
					time = sum(times) / len(times)
					if len(times) > 1:
						print("Aggregating multiple measurements")
						[print(x) for x in tmp]
				else:
					time = float("NaN")
				# Store time
				if not m.isnan(time):
					xvals.append(nodes)
					yvals.append(time)
			# Plot data
			if len(xvals) > 0:
				(lab, col, mar, ls, mfc) = get_lab_col_mar_ls(algo, model)
				ax.plot(xvals, yvals, label = lab, color = col,marker = mar, linestyle = ls,markersize = 5,markerfacecolor = mfc,linewidth = 1)
	# Configure plot
	ax.grid(which = "major")
	ax.grid(which = "minor", linewidth = 0.5)
	ax.set_ylim(0.01,100)
	ax.legend(fontsize = legendfontsize, ncol = 1, loc = "lower right") #TODO Replace with labels
	# Configure x-axis
	ax.set_xscale("log", base = 2)
	ax.set_xlabel("Compute Nodes")
	ax.set_xticks(node_count_list)
	# Configure y-axis
	ax.set_yscale("log", base = 10)
	ax.set_ylabel("Runtime [s]")
	# Save Plot
	plt.savefig("plots/random_weak_scaling_%g.pdf" % density)


def create_kronecker_strong_scaling_plot(vertices, features):
	# Configuration of plot
	node_count_list = [1,4,16,64,256]

	# Read data
	data_all = read_data("kronecker_strong_scaling.csv", True)
	data = [x for x in data_all if x["features"] == features and x["vertices"] == vertices]

	mem_lines = []
	mem_labels = []

	# Create Plot
	(fig, ax) = plt.subplots(1, 1, figsize = (3,2.5))
	plt.subplots_adjust(left=0.225, right = 0.975, top = 0.95, bottom = 0.18)
	
	# Iterate through models (data series)
	for (i, model) in enumerate(["GAT","VA","AGNN"]):
		# Iterate algorithms (data series)
		for algo in ["This Work", "Dist", "Norm", "Dist Sample"]:
			xvals = []
			yvals = []
			# Iterate through node-counts
			for (k, nodes) in enumerate(node_count_list):
				exp_type = model + "_" + algo.upper().replace(" ","_")
				times = [x["time"] for x in data if x["nodes"] == nodes and x["exp_type"] == exp_type]
				tmp = [x for x in data if x["nodes"] == nodes and x["exp_type"] == exp_type]
				if len(times) > 0:
					xvals.append(nodes)
					yvals.append(sum(times) / len(times))
					if len(times) > 1:
						print("Aggregating multiple measurements")
						[print(x) for x in tmp]
			# Plot data
			if len(xvals) > 0:
				(lab, col, mar, ls, mfc) = get_lab_col_mar_ls(algo, model)
				mem_lines += ax.plot(xvals, yvals, label = lab, color = col, marker = mar, linestyle = ls,markersize = 4,markerfacecolor = mfc,linewidth = 1,)
				mem_labels.append(lab)	
	# Configure plot
	ax.grid(which = "major")
	ax.grid(which = "minor", linewidth = 0.5)
	# Configure x-axis
	ax.set_xscale("log", base = 2)
	ax.set_xlabel("Compute Nodes")
	ax.set_xticks(node_count_list)
	ax.set_xticklabels(node_count_list)
	# Configure y-axis
	ax.set_ylim(0.01,1000)
	ax.set_yscale("log", base = 10)
	ax.set_ylabel("Runtime [s]")
	# Save Plot
	plt.savefig("plots/kronecker_strong_scaling_%d_%d.pdf" % (vertices, features))

	global max_lines_in_global_legend
	if len(mem_lines) > max_lines_in_global_legend:
		max_lines_in_global_legend = len(mem_lines)
		figlegend = pylab.figure(figsize=(13,0.35))
		figlegend.legend(mem_lines, mem_labels, 'center', ncol = 6, labelspacing = 0.1, handletextpad = 0.1, fontsize = 9.25,frameon=False)
		figlegend.savefig("plots/global_legend.pdf")


def create_kronecker_strong_scaling_reduced_plot(vertices, features):
	# Configuration of plot
	node_count_list = [1,4,16,64,256]

	# Read data
	data_all = read_data("kronecker_strong_scaling.csv", True)
	data = [x for x in data_all if x["features"] == features and x["vertices"] == vertices]

	mem_lines = []
	mem_labels = []


	# Create Plot
	(fig, ax) = plt.subplots(1, 1, figsize = (3,2.5))
	plt.subplots_adjust(left=0.225, right = 0.975, top = 0.95, bottom = 0.18)

	
	# Iterate through models (data series)
	for (i, model) in enumerate(["GAT","VA","AGNN"]):
		# Iterate algorithms (data series)
		for algo in ["This Work", "Dist Sample", "Norm"]:
			xvals = []
			yvals = []
			# Iterate through node-counts
			for (k, nodes) in enumerate(node_count_list):
				exp_type = model + "_" + algo.upper().replace(" ","_")
				times = [x["time"] for x in data if x["nodes"] == nodes and x["exp_type"] == exp_type]
				tmp = [x for x in data if x["nodes"] == nodes and x["exp_type"] == exp_type]
				if len(times) > 0:
					xvals.append(nodes)
					yvals.append(sum(times) / len(times))
					if len(times) > 1:
						print("Aggregating multiple measurements")
						[print(x) for x in tmp]
			# Plot data
			if len(xvals) > 0:
				(lab, col, mar, ls, mfc) = get_lab_col_mar_ls(algo, model)
				lab = lab.replace("DistDGL sampling", "DistDGL")
				mem_lines += ax.plot(xvals, yvals, label = lab, color = col, marker = mar, linestyle = ls,markersize = 4,markerfacecolor = mfc,linewidth = 1,)
				mem_labels.append(lab)
	# Configure plot
	ax.grid(which = "major")
	ax.grid(which = "minor", linewidth = 0.5)
	if False:
		ax.legend(	fontsize = legendfontsize,
					ncol=1,
					labelspacing=0.2,
					columnspacing=0.4,) #TODO Replace with labels
	# Configure x-axis
	ax.set_xscale("log", base = 2)
	ax.set_xlabel("Compute Nodes")
	ax.set_xticks(node_count_list)
	ax.set_xticklabels(node_count_list)
	# Configure y-axis
	ax.set_ylim(0.01,1000)
	ax.set_yscale("log", base = 10)
	ax.set_ylabel("Runtime [s]")
	# Save Plot
	plt.savefig("plots/kronecker_strong_scaling_reduced_%d_%d.pdf" % (vertices, features))

	global max_lines_in_global_legend_reduced
	if len(mem_lines) > max_lines_in_global_legend_reduced:
		max_lines_in_global_legend_reduced = len(mem_lines)
		figlegend = pylab.figure(figsize=(13,0.2))
		figlegend.legend(mem_lines, mem_labels, 'center', ncol = 9, labelspacing = 0.1, handletextpad = 0.1, fontsize = 9.25,frameon=False)
		figlegend.savefig("plots/global_legend_reduced.pdf")


def create_kronecker_weak_scaling_plot(density, features):
	# Configuration of plot
	node_count_list = [1,4,16]						#TODO: Update
	vertex_count_list = [131072,262144,524288]			#TODO: Update
	vertex_count_labels = ["131k","262k","524k"]

	# Read data
	data_all = read_data("kronecker_weak_scaling.csv", True)
	data = [x for x in data_all if x["features"] == features and x["density"] == density]

	mem_lines = []
	mem_labels = []

	# Create Plot
	(fig, ax) = plt.subplots(1, 1, figsize = (3,2.5))
	plt.subplots_adjust(left=0.225, right = 0.95, top = 0.95, bottom = 0.18)
	
	# Iterate through models (data series)
	for (i, model) in enumerate(["GAT","VA","AGNN"]):
		# Iterate algorithms (data series)
		for algo in ["This Work", "Norm", "Dist Sample"]:
			xvals = []
			yvals = []
			# Iterate through node-counts and vertex-counts
			for (k, (nodes, vertices)) in enumerate(zip(node_count_list, vertex_count_list)):
				exp_type = model + "_" + algo.upper().replace(" ","_")
				times = [x["time"] for x in data if x["nodes"] == nodes and x["vertices"] == vertices and x["exp_type"] == exp_type]
				tmp = [x for x in data if x["nodes"] == nodes and x["vertices"] == vertices and x["exp_type"] == exp_type]
				if len(times) > 0:
					xvals.append(nodes)
					yvals.append(sum(times) / len(times))
					if len(times) > 1:
						print("Aggregating multiple measurements")
						[print(x) for x in tmp]
			# Plot data
			if len(xvals) > 0:
				(lab, col, mar, ls, mfc) = get_lab_col_mar_ls(algo, model)
				lab = lab.replace(" sampling","")
				mem_lines += ax.plot(xvals, yvals, label = lab, color = col, marker = mar, linestyle = ls,markersize = 4,markerfacecolor = mfc,linewidth = 1,)
				mem_labels.append(lab)	
	# Configure plot
	ax.grid(which = "major")
	ax.grid(which = "minor", linewidth = 0.5)
	# Configure x-axis
	ax.set_xscale("log", base = 2)
	ax.set_xlabel("Compute Nodes | Vertices")
	ax.set_xticks(node_count_list)
	ax.set_xticklabels([str(node_count_list[i]) + " | " + vertex_count_labels[i] for i in range(len(node_count_list))])
	ax.set_xlim(0.9,1.15*16)
	# Configure y-axis
	ax.set_ylim(0.1,100)
	#ax.set_ylim(0.1,10)
	ax.set_yscale("log", base = 10)
	ax.set_ylabel("Runtime [s]")
	# Save Plot
	plt.savefig("plots/kronecker_weak_scaling_%g_%d.pdf" % (density, features))

	global max_lines_in_global_legend
	if len(mem_lines) > max_lines_in_global_legend:
		max_lines_in_global_legend = len(mem_lines)
		figlegend = pylab.figure(figsize=(5,0.35))
		figlegend.legend(mem_lines, mem_labels, 'center', ncol = 3, labelspacing = 0.1, handletextpad = 0.1, fontsize = 9.25,frameon=False)
		figlegend.savefig("plots/global_legend_weak_scaling.pdf")

# Real World Graphs - Strong Scaling
create_real_world_strong_scaling_plot("real_world_strong_scaling.csv", 16)
create_real_world_strong_scaling_plot("real_world_strong_scaling.csv", 32)
create_real_world_strong_scaling_plot("real_world_strong_scaling.csv", 64)
create_real_world_strong_scaling_plot("real_world_strong_scaling.csv", 128)

# Random Graphs - Weak Scaling
create_random_weak_scaling_plot(0.01)
create_random_weak_scaling_plot(0.001)
create_random_weak_scaling_plot(0.0001)

# Kronecker Graphs - Strong Scaling
for vertices in [131072,262144,1048576,2097152]:
	for features in [16,128]:
		create_kronecker_strong_scaling_plot(vertices,features)
		create_kronecker_strong_scaling_reduced_plot(vertices,features)

# Kronecker Graphs - Weak Scaling
for density in [0.01,0.001,0.0001]:
	for features in [16,128]:
		create_kronecker_weak_scaling_plot(density,features)
		#create_kronecker_strong_scaling_reduced_plot(vertices,features)

