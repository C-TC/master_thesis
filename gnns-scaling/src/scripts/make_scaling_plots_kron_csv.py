import matplotlib.pyplot as plt
import itertools
import csv
import sys
import numpy as np
import pandas as pd
import os
import glob

#%%
global_color_map = {	"agnn" : 	"#000000",
						"cgnn" : 	"#000099",
						"gat" : 	"#009900",
						"vanilla" : "#990000",
						"?????" : 	"#990099",
				}

arch_marker_map  = {	"DGL" : 	"o",
						"This work" : 	"x",
				}

linestyle_map  = {	"This work" : 	"-",
						"DGL" : 	"--",
				}



global_label_map = {
					"agnn" :            "AGNN",
					"cgnn" :            "C-GNN",
				    "gat" :    "GAT",	
					"vanilla" :    "Vanilla",		
					"?????" :  "..",		
				}

dgl_name_map = {"agnn" : "agnn", "cgnn" : "gcn", "vanilla" : "va", "gat" : "gat"}


nodes = (1,2,4,8,16,32,64,128,256,512)

def to_float(row):
    new_row = []
    for i in row:
        try:
            new_row.append(float(i))
        except:
            new_row.append(0)
    return new_row

def get_size(sizes_string):
    s = sizes_string[1:-1].split(",")
    s = [float(i) for i in s]
    return s[0]

def get_density(sizes_string):
    s = sizes_string[1:-1].split(",")
    s = [float(i) for i in s]
    return s[3]
    
def get_fig_size(model_n, img_shape, base_size = 1.4):
    height = 0
    width = model_n*base_size
    
    if img_shape == "square":
        height = width
    elif img_shape == "rect":
        height = width*3/4
    elif img_shape == "wide_rect":
        height = width*2.5/4
    else:
        print("img_shape MUST BE square OR rect")
    return (width, height);

#csv files:
#datetime	benchmark	framework	nodes	sizes	time
def get_data(target, models, size, density, arch = "gpu"):
    runtimes = []
    errors = []
    for n in nodes:
        files = glob.glob(f"../../data/kronecker/*{arch}*_{n}_*.csv")
        if len(files) == 0:
            break;
        with open(files[0], newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            rows = [row for row in reader if row["benchmark"] == target and get_size(row["sizes"]) == size and get_density(row["sizes"]) == density][1:10]
            times = [float(row["time"]) for row in rows]
            runtimes.append(np.median(times));
            print(n,target, times, "************************")
            errors.append(np.std(times))
    if len(runtimes) == 0:
        return 0
    return (nodes[:len(runtimes)], np.array(runtimes), np.array(errors, dtype=np.float64))
    
def get_data_single_file(model = "vanilla", density = 0.01, size = 131072, filename = "kronecker_dgl_res*4096*.csv" ):
    runtimes = []
    errors = []
    target = dgl_name_map[model];
    print("****************DGL DATA")
    for n in nodes:
        files = glob.glob(f"../../data/{filename}")
        if len(files) == 0:
            break;
        with open(files[0], newline='') as csvfile:
            reader = csv.DictReader(csvfile,  fieldnames = ["datetime","benchmark","framework","nodes","sizes","time"], delimiter=',')
            rows = [row for row in reader if row["benchmark"] == target and int(row["nodes"]) == n and get_size(row["sizes"]) == size and get_density(row["sizes"]) == density]
            times = [float(row["time"]) for row in rows]
            print(f"N: {n}, model {model}, {times}")
            runtimes.append(np.median(times));
            errors.append(np.std(times))
    return (nodes[:len(runtimes)], np.array(runtimes), np.array(errors, dtype=np.float64))
 
#%%

def make_figure_multiple_models(models = ("gat","vanilla","cgnn"), 
                                size = 262144, 
                                density = 0.01, fig_name = "all_models", img_shape = "square"):
    result_dict = {}
    result_dict["DGL"] = {target : get_data_single_file(target, density = density, size = size) for target in models}   
    result_dict["This work"] = {target : get_data(target,models,size,density) for target in models}   
   
    
    figsize = get_fig_size(len(models), img_shape, 1.4)

    fig, ax = plt.subplots(1,1, figsize = figsize)
    
    plt.subplots_adjust(left=0.16, right = 0.99, top = 0.98, bottom = 0.125)
        
    for target in models:
        for arch in result_dict:
            label = global_label_map[target] + " (" + arch + ")" 
            marker = arch_marker_map[arch]
            linestyle = linestyle_map[arch]
            color = global_color_map[target]
            current_nodes, runtimes, errors = result_dict[arch][target]
            ax.plot(current_nodes, runtimes, label = "" + label, marker = marker, fillstyle='none', color=color, linestyle = linestyle)
            ax.fill_between(current_nodes, runtimes - errors, runtimes + errors, alpha=0.2, color = color)
            print(nodes, runtimes, errors)
        
        


    ax.set_xscale('log', base = 2)

    ax.set_xticks(nodes)
    ax.set_yscale('log')
    #ax.set_ylim(bottom = 0)
    ax.set_xlim([2**(-0.5),2**7.5])


    ax.grid(True, which="both")

      
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)

    plt.xlabel("Number of GPUs", fontsize = 14, labelpad=1)
    plt.ylabel("Runtime [s]", fontsize = 14, labelpad=-2)

    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

# Put a legend to the right of the current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.41, 1),ncol = len(models), fontsize = 8.2, labelspacing=0.2, columnspacing = 0.4)
    
    savename = f"../../scaling_images/kron/all_models/{img_shape}/all_models_{size}_{density}.pdf"
    plt.savefig(savename,  bbox_inches='tight')
    print("saved image in ", savename)
#%%    
"""
def make_figure_one_model(model = "vanilla", include_compute = True,
                                size = 262144, 
                                density = 0.01,
                                img_shape = "square"):
              
    result_dict = {target : get_data(target,models,size,density) for target in (model, model + "_compute")}   
    
    figsize = get_fig_size(len(models), img_shape)

    fig, ax = plt.subplots(1,1, figsize = figsize)
    
    plt.subplots_adjust(left=0.16, right = 0.99, top = 0.98, bottom = 0.125)
        
    for i, target in enumerate(result_dict):
        label = global_label_map[model]
        if "compute" in target:
            label += " (compute only)"
        else:
            label += " (compute + communicate)"
        marker = list(global_marker_map.values())[i]
        color = list(global_color_map.values())[i]
        current_nodes, runtimes, errors = result_dict[target]
        ax.plot(current_nodes, runtimes, label = "" + label, marker = marker, fillstyle='none', color=color)
        ax.fill_between(current_nodes, runtimes - errors, runtimes + errors, alpha=0.2, color = color)
        print(nodes, runtimes, errors)
        
    ax.grid()
    ax.set_xlim(0.5, 2**7.5)

    ax.set_xlabel("Number of GPUs",fontsize = 14, labelpad=7)
    ax.set_ylabel("Runtime [s]", fontsize = 14)
    ax.set_xscale('log', base = 2)
    ax.set_xticks(nodes)
    ax.set_yscale('log')
    #ax.set_ylim(bottom = 1)
      
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

# Put a legend to the right of the current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),ncol = 2, prop={'size': 6})
    
    plt.tight_layout()

    plt.savefig(f"../../scaling_images/kron/one_model/{model}_{size}_{density}.pdf", bbox_inches='tight')
"""

#%%

models = ("gat","vanilla","cgnn")
fig_name = "all_models"
shapes = ("square","rect","wide_rect")

sizes = (131072, 262144, 1048576,2097152) 
densities= (0.01, 0.01, 0.0001, 0.0001)
for size, density in zip(sizes, densities):
    for img_shape in shapes:    
        make_figure_multiple_models(models = models, size = size, density = density, fig_name = fig_name, img_shape = img_shape)

#%%

"""
models = ("gat","vanilla","cgnn")
sizes = (131072, 262144, 1048576,2097152) 
densities= (0.01, 0.01, 0.0001, 0.0001)
for size, density in zip(sizes, densities):
        for model in models:
            make_figure_one_model(model = model, size = size, density = density)
"""