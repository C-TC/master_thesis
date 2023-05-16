import matplotlib.pyplot as plt
import itertools
import csv
import sys
import numpy as np
import pandas as pd
import os
import glob

#%%
global_color_map = {	"a_gnn" : 	"#000000",
						"c_gnn" : 	"#000099",
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


global_marker_map = {	"a_gnn" : 	"o",
						"c_gnn" : 	"x",
						"gat" : 	"^",
						"vanilla" : 	"*",
						"?????" : 	"d",
				}

dgl_name_map = {"a_gnn" : "agnn", "c_gnn" : "gcn", "vanilla" : "va", "gat" : "gat"}


global_label_map = {
					"a_gnn" :            "AGNN",
					"c_gnn" :            "C-GNN",
				    "gat" :    "GAT",	
					"vanilla" :    "Vanilla",		
					"?????" :  "..",		
				}


nodes = (1,2,4,8,16,32,64,128,256,512)

def to_float(row):
    new_row = []
    for i in row:
        try:
            new_row.append(float(i))
        except:
            new_row.append(0)
    return new_row


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

def get_size(sizes_string):
    s = sizes_string[1:-1].split(",")
    s = [float(i) for i in s]
    return s[0]

def get_density(sizes_string):
    s = sizes_string[1:-1].split(",")
    s = [float(i) for i in s]
    if len(s) == 3:
        return 0.01 #hack to ensure compatibility with previous results
    return s[3]
    


#csv files:
#datetime	benchmark	framework	nodes	sizes	time
def get_data_many_files(model = "vanilla", density = 0.01, compute = False):
    runtimes = []
    errors = []
    print("****************OUR DATA")


    for n in nodes:
        
        searchstring = f"../../data/weak_results/*{model}*_{n}_*{density}*/*{str(density)}*.csv"
        files = glob.glob(searchstring)
        data_source = "new_data"

        
        if len(files) == 0 and density == 0.01:
            print("SEARCHING WITH OLD FOLDER FORMAT")
            searchstring = f"../../data/weak_results/*{model}*_{n}__gpu__weak*/*gpu*.csv"
            files = glob.glob(searchstring)
            data_source = "old_data"

        
        if len(files) == 0:
            print("SEARCHING ON SINGLE FILE")
            searchstring = f"../../data/weak_results/dace_gpu_{n}_*{density}*.csv"
            files = glob.glob(searchstring)
            data_source = "single_file"

        
        if len(files) == 0:
            print(f"MISSING DATA: N: {n}, model {model},")
            runtimes.append(np.nan)
            errors.append(np.nan)
            continue

            

        target = model;
        if compute:
            target += "_compute"
        with open(files[0], newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            if data_source == "old_data":
                rows = [row for row in reader][1:10]
            else:
                rows = [row for row in reader if row["benchmark"] == target][1:]

            vertices = [row["sizes"] for row in rows]
            times = [float(row["time"]) for row in rows]
            if len(times) == 0:
                print(f"MISSING DATA: N: {n}, model {model},")
                runtimes.append(np.nan)
                errors.append(np.nan)
                continue
            
            print(f"N: {n}, model {model}, {times[:2]}")

            runtimes.append(np.median(times));
            errors.append(np.std(times))
            
    return (nodes[:len(runtimes)], np.array(runtimes), np.array(errors, dtype=np.float64))



def get_data_single_file(model = "vanilla", density = 0.01, filename = "uniform_dgl_res_bs8192_rerun_no_out.csv" ):
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
            rows = [row for row in reader if row["benchmark"] == target and int(row["nodes"]) == n and get_density(row["sizes"]) == density]
            times = [float(row["time"]) for row in rows]
            print(f"N: {n}, model {model}, {times}")
            runtimes.append(np.median(times));
            errors.append(np.std(times))
    return (nodes[:len(runtimes)], np.array(runtimes), np.array(errors, dtype=np.float64))

#%%

def make_figure_multiple_models(models = ("gat","vanilla","c_gnn"), 
                                density = 0.01, arch = "gpu", fig_name = "all_models", img_shape = "wide_rect"):
             
    result_dict = {}
    result_dict["DGL"] = {target : get_data_single_file(target, density = density) for target in models}   
    result_dict["This work"] = {target : get_data_many_files(target,density = density) for target in models}       
    
    figsize = get_fig_size(len(models), img_shape)
        
    fig, ax = plt.subplots(1,1, figsize = figsize)
    
    plt.subplots_adjust(left=0.16, right = 0.99, top = 0.98, bottom = 0.125)
    
    for target in models:
        for arch in result_dict:
            label = global_label_map[target] + " (" + arch + ")"
            marker = arch_marker_map[arch]
            color = global_color_map[target]
            style = linestyle_map[arch]
            current_nodes, runtimes, errors = result_dict[arch][target]
            ax.plot(current_nodes, runtimes, label = "" + label, marker = marker, fillstyle='none', color=color, ls = style)
            #ax.fill_between(current_nodes, runtimes - errors, runtimes + errors, alpha=0.2, color = color)
            if (len(runtimes) < len(nodes)):
                print(f"missing data for {target} {arch}, ", nodes[len(runtimes)-1:])
        
    ax.set_xscale('log', base = 2)

    ax.set_xticks(nodes)
    ax.set_yscale('log')
    #ax.set_ylim(bottom = 0)
    ax.set_xlim([2**(-0.5),2**9.5])


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

    plt.savefig(f"../../scaling_images/uniform/multiple_models/{img_shape}/weak_{fig_name}_d{density}_{img_shape}.pdf",bbox_inches='tight')

#%%
"""
def make_figure_one_model(model = "vanilla", density = 0.01, arch = "gpu", fig_name = "one_model", img_shape = "square"):
             
    result_dict = {}
    result_dict["DGL"] = get_data_single_file(model, density = density)  
    result_dict["This work"] = get_data_many_files(model,density = density)       

    
    figsize = get_fig_size(4, img_shape)
        
    fig, ax = plt.subplots(1,1, figsize = figsize)
    
    plt.subplots_adjust(right = 0.99, top = 0.98, bottom = 0.125)
    
    for arch in result_dict:
            label = global_label_map[model] + " (" + arch + ")"
            marker = arch_marker_map[arch]
            color = global_color_map[model]
            style = linestyle_map[arch]
            current_nodes, runtimes, errors = result_dict[arch]
            if len(runtimes) < 4:
                continue
            ax.plot(current_nodes, runtimes, label = "" + label, marker = marker, fillstyle='none', color=color, ls = style)
            ax.fill_between(current_nodes, runtimes - errors, runtimes + errors, alpha=0.2, color = color)
            if (len(runtimes) < len(nodes)):
                print(f"missing data for {model} {arch}, ", nodes[len(runtimes)-1:])
        
    ax.set_xlabel("Number of GPUs",fontsize = 14)
    ax.set_ylabel("Runtime [s]", fontsize = 14)
    ax.set_xscale('log', base = 2)
    ax.set_xticks(nodes)
    ax.set_yscale('log')
    #ax.set_ylim(bottom = 1)
    ax.grid(True, which="both")

      
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

# Put a legend to the right of the current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),ncol = 2, fontsize = 11)
    plt.subplots_adjust(right = 0.99, top = 0.98, bottom = 0.125)

    plt.savefig(f"../../scaling_images/uniform/one_model/{img_shape}/weak_{model}_{density}.pdf", bbox_inches='tight')
"""
#%%

shapes = ("square","rect")
models = ("gat","vanilla","c_gnn")
for density in (0.01,0.001,0.0001,):
    for img_shape in shapes:
        make_figure_multiple_models(models = models, density = density, img_shape = img_shape, fig_name = "no_agnn")


#%%

models = ("gat","vanilla","c_gnn","a_gnn")
for density in (0.01,0.001,0.0001):
    for img_shape in shapes:
        make_figure_multiple_models(models = models, density = density, img_shape = img_shape, fig_name = "all_models")
