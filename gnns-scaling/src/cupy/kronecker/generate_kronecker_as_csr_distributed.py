import ctypes
import sys
import numpy as np
import scipy.sparse as sps

# Arguments
# - vertex_count:		Number of vertices that the Kronecker graph should have
# - target_edge_count:	Number of edges that the Kronecker graph should have. Note that the real number 
#						of edges of the generated Kronecker graph might differ form this value.
# - rows_of_blocks:		Number of horizontal slices in which the full adjacency matrix should be divided
# 						Note: The number of processes must be equal to rows_of_blocks * cols_of_blocks
# - cols_of_blocks:		Number of vertical slices in which the full adjacency matrix should be divided
# 						Note: The number of processes must be equal to rows_of_blocks * cols_of_blocks
# - target_data_type:	The NumPy datatype that should be used to stores 0s and 1s in the CSR matrix
#
# Return value
# - The block of the adjacency matrix that the current process is responsible for in CSR format

def create_kronecker_graph(vertex_count, target_edge_count, rows_of_blocks, cols_of_blocks, target_data_type): 
	# ---------------------------------------------------------------------------------------------------- 
	# Generate the Kronecker graph in a distributed setting by calling the respective C function
	# ---------------------------------------------------------------------------------------------------- 

	# Compute some parameters related to the distribution of the graph
	entries_per_row_of_blocks = vertex_count // rows_of_blocks
	entries_per_col_of_blocks = vertex_count // cols_of_blocks
	
	# The C function that creates the Kronecker graph will store the length of the edge list and
	# a pointer to the edge list into these two variables
	output_edge_count = (ctypes.c_ulonglong)()
	output_edge_pointer = (ctypes.c_ulonglong)()

	# TODO: Replace this line by absolute path on cluster
	graph_lib = ctypes.CDLL("/home/iffp/Dropbox/SPCL/gnn-rl/gnns-scaling/src/cupy/kronecker/graph.so")

	# Call C function to create the Kronecker graph
	graph_lib.createEdges(	vertex_count, \
							target_edge_count, \
							entries_per_row_of_blocks, \
							entries_per_col_of_blocks, \
							cols_of_blocks, \
							ctypes.byref(output_edge_count), \
							ctypes.byref(output_edge_pointer))   
	
	# Retrieve and convert the edge list that is created by the C function
	real_edge_count	= output_edge_count.value
	edges_as_c_array = ctypes.cast(output_edge_pointer.value, ctypes.POINTER(ctypes.c_ulonglong * real_edge_count * 2))
	tmp = list(np.ndarray((real_edge_count * 2, ), 'ulonglong', edges_as_c_array.contents, order='C'))
	edge_list = [(tmp[i],tmp[i+1]) for i in range(0, 2 * real_edge_count, 2)]

	# ---------------------------------------------------------------------------------------------------- 
	# Identify the block of the adjacency matrix for which the current process is responsible
	# ---------------------------------------------------------------------------------------------------- 

	# Identify Row
	start_id_set = set([x for (x,y) in edge_list])
	min_start_id = min(start_id_set)
	max_start_id = max(start_id_set)
	own_row_of_blocks = -1	
	for i in range(rows_of_blocks):
		if ((i * entries_per_row_of_blocks) <= min_start_id) and \
		   (min_start_id < ((i + 1) * entries_per_row_of_blocks)) and \
		   ((i * entries_per_row_of_blocks) <= max_start_id) and \
		   (max_start_id < ((i + 1) * entries_per_row_of_blocks)):
			own_row_of_blocks = i	
	if own_row_of_blocks < 0:
		print("ERROR: Unable to identify the processes row of blocks")
		sys.exit()

	# Identify Column
	end_id_set = set([y for (x,y) in edge_list])
	min_end_id = min(end_id_set)	
	max_end_id = max(end_id_set)
	own_col_of_blocks = -1	
	for i in range(cols_of_blocks):
		if ((i * entries_per_col_of_blocks) <= min_end_id) and \
		   (min_end_id < ((i + 1) * entries_per_col_of_blocks)) and \
		   ((i * entries_per_col_of_blocks) <= max_end_id) and \
		   (max_end_id < ((i + 1) * entries_per_col_of_blocks)):
			own_col_of_blocks = i	
	if own_col_of_blocks < 0:
		print("ERROR: Unable to identify the processes column of blocks")
		sys.exit()

	# ---------------------------------------------------------------------------------------------------- 
	# Transform the edge list into an adjacency matrix in CSR format
	# ---------------------------------------------------------------------------------------------------- 

	# Remap vertices from global id (id within the whole graph) to local id (id within the processes sub-graph)
	row_shift = own_row_of_blocks * entries_per_row_of_blocks
	col_shift = own_col_of_blocks * entries_per_col_of_blocks
	for i in range(real_edge_count):
		edge_list[i] = (edge_list[i][0] - row_shift, edge_list[i][1] - col_shift)

	# Convert the edge list to a SciPy CSR matrix
	rows = [r for (r,c) in edge_list]
	cols = [c for (r,c) in edge_list]
	vals = np.ones(len(edge_list), dtype=target_data_type) 
	mat = sps.csr_matrix((vals, (rows, cols)), shape=(entries_per_row_of_blocks, entries_per_col_of_blocks), dtype=target_data_type)

	# Replace multi-edges by single-edges  
	mat.data = np.ones(len(mat.data), dtype=target_data_type) 
	
	# Return matrix
	return mat

# Test run
#create_kronecker_graph(16, 128, 2, 2, "f")


