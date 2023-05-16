// Copyright (c) 2023 ETH-Zurich.
//                    All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// main author: Robert Gerstenberger

#include <stdbool.h>

#include "make_graph.h"

#include "mpi.h"

void generateEdgeGraph500Kronecker( int rank, int commsize, uint32_t edge_factor, uint32_t SCALE, uint64_t* output_edge_count, uint32_t** origin, uint32_t** target );
//void distributeEdges( MPI_Offset edge_count, packed_edge* buf, uint32_t lNI, uint32_t lNK, uint32_t Py, bool directed, uint64_t* output_edge_count, uint32_t** origin, uint32_t** target );
void createEdgesDistributed( uint32_t scale, uint32_t edgefactor, uint32_t lNI, uint32_t lNK, uint32_t Py, uint64_t* edge_count, uint32_t** origin, uint32_t** target );

void createEdgesSingleNode( uint32_t scale, uint32_t edgefactor, uint64_t* edge_count, uint32_t** origin, uint32_t** target );
void freeEdges( uint32_t** origin, uint32_t** target );
