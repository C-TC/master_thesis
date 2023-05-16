// Copyright (c) 2022 ETH-Zurich.
//                    All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// main author: Robert Gerstenberger

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "make_graph.h"
#include "utils.h"

#include "mpi.h"

void generateEdgeGraph500Kronecker( int rank, int commsize, uint32_t edge_factor, uint32_t SCALE,
  uint32_t lNI, uint32_t lNK, uint32_t Py, uint64_t* output_edge_count, uint32_t** origin, uint32_t** target, uint32_t** send_count ) {
  int64_t nglobaledges = (int64_t)(edge_factor) << SCALE;

	/* Make the raw graph edges. */

  /* Spread the two 64-bit numbers into five nonzero values in the correct
   * range. */
  uint_fast32_t seed[5];
	uint64_t seed1 = 2, seed2 = 3;
  make_mrg_seed(seed1, seed2, seed);

  /**
    generate the edges of the graph
   */
  int64_t nlocaledges = (nglobaledges + commsize - 1 /* round up */)/commsize;
  int64_t start_edge_index = nlocaledges * rank < nglobaledges ? nlocaledges * rank : nglobaledges; /* minimum( nlocaledges * rank, nglobaledges ) */
  int64_t edge_count = nglobaledges - start_edge_index < nlocaledges ? nglobaledges - start_edge_index : nlocaledges; /* minimum( nglobaledges - start_edge_index, nlocaledges ) */

  packed_edge* buf = (packed_edge*)xmalloc(edge_count * sizeof(packed_edge));
  generate_kronecker_range(seed, SCALE, start_edge_index, start_edge_index + edge_count, buf);

  *output_edge_count = edge_count;

  *origin = malloc( (*output_edge_count) * sizeof(uint32_t) );
  *target = malloc( (*output_edge_count) * sizeof(uint32_t) );

  *send_count = malloc( commsize * sizeof(uint32_t) );
  memset( *send_count, 0, commsize * sizeof(uint32_t) );

  for( size_t i=0 ; i<(*output_edge_count) ; i++ ) {
    (*origin)[i] = get_v0_from_edge( buf+i );
    (*target)[i] = get_v1_from_edge( buf+i );

    int process = (*origin)[i] / lNI * Py + (*target)[i] / lNK;

    assert( (process >= 0) && (process < commsize) );

    (*send_count)[process]++;
  }

  free( buf );
}


void packBufferDispl( int commsize, uint32_t lNI, uint32_t lNK, uint32_t Py, uint64_t edge_count, uint32_t** origin, uint32_t** target, uint32_t** send_count, uint32_t** send_displ, uint32_t** buf ) {
  *send_displ = malloc( commsize * sizeof(uint32_t) );

  (*send_displ)[0] = 0;
  for( int i=1 ; i<commsize ; i++ ) {
    (*send_displ)[i] = (*send_displ)[i-1] + (*send_count)[i-1];
  }

  size_t total_send_count = (*send_displ)[commsize-1] + (*send_count)[commsize-1];

  *buf = malloc( total_send_count * sizeof(uint32_t) );

  uint32_t* idx = malloc( commsize * sizeof(uint32_t) );
  memcpy( idx, *send_displ, commsize * sizeof(uint32_t) );

  for( size_t i=0 ; i<edge_count ; i++ ) {
    int process = (*origin)[i] / lNI * Py + (*target)[i] / lNK;

    assert( (process >= 0) && (process < commsize) );

    (*buf)[idx[process]++] = (*origin)[i];
  }

  free(idx);
}


void packBuffer( int commsize, uint32_t lNI, uint32_t lNK, uint32_t Py, uint64_t edge_count, uint32_t** origin, uint32_t** target, uint32_t** send_displ, uint32_t** buf ) {
  uint32_t* idx = malloc( commsize * sizeof(uint32_t) );
  memcpy( idx, *send_displ, commsize * sizeof(uint32_t) );

  for( size_t i=0 ; i<edge_count ; i++ ) {
    int process = (*origin)[i] / lNI * Py + (*target)[i] / lNK;

    assert( (process >= 0) && (process < commsize) );

    (*buf)[idx[process]++] = (*target)[i];
  }

  free(idx);
}


void freeData( uint32_t** send_count, uint32_t** send_displ, uint32_t** buf ) {
  free( *send_count );
  free( *send_displ );
  free( *buf );
}


#if 0
/**
  nglobalverts = number of vertices
 */
void distributeEdges( MPI_Offset edge_count, packed_edge* buf, uint32_t lNI, uint32_t lNK, uint32_t Py, bool directed, uint64_t* output_edge_count, uint32_t** origin, uint32_t** target ) {
  int rank, commsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);

#ifdef GDEBUG
  for( int64_t i=0 ; i<edge_count ; i++ ) {
    int64_t origin = get_v0_from_edge( buf+i );
    int64_t target = get_v1_from_edge( buf+i );
    printf( "%i: %" PRId64 " -> %" PRId64 "\n", rank, origin, target );
  }
#endif

  /**
    distribute the generated edges to the processes that store the origin and
    the target vertices

    step 1: determine the necessary buffer size in Bytes for each process
    step 2a: start the distribution of the send count (non-blocking MPI_Alltoall)
    step 3: pack the buffers for each process
    step 2b: finish up the distribution of the send count
    step 4: distribute the edges (MPI_Alltoallv)
   */

  /**
    step 1: determine the necessary buffer size in Bytes for each process
   */
  int* send_count = calloc( commsize, sizeof(int) );
  if( send_count == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for( int64_t i=0 ; i<edge_count ; i++ ) {
    int64_t origin_vertex = get_v0_from_edge( buf+i );
    int64_t target_vertex = get_v1_from_edge( buf+i );

    int process = origin_vertex / lNI * Py + target_vertex / lNK;

    assert( (process >= 0) && (process < commsize) );

    send_count[process] += sizeof(packed_edge);

    if( !directed ) {
      process = target_vertex / lNI * Py + origin_vertex / lNK;

      assert( (process >= 0) && (process < commsize) );

      send_count[process] += sizeof(packed_edge);
    }
  }

  /**
    step 2a: start the distribution of the send count
   */
  int* recv_count = malloc( commsize * sizeof(int) );
  if( recv_count == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Request req;

  MPI_Ialltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD, &req );

  /**
    determine displacements
   */
  int* send_displacement = malloc( commsize * sizeof(int) );
  if( send_displacement == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  send_displacement[0] = 0;
  for( int i=1 ; i<commsize ; i++ ) {
    send_displacement[i] = send_displacement[i-1] + send_count[i-1];
  }

  size_t total_send_count = send_displacement[commsize-1] + send_count[commsize-1]; /* in Bytes */

  /**
    step 3: pack the buffers for each process
   */
  char* edge_distribution = malloc( total_send_count );
  if( edge_distribution == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint32_t* idx = calloc( commsize, sizeof(uint32_t) );
  if( idx == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for( int64_t i=0 ; i<edge_count ; i++ ) {
    int64_t origin_vertex = get_v0_from_edge( buf+i );
    int64_t target_vertex = get_v1_from_edge( buf+i );

    int process = origin_vertex / lNI * Py + target_vertex / lNK;

    assert( (process >= 0) && (process < commsize) );

    /**
      copy edge into the process buffers
     */
#ifdef GDEBUG
    printf( "%i: write edge %" PRId64 " -> %" PRId64 " into buffer for process %i at index %lu\n", rank, origin_vertex, target_vertex, process, idx[process] / sizeof(packed_edge) );
#endif
    memcpy( edge_distribution+send_displacement[process]+idx[process], buf+i, sizeof(packed_edge) );
    idx[process] += sizeof(packed_edge);

    if( !directed ) {
      process = target_vertex / lNI * Py + origin_vertex / lNK;

#ifdef GDEBUG
      printf( "%i: write edge %" PRId64 " -> %" PRId64 " into buffer for process %i at index %lu\n", rank, origin_vertex, target_vertex, process, idx[process] / sizeof(packed_edge) );
#endif
      memcpy( edge_distribution+send_displacement[process]+idx[process], buf+i, sizeof(packed_edge) );
      idx[process] += sizeof(packed_edge);
    }
  }

  free(buf);
  free(idx);

  /**
    step 2b: finish up the distribution of the send count
   */
  MPI_Wait( &req, MPI_STATUS_IGNORE );

#ifdef GDEBUG
  for( int i=0 ; i<commsize ; i++ ) {
    printf( "%i: recv_count[%i] = %li\n", rank, i, recv_count[i] / sizeof(packed_edge) );
  }
#endif

  /**
    step 4: distribute the edges
   */
  int* recv_displacement = malloc( commsize * sizeof(int) );
  if( recv_displacement == NULL ) {
    fprintf( stderr, "%i - %s line %i: Not able to allocate memory\n", rank, (char*) __FILE__, __LINE__ );
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  recv_displacement[0] = 0;
  for( int i=1 ; i<commsize ; i++ ) {
    recv_displacement[i] = recv_displacement[i-1] + recv_count[i-1];
  }

  size_t total_recv_count = recv_displacement[commsize-1] + recv_count[commsize-1]; /* in Bytes */

  buf = (packed_edge*) malloc( total_recv_count );

  MPI_Alltoallv( edge_distribution, send_count, send_displacement, MPI_CHAR, buf, recv_count, recv_displacement, MPI_CHAR, MPI_COMM_WORLD );

  size_t received_total_edge_count = total_recv_count/sizeof(packed_edge); /* in number of edges */
#ifdef GDEBUG
  for( size_t i=0 ; i<received_total_edge_count ; i++ ) {
    int64_t origin = get_v0_from_edge( buf+i );
    int64_t target = get_v1_from_edge( buf+i );
    printf( "%i: %" PRId64 " -> %" PRId64 "\n", rank, origin, target );
  }
#endif

  /**
    clean up
   */
  free( edge_distribution );
  free( send_count );
  free( recv_count );
  free( send_displacement );
  free( recv_displacement );

  *output_edge_count = received_total_edge_count;

  *origin = malloc( (*output_edge_count) * sizeof(uint32_t) );
  *target = malloc( (*output_edge_count) * sizeof(uint32_t) );

  for( size_t i=0 ; i<(*output_edge_count) ; i++ ) {
    (*origin)[i] = get_v0_from_edge( buf+i );
    (*target)[i] = get_v1_from_edge( buf+i );
  }

  free( buf );
}

void createEdgesDistributed( uint32_t scale, uint32_t edge_factor, uint32_t lNI, uint32_t lNK, uint32_t Py, uint64_t* edge_count, uint32_t** origin, uint32_t** target ) {

    MPI_Offset edge_count_generation;
    packed_edge* edges_generation;

    int flag;
    MPI_Initialized( &flag );

    if( !flag ) {
      MPI_Init( NULL, NULL );
    }

    generateEdgeGraph500Kronecker( edge_factor, scale, &edge_count_generation, &edges_generation );

    distributeEdges( edge_count_generation, edges_generation, lNI, lNK, Py, true /* directed */, edge_count, origin, target );

    if( !flag ) {
      MPI_Finalize();
    }
}
#endif

void createEdgesSingleNode( uint32_t scale, uint32_t edgefactor, uint64_t* edge_count, uint32_t** origin, uint32_t** target ) {

  /* Spread the two 64-bit numbers into five nonzero values in the correct
   * range. */
  uint_fast32_t seed[5];
	uint64_t seed1 = 2, seed2 = 3;
  make_mrg_seed(seed1, seed2, seed);

  *edge_count = (uint64_t)(edgefactor) << scale;

  packed_edge* buf = (packed_edge*)xmalloc((*edge_count) * sizeof(packed_edge));
  generate_kronecker_range(seed, scale, 0 /* start index */, *edge_count /* last index */, buf);

  *origin = malloc( (*edge_count) * sizeof(uint32_t) );
  *target = malloc( (*edge_count) * sizeof(uint32_t) );

  for( size_t i=0 ; i<(*edge_count) ; i++ ) {
    (*origin)[i] = get_v0_from_edge( buf+i );
    (*target)[i] = get_v1_from_edge( buf+i );
  }

  free( buf );
}

void freeEdges( uint32_t** origin, uint32_t** target ) {
  free( *origin );
  free( *target );
}
