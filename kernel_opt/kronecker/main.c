#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "graph.h"

int main( int argc, char* argv[] ) {

#if 0
  uint32_t lNI = 3;
  uint32_t lNK = 3;
  uint32_t Py = 2;

  uint64_t edge_count; /* global count */
  uint64_t* edges;

  createEdges( 6, 16, lNI, lNK, Py, &edge_count, &edges );

  for( uint64_t i=0 ; i<edge_count ; i++ ) {
    printf("%" PRIu64 "->%" PRIu64 "\n", edges[2*i], edges[2*i+1]);
  }

  free( edges );
#endif

  uint64_t edge_count;
  uint32_t* origin;
  uint32_t* target;

  uint32_t scale = 4;
  size_t edges_num = 32;
  size_t vertex_num = 1 << scale;
  uint32_t edge_factor = edges_num/vertex_num;

  createEdgesSingleNode( scale, edge_factor, &edge_count, &origin, &target );

  char filename[100];

  snprintf( filename, 100, "origin_a%zu_e%zu.txt", vertex_num, edges_num );

  int written = 0;

  FILE *f = fopen(filename, "wb");

  written = fwrite(origin, sizeof(uint32_t), edge_count, f);
  if( written == 0 ) {
    fprintf(stderr, "Something went wrong while writing to file %s.\n", filename);
  }

#if 0
  for( uint64_t i=0 ; i<edge_count ; i++ ) {
    printf("%u->%u\n", origin[i], target[i]);
  }
#endif

  fclose(f);

  snprintf( filename, 100, "target_a%zu_e%zu.txt", vertex_num, edges_num );

  f = fopen(filename, "wb");

  fwrite(target, sizeof(uint32_t), edge_count, f);

  fclose(f);

  freeEdges( &origin, &target );
}
