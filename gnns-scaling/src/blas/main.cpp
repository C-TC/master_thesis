#include <algorithm>
#include <iostream>
#include <vector>

extern "C" {
#include "GraphBLAS/Include/GraphBLAS.h"
}

#include "include/matrix_utils.h"

int main(int argc, char **argv) {

  GrB_init(GrB_BLOCKING);

  int A_rows = 5;
  int H_cols = 3;

  // generate adjacency matrix
  double density = 0.5;
  GrB_Matrix A_mat = gen_adj_matrix(A_rows, density);
  pretty_print_matrix_FP32(A_mat, "A_mat");

  // generate dense matrix
  std::vector<float> dense_values(A_rows * H_cols);

  std::generate(dense_values.begin(), dense_values.end(),
                [&]() { return (((float)(rand()) / RAND_MAX) - 0.5) * 100; });
  GrB_Matrix H_mat;
  GrB_Matrix_new(&H_mat, GrB_FP32, A_rows, H_cols);
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_INP0, GxB_DEFAULT);
  void *val_ptr = (void *)dense_values.data();
  GxB_Matrix_pack_FullR(H_mat, &val_ptr, A_rows * H_cols, false, desc);
  pretty_print_matrix_FP32(H_mat, "H_mat");

#define SEMIRING_TYPE GxB_PLUS_TIMES_FP32 // default
  // #define SEMIRING_TYPE GxB_MIN_PLUS_FP32
  // #define SEMIRING_TYPE GxB_MAX_PLUS_FP32
  GrB_Matrix AH_mat;
  GrB_Matrix_new(&AH_mat, GrB_FP32, A_rows, H_cols);
  GrB_mxm(AH_mat, GrB_NULL, GrB_NULL, SEMIRING_TYPE, A_mat, H_mat, desc);
  mult_matrix_FP32(A_mat, H_mat, SEMIRING_TYPE);
  pretty_print_matrix_FP32(AH_mat, "AH_mat");

  FILE *sparse_file = NULL;
  FILE *dense_file = NULL;
  FILE *out_file = NULL;
  sparse_file = fopen(argv[1], "r");
  dense_file = fopen(argv[2], "r");
  out_file = fopen(argv[3], "w");
  if (sparse_file == NULL || dense_file == NULL || out_file == NULL) {
    fprintf(stderr, "unable to read input files or create output file\n");
    exit(1);
  }
  GrB_Matrix sparse_mat;
  GrB_Matrix dense_mat;
  GrB_Matrix out_mat;
  read_matrix(&sparse_mat, sparse_file, false, false, false, false, true);
  pretty_print_matrix_FP32(sparse_mat, "sprase_mat");

  fclose(sparse_file);
  fclose(dense_file);
  fclose(out_file);
  sparse_file = NULL;
  dense_file = NULL;
  out_file = NULL;

  GrB_finalize();

  return GrB_SUCCESS;
}
