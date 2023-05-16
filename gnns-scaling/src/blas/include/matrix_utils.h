#pragma once
#include <algorithm>
#include <iostream>
#include <vector>

extern "C" {
#include "../GraphBLAS/Include/GraphBLAS.h"
}

#include "graphblas_demos.h"

void scale2 (double *z, const double *x)
{
    (*z) = (*x) / 2.0 ;
}

#define FREE_ALL                            \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&B) ;                  \
    GrB_Matrix_free (&C) ;                  \
    if (I != NULL) free (I) ;               \
    if (J != NULL) free (J) ;               \
    if (X != NULL) free (X) ;


GB_PUBLIC
GrB_Info read_matrix        // read a double-precision or boolean matrix
(
    GrB_Matrix *A_output,   // handle of matrix to create
    FILE *f,                // file to read the tuples from
    bool make_symmetric,    // if true, return A as symmetric
    bool no_self_edges,     // if true, then remove self edges from A
    bool one_based,         // if true, input matrix is 1-based
    bool boolean,           // if true, input is GrB_BOOL, otherwise GrB_FP64
    bool pr                 // if true, print status to stdout
)
{

    int64_t len = 256 ;
    int64_t ntuples = 0 ;
    double x ;
    GrB_Index nvals ;

    //--------------------------------------------------------------------------
    // set all pointers to NULL so that FREE_ALL can free everything safely
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, A = NULL, B = NULL ;
    GrB_Descriptor dt1 = NULL, dt2 = NULL ;
    GrB_UnaryOp scale2_op = NULL ;

    //--------------------------------------------------------------------------
    // allocate initial space for tuples
    //--------------------------------------------------------------------------

    size_t xsize = ((boolean) ? sizeof (bool) : sizeof (float)) ;
    GrB_Index *I = (GrB_Index *) malloc (len * sizeof (GrB_Index)), *I2 = NULL ;
    GrB_Index *J = (GrB_Index *) malloc (len * sizeof (GrB_Index)), *J2 = NULL ;
    void *X = malloc (len * xsize) ;
    bool *Xbool ;
    float *Xfloat ;
    void *X2 = NULL ;
    if (I == NULL || J == NULL || X == NULL)
    {
        // out of memory
        if (pr) printf ("out of memory for initial tuples\n") ;
        FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    Xbool   = (bool   *) X ;
    Xfloat = (float *) X ;

    //--------------------------------------------------------------------------
    // read in the tuples from stdin, one per line
    //--------------------------------------------------------------------------

    // format warnings vary with compilers, so read in as double
    double i2, j2 ;
    while (fscanf (f, "%lg %lg %lg\n", &i2, &j2, &x) != EOF)
    {
        int64_t i = (int64_t) i2 ;
        int64_t j = (int64_t) j2 ;
        if (ntuples >= len)
        {
            I2 = (GrB_Index *) realloc (I, 2 * len * sizeof (GrB_Index)) ;
            J2 = (GrB_Index *) realloc (J, 2 * len * sizeof (GrB_Index)) ;
            X2 = realloc (X, 2 * len * xsize) ;
            if (I2 == NULL || J2 == NULL || X2 == NULL)
            {
                if (pr) printf ("out of memory for tuples\n") ;
                FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
            I = I2 ; I2 = NULL ;
            J = J2 ; J2 = NULL ;
            X = X2 ; X2 = NULL ;
            len = len * 2 ;
            Xbool   = (bool   *) X ;
            Xfloat = (float *) X ;
        }
        if (one_based)
        {
            i-- ;
            j-- ;
        }
        I [ntuples] = i ;
        J [ntuples] = j ;
        if (boolean)
        {
            Xbool [ntuples] = (x != 0) ;
        }
        else
        {
            Xfloat [ntuples] = x ;
        }
        ntuples++ ;
    }

    //--------------------------------------------------------------------------
    // find the dimensions
    //--------------------------------------------------------------------------

    if (pr) printf ("ntuples: %.16g\n", (double) ntuples) ;
    int64_t nrows = 0 ;
    int64_t ncols = 0 ;
    for (int64_t k = 0 ; k < ntuples ; k++)
    {
        nrows = MAX (nrows, I [k]) ;
        ncols = MAX (ncols, J [k]) ;
    }
    nrows++ ;
    ncols++ ;

    if (pr) printf ("nrows %.16g ncols %.16g\n",
        (double) nrows, (double) ncols) ;

    //--------------------------------------------------------------------------
    // prune self edges
    //--------------------------------------------------------------------------

    // but not if creating the augmented system aka a bipartite graph
    if (no_self_edges && ! (make_symmetric && nrows != ncols))
    {
        int64_t ntuples2 = 0 ;
        for (int64_t k = 0 ; k < ntuples ; k++)
        {
            if (I [k] != J [k])
            {
                // keep this off-diagonal edge
                I [ntuples2] = I [k] ;
                J [ntuples2] = J [k] ;
                if (boolean)
                {
                    Xbool [ntuples2] = Xbool [k] ;
                }
                else
                {
                    Xfloat [ntuples2] = Xfloat [k] ;
                }
                ntuples2++ ;
            }
        }
        ntuples = ntuples2 ;
    }

    //--------------------------------------------------------------------------
    // build the matrix, summing up duplicates, and then free the tuples
    //--------------------------------------------------------------------------

    GrB_Type xtype ;
    GrB_BinaryOp xop, xop_first ;
    if (boolean)
    {
        xtype = GrB_BOOL ;
        xop   = GrB_LOR ;
        xop_first  = GrB_FIRST_BOOL ;
    }
    else
    {
        xtype = GrB_FP32 ;
        xop   = GrB_PLUS_FP32 ;
        xop_first  = GrB_FIRST_FP32 ;
    }

    GrB_Info info ;
    OK (GrB_Matrix_new (&C, xtype, nrows, ncols)) ;

    if (boolean)
    {
        OK (GrB_Matrix_build_BOOL (C, I, J, Xbool, ntuples, xop)) ;
    }
    else
    {
        OK (GrB_Matrix_build_FP32 (C, I, J, Xfloat, ntuples, xop)) ;
    }

    free (I) ; I = NULL ;
    free (J) ; J = NULL ;
    free (X) ; X = NULL ;

    //--------------------------------------------------------------------------
    // construct the descriptors
    //--------------------------------------------------------------------------

    // descriptor dt2: transpose the 2nd input
    OK (GrB_Descriptor_new (&dt2)) ;
    OK (GrB_Descriptor_set (dt2, GrB_INP1, GrB_TRAN)) ;

    // descriptor dt1: transpose the 1st input
    OK (GrB_Descriptor_new (&dt1)) ;
    OK (GrB_Descriptor_set (dt1, GrB_INP0, GrB_TRAN)) ;

    //--------------------------------------------------------------------------
    // create the output matrix
    //--------------------------------------------------------------------------

    if (make_symmetric)
    {

        //----------------------------------------------------------------------
        // ensure the matrix is symmetric
        //----------------------------------------------------------------------

        if (pr) printf ("make symmetric\n") ;
        if (nrows == ncols)
        {

            //------------------------------------------------------------------
            // A = (C+C')/2
            //------------------------------------------------------------------

            if (pr) printf ("A = (C+C')/2\n") ;
            OK (GrB_Matrix_new (&A, xtype, nrows, nrows)) ;
            OK (GrB_Matrix_eWiseAdd_BinaryOp (A, NULL, NULL, xop, C, C, dt2)) ;
            OK (GrB_Matrix_free (&C)) ;

            if (boolean)
            {
                *A_output = A ;
                A = NULL ;
            }
            else
            {
                OK (GrB_Matrix_new (&C, xtype, nrows, nrows)) ;
                OK (GrB_UnaryOp_new (&scale2_op, 
                    (GxB_unary_function) scale2, xtype, xtype)) ;
                OK (GrB_Matrix_apply (C, NULL, NULL, scale2_op, A, NULL)) ;
                OK (GrB_Matrix_free (&A)) ;
                OK (GrB_UnaryOp_free (&scale2_op)) ;
                *A_output = C ;
                C = NULL ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // A = [0 C ; C' 0], a bipartite graph
            //------------------------------------------------------------------

            // no self edges will exist
            if (pr) printf ("A = [0 C ; C' 0], a bipartite graph\n") ;

            int64_t n = nrows + ncols ;
            OK (GrB_Matrix_new (&A, xtype, n, n)) ;

            GrB_Index I_range [3], J_range [3] ;

            I_range [GxB_BEGIN] = 0 ;
            I_range [GxB_END  ] = nrows-1 ;

            J_range [GxB_BEGIN] = nrows ;
            J_range [GxB_END  ] = ncols+nrows-1 ;

            // A (nrows:n-1, 0:nrows-1) += C'
            OK (GrB_Matrix_assign (A, NULL, xop_first, // or NULL,
                C, J_range, GxB_RANGE, I_range, GxB_RANGE, dt1)) ;

            // A (0:nrows-1, nrows:n-1) += C
            OK (GrB_Matrix_assign (A, NULL, xop_first, // or NULL,
                C, I_range, GxB_RANGE, J_range, GxB_RANGE, NULL)) ;

            // force completion; if this statement does not appear, the
            // timing will not account for the final build, which would be
            // postponed until A is used by the caller in another GraphBLAS
            // operation.
            GrB_Matrix_nvals (&nvals, A) ;

            *A_output = A ;
            // set A to NULL so the FREE_ALL macro does not free *A_output
            A = NULL ;

        }
    }
    else
    {

        //----------------------------------------------------------------------
        // return the matrix as-is
        //----------------------------------------------------------------------

        if (pr) printf ("leave A as-is\n") ;
        *A_output = C ;
        // set C to NULL so the FREE_ALL macro does not free *A_output
        C = NULL ;
    }

    //--------------------------------------------------------------------------
    // success: free everything except the result, and return it to the caller
    //--------------------------------------------------------------------------

    FREE_ALL ;
    if (pr) printf ("\nMatrix from file:\n") ;
    GxB_Matrix_fprint (*A_output, "*A_output", pr ? GxB_SHORT : GxB_SILENT,
        stdout) ;
    return (GrB_SUCCESS) ;
}

void pretty_print_matrix_FP32(GrB_Matrix &mat, char const *label) {
  GrB_Index M, N;
  GrB_Matrix_nrows(&M, mat);
  GrB_Matrix_ncols(&N, mat);

  float val;
  GrB_Info ret_val;

  printf("Matrix: %s =\n", label);
  for (GrB_Index row = 0; row < M; ++row) {
    printf("[");
    for (GrB_Index col = 0; col < N; ++col) {
      ret_val = GrB_Matrix_extractElement_FP32(&val, mat, row, col);
      if (GrB_SUCCESS == ret_val) {
        if (col == 0) {
          printf("%3.1f", val);
        } else {
          printf(", %3.1f", val);
        }
      } else if (GrB_NO_VALUE == ret_val) {
        if (col == 0) {
          printf("  -");
        } else {
          printf(",   -");
        }
      } else {
        if (col == 0) {
          printf("  ERR");
        } else {
          printf(", ERR");
        }
      }
    }
    printf("]\n");
  }
}

GrB_Matrix gen_adj_matrix(const int num_rows, const double density) {
  const int nnz = (int)(num_rows * num_rows * density);

  std::vector<GrB_Index> row_indices(nnz);
  std::vector<GrB_Index> col_indices(nnz);
  std::vector<float> values(nnz);

  std::generate(row_indices.begin(), row_indices.end(),
                [num_rows]() { return rand() % num_rows; });
  std::generate(col_indices.begin(), col_indices.end(),
                [num_rows]() { return rand() % num_rows; });
  std::generate(values.begin(), values.end(), [density]() {
    return 1.0f ? (rand() / RAND_MAX) < density : 0.0f;
  });

  GrB_Matrix mat;
  GrB_Index const NUM_NODES = num_rows;
  GrB_Index const NUM_EDGES = nnz;
  GrB_Matrix_new(&mat, GrB_FP32, NUM_NODES, NUM_NODES);
  GrB_Matrix_build_FP32(mat, row_indices.data(), col_indices.data(),
                        values.data(), NUM_EDGES, GrB_LOR);

  return mat;
}

GrB_Matrix gen_dense_matrix(const int num_rows, const int num_cols) {
  std::vector<GrB_Index> row_indices(num_rows * num_cols);
  std::vector<GrB_Index> col_indices(num_rows * num_cols);
  std::vector<float> values(num_rows * num_cols);

  int i = 0;
  std::generate(row_indices.begin(), row_indices.end(), [&]() {
    int ret = i % num_rows;
    i++;
    return ret;
  });
  i = 0;
  int col = 0;
  std::generate(col_indices.begin(), col_indices.end(), [&]() {
    int ret = (int)i / num_rows;
    i++;
    return ret;
  });
  std::generate(values.begin(), values.end(),
                [&]() { return (((float)(rand()) / RAND_MAX) - 0.5) * 100; });

  GrB_Matrix mat;
  GrB_Index const NUM_ROWS = num_rows;
  GrB_Index const NUM_COLS = num_cols;
  GrB_Index const NUM_EDGES = num_rows * num_cols;
  GrB_Matrix_new(&mat, GrB_FP32, NUM_ROWS, NUM_COLS);
  GrB_Matrix_build_FP32(mat, row_indices.data(), col_indices.data(),
                        values.data(), NUM_EDGES, GrB_FIRST_FP32);

  return mat;
}

GrB_Matrix gen_dense_full_matrix(const int num_rows, const int num_cols) {
  std::vector<float> values(num_rows * num_cols);

  std::generate(values.begin(), values.end(),
                [&]() { return (((float)(rand()) / RAND_MAX) - 0.5) * 100; });
  for (auto &e : values) {
    std::cout << e << ' ';
  }
  std::cout << "\n";

  GrB_Matrix mat;
  GrB_Index const NUM_ROWS = num_rows;
  GrB_Index const NUM_COLS = num_cols;
  GrB_Index const NUM_EDGES = num_rows * num_cols;
  GrB_Matrix_new(&mat, GrB_FP32, num_rows, num_cols);
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_INP0, GxB_DEFAULT);
  void* val_ptr = (void*) values.data();
  GxB_Matrix_pack_FullR(mat, &val_ptr, (GrB_Index) num_rows * num_cols, false, desc);
  pretty_print_matrix_FP32(mat, "dense_full_func");
  return mat;
}

GrB_Matrix mult_matrix_FP32(GrB_Matrix &mat1, GrB_Matrix &mat2, GrB_Semiring semiring) {
  GrB_Matrix result;
  GrB_Index num_rows;
  GrB_Index num_cols;
  GrB_Matrix_nrows(&num_rows, mat1);
  GrB_Matrix_ncols(&num_cols, mat2);
  GrB_Matrix_new(&result, GrB_FP32, num_rows, num_cols);

  GrB_Info info;
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_INP0, GxB_DEFAULT);

  info = GrB_mxm(result, GrB_NULL, GrB_NULL, semiring, mat1, mat2, desc);
  return result;
}