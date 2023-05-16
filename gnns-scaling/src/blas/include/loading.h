#include "../../combblas/cnpy/cnpy.h"
extern "C" {
#include "../GraphBLAS/Include/GraphBLAS.h"
}

GrB_Matrix load_H(string dataPath)
{
    cnpy::npz_t dataNpz = cnpy::npz_load(dataPath);

    auto feat_shape = dataNpz["feature"].shape;
    auto feat_nrow = feat_shape[0];
    auto feat_ncol = feat_shape[1];
    auto feat_data = (void *) dataNpz["feature"].data<float>();
    auto nnz = feat_nrow * feat_ncol;

    GrB_Matrix H;
    GrB_Matrix_new(&H, GrB_FP32, feat_nrow, feat_ncol);

    GrB_Info info;
    info = GxB_Matrix_pack_FullR(H, &feat_data, nnz, true, GrB_NULL);

    if (info != GrB_SUCCESS)
    {
        std::cout << "Exit code creating H: " << info << std::endl;
    }

    return H;
}

GrB_Matrix load_A(string graphPath)
{
    cnpy::npz_t graphNpz = cnpy::npz_load(graphPath);

    auto shape = graphNpz["shape"].data<int64_t>();

    assert(shape[0] == shape[1]);

    GrB_Index nnz = graphNpz["row"].shape[0];

    vector<GrB_Index> I, J;

    auto row_ptr = graphNpz["row"].data<int32_t>();
    for (int i = 0; i < nnz; i++)
    {
        I.push_back(row_ptr[i]);
    }

    auto col_ptr = graphNpz["col"].data<int32_t>();
    for (int i = 0; i < nnz; i++)
    {
        J.push_back(col_ptr[i]);
    }

    auto values = graphNpz["data"].data<float>();

    GrB_Matrix A;
    GrB_Matrix_new(&A, GrB_FP32, shape[0], shape[1]);

    GrB_Info info;
    info = GrB_Matrix_build_FP32(A, I.data(), J.data(), values, nnz, GxB_IGNORE_DUP);

    if (info != GrB_SUCCESS)
    {
        std::cout << "Exit code creating A: " << info << std::endl;
    }

    return A;
}
