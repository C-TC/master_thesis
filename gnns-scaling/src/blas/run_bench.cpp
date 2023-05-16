#include <iostream>
#include <vector>
#include <fstream>

extern "C" {
#include "GraphBLAS/Include/GraphBLAS.h"
}

#include "include/matrix_utils.h"
#include "include/loading.h"
#include "include/timer.h"
#include "include/avg_semiring.h"

const int NUM_REPS = 10;

std::map<string, double> run_exp(string dataset)
{
    std::map<string, double> results;
    double worst_time;
    Timer timer;
    // these path definetly have to be changed
    string dataPath = "../../../data/kron/" + dataset + "/reddit_data.npz";
    GrB_Matrix H = load_H(dataPath);

    string graphPath = "../../../data/kron/" + dataset + "/reddit_" + dataset + "_graph.npz";
    GrB_Matrix A = load_A(graphPath);

    GrB_Index a_nnz, h_nnz;
    GrB_Matrix_nvals(&a_nnz, A);
    GrB_Matrix_nvals(&h_nnz, H);
    std::cout << "A non-zeros: " << a_nnz << std::endl;
    std::cout << "H non-zeros: " << h_nnz << std::endl;

    GrB_Index m, n;
    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, H);
    std::cout << "A rows: " << m << std::endl;
    std::cout << "H cols: " << n << std::endl;

    GrB_Info info;
    GrB_Type UDF_TUPLE;
    GrB_Type_new(&UDF_TUPLE, sizeof(AvgTuple));

    GrB_BinaryOp mulop, addop;
    GrB_BinaryOp_new(&mulop, mul, UDF_TUPLE, GrB_FP32, GrB_FP32);
    GrB_BinaryOp_new(&addop, add, UDF_TUPLE, UDF_TUPLE, UDF_TUPLE);

    GrB_Monoid monoid;
    AvgTuple dummyIdentity;
    GrB_Monoid_new_UDT(&monoid, addop, (void *) &dummyIdentity);

    GrB_Semiring UDF_AVG_SEM;
    GrB_Semiring_new(&UDF_AVG_SEM, monoid, mulop);

    GrB_Matrix AH;
    GrB_Matrix_new(&AH, UDF_TUPLE, m, n);

    vector<double> times;

    for (int i = 0; i < NUM_REPS; ++i)
    {
        timer.start();
        info = GrB_mxm(AH, GrB_NULL, GrB_NULL, UDF_AVG_SEM, A, H, GrB_NULL);
        timer.stop();
        if (info != GrB_SUCCESS)
        {
            std::cout << "Exit code during mxm AH: " << info << std::endl;
        }
        times.push_back(timer.elapsedMilliseconds());
        GrB_Matrix_clear(AH);
    }
    worst_time = *max_element(times.begin(), times.end());
    std::cout << "AVG Semiring - millis: " << worst_time << std::endl;
    results["AVG"] = worst_time;
    times.clear();

    GrB_Matrix_new(&AH, GrB_FP32, m, n);

    for (int i = 0; i < NUM_REPS; ++i)
    {
        timer.start();
        info = GrB_mxm(AH, GrB_NULL, GrB_NULL, GxB_TIMES_MAX_FP32, A, H, GrB_NULL);
        timer.stop();
        if (info != GrB_SUCCESS)
        {
            std::cout << "Exit code during mxm AH: " << info << std::endl;
        }
        times.push_back(timer.elapsedMilliseconds());
        GrB_Matrix_clear(AH);
    }
    worst_time = *max_element(times.begin(), times.end());
    std::cout << "MAX Semiring - millis: " << *max_element(times.begin(), times.end()) << std::endl;
    results["MAX"] = worst_time;
    times.clear();

    timer.start();
    for (int i = 0; i < NUM_REPS; ++i)
    {
        timer.start();
        info = GrB_mxm(AH, GrB_NULL, GrB_NULL, GxB_TIMES_PLUS_FP32, A, H, GrB_NULL);
        timer.stop();
        if (info != GrB_SUCCESS)
        {
            std::cout << "Exit code during mxm AH: " << info << std::endl;
        }
        times.push_back(timer.elapsedMilliseconds());
        GrB_Matrix_clear(AH);
    }
    worst_time = *max_element(times.begin(), times.end());
    std::cout << "SUM Semiring - millis: " << *max_element(times.begin(), times.end()) << std::endl;
    results["SUM"] = worst_time;
    times.clear();

    return results;
}

int main(int argc, char **argv) {
    std::ofstream outdata;
    outdata.open("results.csv");
    outdata << "dataset,sum,max,avg" << std::endl;
    GrB_init(GrB_BLOCKING);

    /*
       Kron_s-17_e-1311_s-0.01
       Kron_s-18_e-2622_s-0.01
       Kron_s-20_e-105_s-0.0001
       Kron_s-21_e-210_s-0.0001
    */
    vector<string> datasets = {
        "n1_a17_e14_s0.0001", // super small for debug
        /* "n1_a17_e1311_s0.01", */
        /* "n1_a18_e2622_s0.01", */
        /* "n1_a20_e105_s0.0001", */
        /* "n1_a21_e210_s0.0001", */
    };
    for (string dataset: datasets)
    {
        std::cout << "Dataset: " << dataset << std::endl;
        auto results = run_exp(dataset);
        outdata << dataset << "," << results["SUM"] << ","
            << results["MAX"] << "," << results["AVG"] << endl;
    }

    outdata.close();

    return 0;
}
