#include <iostream>
#include <mpi.h>
#include "CombBLAS/CombBLAS.h"
#include "cnpy/cnpy.h"


int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif

    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }

    cnpy::npz_t my_npz = cnpy::npz_load("../../../data/n1_s131072/reddit_n1_s131072_graph.npz");

    for (cnpy::npz_t::iterator it = my_npz.begin(); it != my_npz.end(); ++it) {
      std::cout << "Key: " << it->first << std::endl;
    }


    MPI_Finalize();
    return 0;
}
