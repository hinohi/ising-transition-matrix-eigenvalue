#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>

using namespace Spectra;

using T = Eigen::Triplet<double>;

int main() {
    // A band matrix with 1 on the main diagonal, 2 on the below-main subdiagonal,
    // and 3 on the above-main subdiagonal
    const int n = 10;
    std::vector<T> triplets;
    for (int i = 0; i < n; i++) {
        triplets.emplace_back(i, i, 1.0);
        if (i > 0) {
            triplets.emplace_back(i - 1, i, 3.0);
        }
        if (i < n - 1) {
            triplets.emplace_back(i + 1, i, 2.0);
        }
    }
    Eigen::SparseMatrix<double> M(n, n);
    M.insertFromTriplets(triplets.begin(), triplets.end());

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, 3, 6);

    // Initialize and compute
    eigs.init();
    eigs.compute(SortRule::LargestMagn);

    // Retrieve results
    Eigen::VectorXcd evalues;
    if (eigs.info() == CompInfo::Successful) {
        evalues = eigs.eigenvalues();
        std::cout << "Eigenvalues found:\n" << evalues << std::endl;
    } else {
        std::cout << "Not Found" << std::endl;
    }
    return 0;
}
