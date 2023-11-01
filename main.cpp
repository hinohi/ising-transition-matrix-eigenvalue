#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>
#include <cmath>

using namespace Spectra;
using T = Eigen::Triplet<double>;

const int N = 3;

int calc_local_energy(uint64_t spin, int pos) {
    int row = pos / N;
    int col = pos % N;
    int energy = -2;
    if (spin & (1 << (row * N + (col + 1) % N))) {
        energy += 1;
    }
    if (spin & (1 << (row * N + (col + N + 1) % N))) {
        energy += 1;
    }
    if (spin & (1 << ((row + 1) % N * N + col))) {
        energy += 1;
    }
    if (spin & (1 << ((row + N + 1) % N * N + col))) {
        energy += 1;
    }
    return energy;
}

/// spin=1,-1
double flip_prob_metropolis(double beta, int local_energy, int spin) {
    int diff = local_energy * spin * 2;
    if (diff < 0) {
        return 1.0;
    } else {
        return std::exp(static_cast<double>(-diff) * beta);
    }
}

void gen_uniform_random_choice_flip_prob(double beta, uint64_t spin, std::vector<T> &triplets,
                                         const std::function<double(double, int, int)>& flip_prob) {
    double stay_prob = 0.0;
    for (int pos = 0; pos < N * N; pos++) {
        auto flipped_spin = spin ^ (1 << pos);
        auto local_energy = calc_local_energy(spin, pos);
        auto p = flip_prob(beta, local_energy, (spin & (1 << pos)) ? 1 : -1);
        triplets.emplace_back(spin, flipped_spin, p / static_cast<double>(N * N));
        stay_prob += 1.0 - p;
    }
    triplets.emplace_back(spin, spin, stay_prob / static_cast<double>(N * N));
}

int main() {
    const double beta = 1.0;
    const uint64_t matrix_size = 1 << (N * N);
    std::vector<T> triplets;
    for (uint64_t spin = 0; spin < matrix_size; spin++) {
        gen_uniform_random_choice_flip_prob(beta, spin, triplets, flip_prob_metropolis);
    }

    Eigen::SparseMatrix<double> M(matrix_size, matrix_size);
    M.insertFromTriplets(triplets.begin(), triplets.end());

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, 2, 6);

    // Initialize and compute
    eigs.init();
    eigs.compute(SortRule::LargestMagn);

    if (eigs.info() == CompInfo::Successful) {
        std::cout << "Eigenvalues found:\n" << eigs.eigenvalues() << std::endl;
    } else {
        std::cout << "Not Found" << std::endl;
    }
    return 0;
}
