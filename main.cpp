#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace Spectra;
using T = Eigen::Triplet<double>;

const int N = 4;
const uint64_t MATRIX_SIZE = 1 << (N * N);

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
    if (diff <= 0) {
        return 1.0;
    } else {
        return std::exp(static_cast<double>(-diff) * beta);
    }
}

void gen_uniform_random_choice_flip_prob(double beta, uint64_t spin, std::vector<T> &triplets,
                                         const std::function<double(double, int, int)> &flip_prob) {
    double stay_prob = 0.0;
    for (int pos = 0; pos < N * N; pos++) {
        auto flipped_spin = spin ^ (1 << pos);
        auto local_energy = calc_local_energy(spin, pos);
        auto p = flip_prob(beta, local_energy, (spin & (1 << pos)) ? 1 : -1);
        triplets.emplace_back(flipped_spin, spin, p / static_cast<double>(N * N));
        stay_prob += 1.0 - p;
    }
    triplets.emplace_back(spin, spin, stay_prob / static_cast<double>(N * N));
}

std::vector<double> calc_second_eigenvalue(double beta) {
    std::vector<T> triplets;
    for (uint64_t spin = 0; spin < MATRIX_SIZE; spin++) {
        gen_uniform_random_choice_flip_prob(beta, spin, triplets, flip_prob_metropolis);
    }

    Eigen::SparseMatrix<double> mat(MATRIX_SIZE, MATRIX_SIZE);
    mat.insertFromTriplets(triplets.begin(), triplets.end());

    SparseGenMatProd<double> op(mat);
    GenEigsSolver<SparseGenMatProd<double>> eigs(op, 4, 10);
    eigs.init();
    eigs.compute(SortRule::LargestMagn);
    assert(eigs.info() == CompInfo::Successful);
    auto eigenvalues = eigs.eigenvalues();
    assert(std::abs(eigenvalues[0].real() - 1.0) < 1e-6);
    assert(std::abs(eigenvalues[0].imag()) < 1e-6);
    return {
            std::abs(eigenvalues[1]),
            std::abs(eigenvalues[2]),
            std::abs(eigenvalues[3]),
    };
}

int main() {
    const int temperature_step = 64;
    for (int ti = 1; ti < 10 * temperature_step; ti++) {
        double t = static_cast<double>(ti) / static_cast<double>(temperature_step);
        double beta = 1.0 / t;
        auto e = calc_second_eigenvalue(beta);
        std::cout << t << " " << e[0] << " " << e[1] << " " << e[2] << std::endl;
    }
    return 0;
}
