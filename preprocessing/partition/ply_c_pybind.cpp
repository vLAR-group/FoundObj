// pybind11 wrapper for libply_c (compute_geof)
// Replaces boost.python binding with pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace py = pybind11;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;
typedef Eigen::Matrix<float, 3, 1> Vector3f;

py::array_t<float> compute_geof(
    py::array_t<float, py::array::c_style> xyz,
    py::array_t<uint32_t, py::array::c_style> target,
    int k_nn)
{
    auto xyz_buf = xyz.request();
    auto tgt_buf = target.request();

    const std::size_t n_ver = xyz_buf.shape[0];
    const float* xyz_data = static_cast<float*>(xyz_buf.ptr);
    const uint32_t* target_data = static_cast<uint32_t*>(tgt_buf.ptr);

    // Output: (n_ver, 4) - linearity, planarity, scattering, verticality
    py::array_t<float> result({(int)n_ver, 4});
    auto res_buf = result.request();
    float* geof = static_cast<float*>(res_buf.ptr);

    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++) {
        // Build position matrix: point + k_nn neighbors
        Eigen::MatrixXf position(k_nn + 1, 3);
        position(0, 0) = xyz_data[3 * i_ver];
        position(0, 1) = xyz_data[3 * i_ver + 1];
        position(0, 2) = xyz_data[3 * i_ver + 2];

        std::size_t i_edg = k_nn * i_ver;
        for (std::size_t i_nei = 0; i_nei < (std::size_t)k_nn; i_nei++) {
            std::size_t ind_nei = target_data[i_edg];
            position(i_nei + 1, 0) = xyz_data[3 * ind_nei];
            position(i_nei + 1, 1) = xyz_data[3 * ind_nei + 1];
            position(i_nei + 1, 2) = xyz_data[3 * ind_nei + 2];
            i_edg++;
        }

        // Covariance matrix
        Eigen::MatrixXf centered = position.rowwise() - position.colwise().mean();
        Matrix3f cov = (centered.adjoint() * centered) / float(k_nn + 1);

        // Eigendecomposition
        Eigen::EigenSolver<Matrix3f> es(cov);
        std::vector<float> ev = {
            es.eigenvalues()[0].real(),
            es.eigenvalues()[1].real(),
            es.eigenvalues()[2].real()
        };

        // Sort descending
        std::vector<int> indices = {0, 1, 2};
        std::sort(indices.begin(), indices.end(),
                  [&](int i1, int i2) { return ev[i1] > ev[i2]; });

        std::vector<float> lambda = {
            std::max(ev[indices[0]], 0.f),
            std::max(ev[indices[1]], 0.f),
            std::max(ev[indices[2]], 0.f)
        };

        std::vector<float> v1 = {
            es.eigenvectors().col(indices[0])(0).real(),
            es.eigenvectors().col(indices[0])(1).real(),
            es.eigenvectors().col(indices[0])(2).real()
        };
        std::vector<float> v2 = {
            es.eigenvectors().col(indices[1])(0).real(),
            es.eigenvectors().col(indices[1])(1).real(),
            es.eigenvectors().col(indices[1])(2).real()
        };
        std::vector<float> v3 = {
            es.eigenvectors().col(indices[2])(0).real(),
            es.eigenvectors().col(indices[2])(1).real(),
            es.eigenvectors().col(indices[2])(2).real()
        };

        // Dimensionality features
        float linearity = (sqrtf(lambda[0]) - sqrtf(lambda[1])) / sqrtf(lambda[0]);
        float planarity = (sqrtf(lambda[1]) - sqrtf(lambda[2])) / sqrtf(lambda[0]);
        float scattering = sqrtf(lambda[2]) / sqrtf(lambda[0]);

        // Verticality
        std::vector<float> unary = {
            lambda[0] * fabsf(v1[0]) + lambda[1] * fabsf(v2[0]) + lambda[2] * fabsf(v3[0]),
            lambda[0] * fabsf(v1[1]) + lambda[1] * fabsf(v2[1]) + lambda[2] * fabsf(v3[1]),
            lambda[0] * fabsf(v1[2]) + lambda[1] * fabsf(v2[2]) + lambda[2] * fabsf(v3[2])
        };
        float norm = sqrtf(unary[0]*unary[0] + unary[1]*unary[1] + unary[2]*unary[2]);
        float verticality = unary[2] / norm;

        geof[i_ver * 4 + 0] = linearity;
        geof[i_ver * 4 + 1] = planarity;
        geof[i_ver * 4 + 2] = scattering;
        geof[i_ver * 4 + 3] = verticality;
    }

    return result;
}

PYBIND11_MODULE(libply_c, m) {
    m.def("compute_geof", &compute_geof,
          py::arg("xyz"), py::arg("target"), py::arg("k_nn"));
}
