// pybind11 wrapper for libcp (cut-pursuit)
// Replaces boost.python binding with pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "include/API.h"

namespace py = pybind11;

py::tuple cutpursuit(
    py::array_t<float, py::array::c_style> obs,
    py::array_t<uint32_t, py::array::c_style> source,
    py::array_t<uint32_t, py::array::c_style> target,
    py::array_t<float, py::array::c_style> edge_weight,
    float lambda,
    int cutoff = 0,
    int spatial = 0,
    float weight_decay = 1.0f)
{
    srand(0);

    auto obs_buf = obs.request();
    auto src_buf = source.request();
    auto tgt_buf = target.request();
    auto ew_buf = edge_weight.request();

    const uint32_t n_ver = obs_buf.shape[0];
    const uint32_t n_edg = src_buf.shape[0];
    const uint32_t n_obs = obs_buf.shape[1];

    const float* obs_data = static_cast<float*>(obs_buf.ptr);
    const uint32_t* source_data = static_cast<uint32_t*>(src_buf.ptr);
    const uint32_t* target_data = static_cast<uint32_t*>(tgt_buf.ptr);
    const float* edge_weight_data = static_cast<float*>(ew_buf.ptr);

    std::vector<float> solution(n_ver * n_obs);
    std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver, 0);
    std::vector<std::vector<uint32_t>> components(1, std::vector<uint32_t>(1, 0));

    if (spatial == 0) {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data,
            target_data, edge_weight_data, &node_weight[0],
            solution.data(), in_component, components,
            lambda, (uint32_t)cutoff, 1.f, 4.f, weight_decay, 0.f);
    } else {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data,
            target_data, edge_weight_data, &node_weight[0],
            solution.data(), in_component, components,
            lambda, (uint32_t)cutoff, 2.f, 4.f, weight_decay, 0.f);
    }

    // Convert in_component to numpy array
    py::array_t<uint32_t> py_in_component(n_ver);
    auto ic_buf = py_in_component.request();
    memcpy(ic_buf.ptr, in_component.data(), n_ver * sizeof(uint32_t));

    return py::make_tuple(components, py_in_component);
}

PYBIND11_MODULE(libcp, m) {
    m.def("cutpursuit", &cutpursuit,
          py::arg("obs"),
          py::arg("source"),
          py::arg("target"),
          py::arg("edge_weight"),
          py::arg("lambda"),
          py::arg("cutoff") = 0,
          py::arg("spatial") = 0,
          py::arg("weight_decay") = 1.0f);
}
