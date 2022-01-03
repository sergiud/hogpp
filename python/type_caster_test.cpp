//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2022 Sergiu Deitsch <sergiu.deitsch@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "type_caster/opencv.hpp"
#include "type_caster/tensor.hpp"

namespace {

template<class T>
T pass(const T& a)
{
    return a;
}

void init_opencv(pybind11::module& m)
{
    // clang-format off
    m.def
    (
        "pass_"
        , &pass<cv::Mat>
    )
    .def
    (
        "pass_bounds_"
        , &pass<cv::Rect>
    )
    .def
    (
        "pass_bounds_"
        , &pass<cv::Rect2f>
    )
    .def
    (
        "pass_bounds_"
        , &pass<cv::Rect2d>
    )
    ;
    // clang-format on
}

template<int Options>
void init_tensor(pybind11::module& m)
{
    // clang-format off
    m.def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 0, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 1, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 2, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 3, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 4, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<float, 5, Options> >
    )
    ;
    // clang-format on

    // clang-format off
    m.def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 0, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 1, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 2, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 3, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 4, Options> >
    )
    .def
    (
        "pass_"
        , &pass<Eigen::Tensor<double, 5, Options> >
    )
    ;
    // clang-format on
}

} // namespace

PYBIND11_MODULE(type_caster_test, m)
{
    auto tensor = m.def_submodule("tensor");
    auto f_style = tensor.def_submodule("f_style");
    auto c_style = tensor.def_submodule("c_style");

    auto opencv = m.def_submodule("opencv");

    init_opencv(opencv);
    init_tensor<Eigen::ColMajor>(f_style);
    init_tensor<Eigen::RowMajor>(c_style);
}
