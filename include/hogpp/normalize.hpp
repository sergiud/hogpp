//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef HOGPP_NORMALIZE_HPP
#define HOGPP_NORMALIZE_HPP

namespace hogpp {

template<class Tensor>
constexpr void normalize(Tensor& block, typename Tensor::Scalar den)
{
    if (den != typename Tensor::Scalar{0}) {
        block = block / den;
    }
}

} // namespace hogpp

#endif // HOGPP_NORMALIZE_HPP
