//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2025 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#include <hogpp/prefix.hpp>

#include "binning.hpp"

namespace pybind11::detail {

bool type_caster<BinningType>::load(handle src, bool /*unused*/)
{
    auto name = pybind11::cast<std::string>(src);

    if (name == "signed") {
        value = BinningType::Signed;
    }
    else if (name == "unsigned") {
        value = BinningType::Unsigned;
    }
    else {
        return false;
    }

    return true;
}

handle type_caster<BinningType>::cast(BinningType in,
                                      return_value_policy /*policy*/,
                                      handle /*parent*/)
{
    str result;

    switch (in) {
        case BinningType::Signed:
            result = "signed";
            break;
        case BinningType::Unsigned:
            result = "unsigned";
            break;
    }

    return result.release();
}

} // namespace pybind11::detail

template class Binning<float>;
template class Binning<double>;

#include <hogpp/suffix.hpp>
