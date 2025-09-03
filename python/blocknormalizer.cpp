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

#include "blocknormalizer.hpp"

namespace pybind11::detail {

bool type_caster<BlockNormalizerType>::load(handle src, bool /*unused*/)
{
    auto name = pybind11::cast<std::string>(src);

    if (name == "l1") {
        value = BlockNormalizerType::L1;
    }
    else if (name == "l1-hys") {
        value = BlockNormalizerType::L1Hys;
    }
    else if (name == "l2") {
        value = BlockNormalizerType::L2;
    }
    else if (name == "l1-sqrt") {
        value = BlockNormalizerType::L1sqrt;
    }
    else if (name == "l2-hys") {
        value = BlockNormalizerType::L2Hys;
    }
    else {
        return false;
    }

    return true;
}

handle type_caster<BlockNormalizerType>::cast(BlockNormalizerType in,
                                              return_value_policy /*policy*/,
                                              handle /*parent*/)
{
    str result;

    switch (in) {
        case BlockNormalizerType::L1:
            result = "l1";
            break;
        case BlockNormalizerType::L1Hys:
            result = "l1-hys";
            break;
        case BlockNormalizerType::L2:
            result = "l2";
            break;
        case BlockNormalizerType::L2Hys:
            result = "l2-hys";
            break;
        case BlockNormalizerType::L1sqrt:
            result = "l1-sqrt";
            break;
    }

    return result.release();
}

} // namespace pybind11::detail

template class BlockNormalizer<float>;
template class BlockNormalizer<double>;

#include <hogpp/suffix.hpp>
