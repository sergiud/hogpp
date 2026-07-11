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

#include <nanobind/stl/string.h>

#include "blocknormalizer.hpp"

namespace nanobind::detail {

bool type_caster<BlockNormalizerType>::from_python(
    handle src, std::uint8_t flags, cleanup_list* /*cleanup*/) noexcept
{
    std::string name;

    if (!try_cast(
            src, name,
            (flags & static_cast<std::uint8_t>(cast_flags::convert)) != 0)) {
        return false;
    }

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

handle type_caster<BlockNormalizerType>::from_cpp(BlockNormalizerType in,
                                                  rv_policy /*policy*/,
                                                  cleanup_list* /*cleanup*/)
{
    str result;

    switch (in) {
        case BlockNormalizerType::L1:
            result = str{"l1"};
            break;
        case BlockNormalizerType::L1Hys:
            result = str{"l1-hys"};
            break;
        case BlockNormalizerType::L2:
            result = str{"l2"};
            break;
        case BlockNormalizerType::L2Hys:
            result = str{"l2-hys"};
            break;
        case BlockNormalizerType::L1sqrt:
            result = str{"l1-sqrt"};
            break;
    }

    return result.release();
}

} // namespace nanobind::detail

template class BlockNormalizer<float>;
template class BlockNormalizer<double>;

#include <hogpp/suffix.hpp>
