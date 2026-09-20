//
// HOGpp - Fast histogram of oriented gradients computation using integral
// histograms
//
// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
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

#ifndef PYTHON_HOGPP_MASKINGREF_HPP
#define PYTHON_HOGPP_MASKINGREF_HPP

#include <Eigen/Core>

#include <hogpp/prefix.hpp>

namespace pyhogpp::inline HOGPP_TARGET {

/**
 * @brief Non-owning type-erased reference to a masking callable.
 *
 * hogpp::IntegralHOGDescriptor::compute() is templated on the masking
 * callable's exact type, so every distinct lambda expression passed to it
 * (even ones that are structurally identical, since each lambda
 * expression mints its own closure type) forces a separate instantiation
 * of compute()'s body, including the tensor-expression code it contains.
 * Wrapping every masking callable used across this file's call sites in
 * this single concrete type instead collapses them all to one
 * instantiation, which is what actually reduces the debug info bloat:
 * moving that one instantiation to a dedicated translation unit only
 * helps once there is a single (or few) instantiation to move there.
 *
 * A plain function pointer plus a void* context, rather than
 * std::function, since the referenced callable's lifetime always spans
 * exactly the enclosing compute() call: no ownership, copyability or
 * allocation is ever needed, and this is invoked once per pixel.
 */
class MaskingRef
{
public:
    template<class F>
    explicit MaskingRef(F& f) noexcept
        : object_{&f}
        , invoke_{[](const void* object, Eigen::DenseIndex i,
                     Eigen::DenseIndex j) {
            return static_cast<bool>((*static_cast<const F*>(object))(i, j));
        }}
    {
    }

    [[nodiscard]] bool operator()(Eigen::DenseIndex i,
                                  Eigen::DenseIndex j) const
    {
        return invoke_(object_, i, j);
    }

private:
    const void* object_;
    bool (*invoke_)(const void*, Eigen::DenseIndex, Eigen::DenseIndex);
};

} // namespace pyhogpp::inline HOGPP_TARGET

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_MASKINGREF_HPP
