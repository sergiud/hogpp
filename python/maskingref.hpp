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

#include <concepts>
#include <type_traits>

#include <Eigen/Core>

#include <hogpp/prefix.hpp>

namespace pyhogpp {

/**
 * @brief Non-owning type-erased reference to a masking callable.
 *
 * hogpp::IntegralHOGDescriptor::compute() is templated on the type of the
 * masking callable, and every lambda expression has a distinct type, even if
 * it is structurally identical to another one. Each masking lambda passed to
 * compute() would therefore instantiate its body again, including the tensor
 * expressions it contains. Wrapping the masking callables in this single type
 * reduces them to one instantiation of compute() per descriptor type.
 *
 * The callable is referenced by a function pointer and a context pointer
 * rather than wrapped in std::function, since it outlives every call of
 * compute() it is passed to. Nothing has to be owned or allocated, which
 * matters for a callable invoked once per pixel.
 */
class MaskingRef
{
public:
    // Constructing from another MaskingRef copies it instead of referring to
    // it, since the other one may not outlive this one.
    template<class F>
        requires(!std::same_as<std::remove_cvref_t<F>, MaskingRef>)
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

} // namespace pyhogpp

#include <hogpp/suffix.hpp>

#endif // PYTHON_HOGPP_MASKINGREF_HPP
