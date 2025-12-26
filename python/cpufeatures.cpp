#include <algorithm>

#include "cpufeature.hpp"
#include "cpufeatures.hpp"
#include "moduledispatch.hpp"

namespace pyhogpp {
namespace {

void supportedCPUFeatureNames(std::vector<std::string_view>& /*names*/,
                              CPUFeatures<> /*unused*/)
{
}

template<ISA Type, ISA... Types>
void supportedCPUFeatureNames(std::vector<std::string_view>& names,
                              CPUFeatures<Type, Types...> /*unused*/)
{
    if constexpr (ModuleDispatchSupported<Type>) {
        if (CPUFeature<Type>::supported()) {
            names.push_back(CPUFeature<Type>::name());
        }
    }

    supportedCPUFeatureNames(names, CPUFeatures<Types...>{});
}

} // namespace

std::vector<std::string_view> supportedCPUFeatureNames()
{
    std::vector<std::string_view> names;
    supportedCPUFeatureNames(names, AvailableCPUFeatures{});
    std::ranges::sort(names);
    return names;
}

} // namespace pyhogpp
