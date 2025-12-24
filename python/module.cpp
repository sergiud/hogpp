#include <algorithm>

#include "cpufeature.hpp"
#include "cpufeatures.hpp"
#include "module.hpp"

void HOGppModule<ISA::Default>::initialize(pybind11::module& m)
{
    void init_hogpp_default(pybind11::module & m);
    init_hogpp_default(m);
}

namespace {

void supportedCPUFeatureNames(std::vector<std::string_view>& /*names*/,
                              CPUFeatures<> /*unused*/)
{
}

template<ISA Type, ISA... Types>
void supportedCPUFeatureNames(std::vector<std::string_view>& names,
                              CPUFeatures<Type, Types...> /*unused*/)
{
    if constexpr (HOGppModuleSupported<Type>) {
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
