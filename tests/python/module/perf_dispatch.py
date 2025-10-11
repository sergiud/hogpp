# HOGpp - Fast histogram of oriented gradients computation using integral
# histograms
#
# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deliberately not named test_*, so plain `pytest tests/python/module` (used
# by both CTest's python_hogpp target and cibuildwheel's test-command) does
# not pick this up: benchmarks are slow and their timings are not a
# pass/fail signal. Run this file explicitly, one HOGPP_DISPATCH value at a
# time, via benchmark_isas.py in this directory.
#
# Requires the pytest-benchmark package (not a project dependency, since
# only benchmark_isas.py needs it): pip install pytest-benchmark

import numpy as np
import os

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

from hogpp import IntegralHOGDescriptor


def test_compute(benchmark):
    rng = np.random.default_rng(0)
    image = rng.random((512, 512), dtype=np.float32)
    descriptor = IntegralHOGDescriptor()

    benchmark(descriptor.compute, image)
