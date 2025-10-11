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

"""Shared helper for enumerating the ISAs a built hogpp extension supports on
the current machine. Used by test_dispatch.py and benchmark_isas.py.
"""

import functools
import re
import subprocess


@functools.cache
def available_isas(python_executable, env_items):
    """Returns the ISA names hogpp reports as supported, by parsing the
    error message produced when HOGPP_DISPATCH is set to an unknown value.
    python_executable and env_items (a hashable tuple of os.environ.items())
    are explicit, cacheable parameters rather than implicit globals, since
    callers each spawn their own subprocess to query this.
    """
    env = dict(env_items)
    env['HOGPP_DISPATCH'] = 'invalid'

    result = subprocess.run(
        [python_executable, '-c', 'import hogpp'],
        env=env,
        capture_output=True,
        text=True,
    )

    match = re.search(
        r'The following CPU features are supported: ([.A-Z0-9, ]*)\.',
        result.stderr,
    )

    if not match:
        return ()

    return tuple(name.strip() for name in match.group(1).split(',') if name.strip())
