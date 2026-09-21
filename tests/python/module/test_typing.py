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

import importlib.util
import os
import subprocess
import sys
import textwrap

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    path = os.getenv('HOGPPPATH')
    if path:
        os.add_dll_directory(path)

import pytest


def _package_dir():
    spec = importlib.util.find_spec('hogpp')
    return os.path.dirname(spec.origin)


def test_py_typed_marker_is_shipped():
    # PEP 561 requires an inline-typed package to ship a py.typed marker
    # file next to its top-level module so type checkers pick it up.
    assert os.path.isfile(os.path.join(_package_dir(), 'py.typed'))


def test_stub_files_are_shipped():
    assert os.path.isfile(os.path.join(_package_dir(), '__init__.pyi'))
    assert os.path.isfile(os.path.join(_package_dir(), '_hogpp.pyi'))


def test_stub_type_checks_cleanly(tmp_path):
    pytest.importorskip('mypy')

    script = tmp_path / 'check_typing.py'
    script_text = textwrap.dedent(
        """
        from hogpp import IntegralHOGDescriptor

        desc: IntegralHOGDescriptor = IntegralHOGDescriptor(
            n_bins=9, binning='unsigned', cell_size=(8, 8)
        )
        desc.compute([[0.0]])
        features: object = desc([0, 0, 1, 1])
        ok: bool = bool(desc)
        n_bins: int = desc.n_bins_
        """
    )
    script.write_text(script_text)

    env = dict(os.environ, PYTHONPATH=os.path.dirname(_package_dir()))
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'mypy',
            '--strict',
            '--python-executable',
            sys.executable,
            str(script),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
