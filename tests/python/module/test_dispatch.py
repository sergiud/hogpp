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

from _dispatch_isas import available_isas as _available_isas
import os
import pytest
import subprocess
import sys

# HOGPP_DISPATCH is only consulted once while the extension module is being
# initialized. Neither importlib.reload() nor deleting the sys.modules entry
# and re-importing re-runs that initialization in this process: CPython
# caches the initialized extension module (this holds even though pybind11
# uses PEP 489 multi-phase init, m_size=0). Each scenario below therefore
# needs its own interpreter.


@pytest.fixture
def hogpp_import():
    def _run(dispatch):
        env = dict(os.environ)

        if dispatch is None:
            env.pop('HOGPP_DISPATCH', None)
        else:
            env['HOGPP_DISPATCH'] = dispatch

        return subprocess.run(
            [sys.executable, '-c', 'import hogpp'],
            env=env,
            capture_output=True,
            text=True,
        )

    return _run


@pytest.fixture(scope='session')
def available_isas():
    return _available_isas(sys.executable, tuple(os.environ.items()))


def test_invalid_dispatch_reports_error(hogpp_import):
    result = hogpp_import('foobar')

    assert result.returncode != 0
    assert (
        'The instruction set specified by the HOGPP_DISPATCH environment '
        'variable ("foobar") is neither available nor supported.'
    ) in result.stderr


def test_invalid_dispatch_does_not_suggest_unrelated_isa(hogpp_import):
    # A name unrelated to any supported ISA must not receive a "Did you
    # mean" suggestion.
    result = hogpp_import('zzzzzzzzzzzzzzzzzzzz')

    assert result.returncode != 0
    assert 'Did you mean' not in result.stderr


def test_dispatch_suggests_close_match(hogpp_import, available_isas):
    if not available_isas:
        pytest.skip('no additional ISA dispatch compiled in')

    isa = available_isas[0]

    # Corrupt the first character rather than truncating the name: no
    # canonical ISA name starts with 'Z', so this can't accidentally collide
    # with another name that is compiled in but unsupported on this machine
    # (e.g. truncating "AVX2" to "AVX" would hit the exact-match path for the
    # real ISA::AVX entry instead of the fuzzy-suggestion path, since AVX may
    # be compiled in even when this CPU doesn't support it).
    typo = 'Z' + isa[1:]

    result = hogpp_import(typo)

    assert result.returncode != 0
    assert f'Did you mean {isa}?' in result.stderr


def test_empty_dispatch_uses_generic(hogpp_import):
    result = hogpp_import('')

    assert result.returncode == 0, result.stderr


def test_dispatch_with_available_isa_succeeds(hogpp_import, isa):
    result = hogpp_import(isa)

    assert result.returncode == 0, result.stderr


def pytest_generate_tests(metafunc):
    if 'isa' in metafunc.fixturenames:
        metafunc.parametrize(
            'isa', _available_isas(sys.executable, tuple(os.environ.items()))
        )
