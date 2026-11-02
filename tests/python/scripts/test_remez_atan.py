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
import pathlib
import pytest
import re

pytest.importorskip('mpmath')

ROOT = pathlib.Path(__file__).resolve().parents[3]
HEADER = ROOT / 'include' / 'hogpp' / 'fastatan.hpp'


@pytest.fixture(scope='module')
def remez_atan():
    spec = importlib.util.spec_from_file_location(
        'remez_atan', ROOT / 'scripts' / 'remez_atan.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The widest long double format is IEEE 754 binary128, whose significand has
# this many bits.
LONG_DOUBLE_DIGITS = 113


def _header_coefficients():
    text = HEADER.read_text()
    return [
        re.search(
            rf'fastAtanDegree{degree} =\s*static_cast<T>\(([^)]+?)L\)', text
        ).group(1)
        for degree in (3, 5, 7)
    ]


# The coefficients compiled into the library must carry the minimax
# coefficients to the precision of every long double format.
def test_header_coefficients_are_long_double_accurate(remez_atan):
    mp = remez_atan.mp
    exact = remez_atan.minimax_coefficients()

    for literal, coefficient in zip(_header_coefficients(), exact, strict=True):
        assert abs(mp.mpf(literal) - coefficient) <= abs(coefficient) * mp.ldexp(
            1, -LONG_DOUBLE_DIGITS
        ), literal


# The script must print the coefficients exactly as the header spells them.
def test_coefficients_match_header(remez_atan, capsys):
    remez_atan.main()

    output = capsys.readouterr().out
    derived = [
        re.search(rf'^{name} = (\S+)$', output, re.MULTILINE).group(1)
        for name in ('a', 'b', 'c')
    ]

    assert derived == _header_coefficients()


# A failed exchange must state how many alternation points it found.
def test_alternation_error_reports_count(remez_atan, monkeypatch):
    monkeypatch.setattr(
        remez_atan,
        'find_interior_extrema',
        lambda _: [remez_atan.mp.mpf('0.5'), remez_atan.mp.mpf('0.75')],
    )

    with pytest.raises(RuntimeError, match=r'found 3'):
        remez_atan.main()
