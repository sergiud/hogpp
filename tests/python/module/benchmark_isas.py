#!/usr/bin/env python3
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

"""Benchmark perf_dispatch.py once per ISA this build supports (plus
generic), then print one final comparison table with all of them side by
side.

HOGPP_DISPATCH is only consulted once per process, so each ISA needs its own
interpreter: this cannot use pytest-benchmark's in-process parametrization
and instead runs one subprocess per ISA, each via --benchmark-save.
--benchmark-save's own file-naming convention (NNNN_<label>.json) is what
labels each row in `pytest-benchmark compare`'s output: the label is not
otherwise present in a run's JSON content.

The table cannot be built up one row at a time as each ISA finishes: `compare`
recomputes every row's relative percentage against all files given to it, so
adding a row after a new fastest (or slowest) result would retroactively
change rows already shown.

Usage:
  python benchmark_isas.py                  # sweep and print a comparison table
  python benchmark_isas.py --json out.json  # also write one merged JSON file
                                             # (for benchmark-action/github-action-benchmark)
Requires: pip install pytest-benchmark
"""

from _dispatch_isas import available_isas
import argparse
import json
import logging
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_TEST = os.path.join(HERE, 'perf_dispatch.py')

# pytest-benchmark prints this line to stderr after a --benchmark-save run.
SAVED_PATH_RE = re.compile(r'^Saved benchmark data in: (.+)$', re.MULTILINE)

logger = logging.getLogger('benchmark_isas')


def run_benchmark(dispatch, label):
    env = dict(os.environ)
    env['HOGPP_DISPATCH'] = dispatch

    logger.info('Benchmarking %s (HOGPP_DISPATCH=%r)', label, dispatch)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            BENCHMARK_TEST,
            '--benchmark-only',
            '-q',
            f'--benchmark-save={label}',
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    logger.debug('%s', result.stdout)

    match = SAVED_PATH_RE.search(result.stderr)

    if not match:
        raise RuntimeError(
            f"could not find pytest-benchmark's saved file path for {label!r} "
            'in its output'
        )

    return match.group(1)


def print_table(paths):
    # Left to inherit stdout/stderr rather than captured: pytest-benchmark
    # itself builds and formats the table, this just lets it print normally.
    subprocess.run(
        [sys.executable, '-m', 'pytest_benchmark', 'compare'] + paths,
        check=True,
    )


def merge(paths, labels, output_path):
    combined = None

    for path, label in zip(paths, labels):
        with open(path) as f:
            data = json.load(f)

        # Disambiguate each ISA's entry: every run otherwise reports the
        # same "test_compute" name, and a tool tracking these over time
        # would not be able to tell them apart.
        for benchmark in data['benchmarks']:
            benchmark['name'] = f"{benchmark['name']}[{label}]"
            benchmark['fullname'] = f"{benchmark['fullname']}[{label}]"
            benchmark['group'] = label

        if combined is None:
            combined = data
        else:
            combined['benchmarks'].extend(data['benchmarks'])

    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json',
        metavar='PATH',
        help='also write one merged pytest-benchmark JSON file to PATH',
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="show each ISA's own pytest-benchmark output as it runs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s',
    )

    labels = ['generic'] + list(
        available_isas(sys.executable, tuple(os.environ.items()))
    )
    paths = []

    for label in labels:
        paths.append(run_benchmark('' if label == 'generic' else label, label))

    print_table(paths)

    if args.json:
        merge(paths, labels, args.json)
        logger.info('Wrote merged results to %s', args.json)


if __name__ == '__main__':
    main()
