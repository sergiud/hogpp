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

"""Compute the minimax polynomial of hogpp::detail::fastAtanUnit().

The polynomial

    atan(x) ~= x + a*x**3 + b*x**5 + c*x**7,  x in [0, 1]

is used by include/hogpp/fastatan.hpp and is computed with the Remez exchange
algorithm. The basis functions x**3, x**5 and x**7 form a 3-dimensional
Chebyshev (Haar) system on (0, 1], so the error of the best approximation
equioscillates at no fewer than 4 points. The exchange assumes exactly 4 and
raises otherwise.

Requires mpmath, e.g. pip install mpmath.
"""

import mpmath as mp

mp.mp.dps = 80

DEGREES = (3, 5, 7)
ALTERNATION_POINTS = len(DEGREES) + 1
MAX_ITERATIONS = 50
# Number of intervals of the grid on which sign changes of the derivative of
# the error locate its extrema.
SEARCH_INTERVALS = 2000
# Number of intervals of the grid on which the maximum error is verified.
VERIFICATION_INTERVALS = 500000
# Largest change of an alternation point between iterations at convergence.
TOLERANCE = mp.mpf('1e-60')


def residual(x):
    """Return the part of atan(x) the polynomial approximates."""
    return mp.atan(x) - x


def basis(x):
    """Return the basis functions of the polynomial evaluated at x."""
    return [x**degree for degree in DEGREES]


def solve_step(points):
    """Solve for the coefficients and the levelled error at the points."""
    n = len(points)
    matrix_a = mp.matrix(n, n)
    rhs = mp.matrix(n, 1)
    for i, x in enumerate(points):
        row = [*basis(x), (-1) ** i]
        for j, v in enumerate(row):
            matrix_a[i, j] = v
        rhs[i] = residual(x)
    sol = mp.lu_solve(matrix_a, rhs)
    return [sol[i] for i in range(n)]


def error_fn(coefficients):
    """Return the approximation error of the polynomial as a function."""
    return lambda x: residual(x) - mp.fdot(coefficients, basis(x))


def find_interior_extrema(f, n=SEARCH_INTERVALS):
    """Return the extrema of f in (0, 1) at sign changes of its derivative."""
    xs = [mp.mpf(i) / n for i in range(1, n)]

    def df(x):
        return mp.diff(f, x)

    vals = [df(x) for x in xs]
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            root = mp.findroot(df, (xs[i] + xs[i + 1]) / 2)
            roots.append(root)
    return roots


def minimax_coefficients():
    """Return the coefficients a, b and c of the minimax polynomial."""
    points = [mp.mpf(i) / ALTERNATION_POINTS for i in range(1, ALTERNATION_POINTS + 1)]

    for _ in range(MAX_ITERATIONS):
        *coefficients, _ = solve_step(points)
        f = error_fn(coefficients)
        new_points = sorted({*find_interior_extrema(f), mp.mpf(1)})
        if len(new_points) != ALTERNATION_POINTS:
            raise RuntimeError(
                f'expected {ALTERNATION_POINTS} alternation points but found '
                f'{len(new_points)}'
            )
        if all(
            abs(new - old) < TOLERANCE
            for new, old in zip(new_points, points, strict=True)
        ):
            points = new_points
            break
        points = new_points
    else:
        raise RuntimeError(
            f'the exchange did not converge within {MAX_ITERATIONS} iterations'
        )

    *coefficients, _ = solve_step(points)
    return coefficients


def main():
    """Run the exchange and print the maximum error and the coefficients."""
    coefficients = minimax_coefficients()
    f = error_fn(coefficients)

    grid = [
        mp.mpf(i) / VERIFICATION_INTERVALS for i in range(VERIFICATION_INTERVALS + 1)
    ]
    max_err = max(abs(f(x)) for x in grid)
    print('max |error| on fine grid:', mp.nstr(max_err, 20))
    for name, coefficient in zip('abc', coefficients, strict=True):
        print(name, '=', repr(float(coefficient)))


if __name__ == '__main__':
    main()
