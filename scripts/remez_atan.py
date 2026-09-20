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

# Computes the minimax odd-polynomial approximation
#
#     atan(x) ~= x + a*x**3 + b*x**5 + c*x**7,  x in [0, 1]
#
# used by hogpp::detail::fastAtanUnit() (include/hogpp/fastatan.hpp), via
# the Remez exchange algorithm. The basis functions {x**3, x**5, x**7} form
# a 3-dimensional Chebyshev (Haar) system on (0, 1], so the best
# approximation from this space equioscillates at exactly 4 points.
#
# Requires: pip install mpmath

import mpmath as mp

mp.mp.dps = 80


def residual(x):
    return mp.atan(x) - x


def basis(x):
    return [x**3, x**5, x**7]


def solve_step(points):
    n = len(points)
    matrix_a = mp.matrix(n, n)
    rhs = mp.matrix(n, 1)
    for i, x in enumerate(points):
        row = basis(x) + [(-1) ** i]
        for j, v in enumerate(row):
            matrix_a[i, j] = v
        rhs[i] = residual(x)
    sol = mp.lu_solve(matrix_a, rhs)
    return sol[0], sol[1], sol[2], sol[3]


def error_fn(a, b, c):
    return lambda x: residual(x) - (a * x**3 + b * x**5 + c * x**7)


def find_interior_extrema(f, n=2000):
    xs = [mp.mpf(i) / n for i in range(1, n)]
    df = lambda x: mp.diff(f, x)
    vals = [df(x) for x in xs]
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            root = mp.findroot(df, (xs[i] + xs[i + 1]) / 2)
            roots.append(root)
    return roots


def main():
    points = [mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('0.75'), mp.mpf('1.0')]

    for iteration in range(50):
        a, b, c, unused_e = solve_step(points)
        f = error_fn(a, b, c)
        interior = find_interior_extrema(f)
        new_points = sorted(set(interior + [mp.mpf(1)]))
        if len(new_points) != 4:
            raise RuntimeError('expected 4 alternation points')
        if all(abs(new_points[i] - points[i]) < mp.mpf('1e-60')
               for i in range(4)):
            points = new_points
            break
        points = new_points

    a, b, c, equiosc_e = solve_step(points)
    f = error_fn(a, b, c)

    grid = [mp.mpf(i) / 500000 for i in range(0, 500001)]
    max_err = max(abs(f(x)) for x in grid)
    print('max |error| on fine grid:', mp.nstr(max_err, 20))
    print('a =', repr(float(a)))
    print('b =', repr(float(b)))
    print('c =', repr(float(c)))


if __name__ == '__main__':
    main()
