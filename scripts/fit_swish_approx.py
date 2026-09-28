"""Reproduce the Swish-approximation fits of I-Parakeet (Sec. 3.2, Table 3 "Max err.").

tanh_hat(u; a, c) = sgn(u) * (a * (min(|u|, c) - c)^2 + 1)          (Eq. 12)
sw_hat(x; a, c)   = x * (1 + tanh_hat(x / 2; a, c)) / 2               (Eq. 13)

Four fits are compared: {L_inf, L2} x {Swish output, tanh}, plus Hard-Swish.
The paper reports (a*, c*) = (-0.1240, 2.4632) for the L_inf fit to Swish.
"""

import argparse

import numpy as np
from scipy.optimize import minimize


def tanh_hat(u, a, c):
    return np.sign(u) * (a * (np.minimum(np.abs(u), c) - c) ** 2 + 1.0)


def swish(x):
    return x / (1.0 + np.exp(-x))


def swish_hat(x, a, c):
    return x * (1.0 + tanh_hat(x / 2.0, a, c)) / 2.0


def hard_swish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def fit(loss, x0_list):
    best = None
    for x0 in x0_list:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmax", type=float, default=30.0,
                        help="half-width of the x grid (errors vanish for large |x|)")
    parser.add_argument("--step", type=float, default=1e-3)
    args = parser.parse_args()

    x = np.arange(-args.xmax, args.xmax + args.step, args.step)
    u = x[x >= 0] / 2.0  # tanh is fitted on u >= 0 and mirrored (Eq. 12)
    sw = swish(x)
    th = np.tanh(u)

    def sw_err(p):
        return np.abs(sw - swish_hat(x, *p))

    def th_err(p):
        return np.abs(th - tanh_hat(u, *p))

    starts = [np.array([a, c]) for a in (-0.3, -0.2, -0.12, -0.05) for c in (1.5, 2.0, 2.5, 3.0)]
    fits = {
        "L_inf fit to Swish": fit(lambda p: sw_err(p).max(), starts),
        "L2 fit to Swish": fit(lambda p: np.mean(sw_err(p) ** 2), starts),
        "L_inf fit to tanh": fit(lambda p: th_err(p).max(), starts),
        "L2 fit to tanh": fit(lambda p: np.mean(th_err(p) ** 2), starts),
    }

    print(f"{'configuration':<22} {'a':>9} {'c':>8} {'max|sw err|':>12}")
    for name, (a, c) in fits.items():
        print(f"{name:<22} {a:9.4f} {c:8.4f} {sw_err((a, c)).max():12.4f}")
    print(f"{'Hard-Swish':<22} {'-':>9} {'-':>8} {np.abs(sw - hard_swish(x)).max():12.4f}")
    print(f"{'paper (a*, c*)':<22} {-0.1240:9.4f} {2.4632:8.4f} {sw_err((-0.1240, 2.4632)).max():12.4f}")


if __name__ == "__main__":
    main()
