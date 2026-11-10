#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 PROJECT 10 - PHYSICS: MAXWELL-BOLTZMANN VELOCITY DISTRIBUTION
 MODULE: Thành viên 3 - Tích phân xác định & Kiểm tra dữ liệu vào
=====================================================================
"""

from __future__ import annotations

import argparse
import math
import numbers
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

# -----------------------------------------------------------------
# 1. Hằng số vật lý (NIST CODATA 2022 / SI 2019, giá trị chính xác)
# -----------------------------------------------------------------
K_B = 1.380649e-23           # J/K
N_A = 6.02214076e23          # 1/mol
C_LIGHT = 299_792_458.0      # m/s

# Khối lượng mol [g/mol] theo IUPAC 2021
MOLAR_MASS_G_PER_MOL = {
    "He": 4.002602,
    "N2": 2 * 14.007,
    "O2": 2 * 15.999,
    "Ar": 39.95,
}

# Tham số thuật toán
TAIL_CUT   = 10.0            # Cắt đuôi tại v = TAIL_CUT·a
DEFAULT_TOL = 1e-10          # Sai số tuyệt đối cho xác suất P
INITIAL_N  = 200             # Số khoảng Simpson ban đầu (phải chẵn)
MAX_N      = 2_000_000       # Giới hạn trên số khoảng

# -----------------------------------------------------------------
# 2. Ngoại lệ dùng chung
# -----------------------------------------------------------------
class InputValidationError(ValueError):
    """Dữ liệu đầu vào không hợp lệ (kế thừa ValueError theo docx)."""
    pass

# -----------------------------------------------------------------
# 3. Kiểm tra dữ liệu đầu vào (Theo Chương 1.1 & 1.2 của Docx)
# -----------------------------------------------------------------
def _as_finite_real(value, name: str) -> float:
    """Ép về float; từ chối bool / non-number / NaN / ±Inf."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not bool ({value!r}).")
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r} ({type(value).__name__}).")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite (no NaN/Inf), got {x}.")
    return x

def _positive_real(value, name: str) -> float:
    x = _as_finite_real(value, name)
    if x <= 0.0:
        raise ValueError(f"{name} must be greater than 0, got {x}.")
    return x

def validate_inputs(T, M, v1, v2, N=None) -> tuple:
    """
    Step 1: Check data types and finite numerical properties.
    Step 2: Validate positive physical value domains.
    Step 3: Normalize integration boundaries (swap, degenerate case, N adjustment).
    Step 4: Return cleaned and normalized dataset (T, M, v1, v2, N).
    """
    # Step 1: Data types & Finite check
    T_f  = _as_finite_real(T, "T")
    M_f  = _as_finite_real(M, "M")
    v1_f = _as_finite_real(v1, "v1")
    v2_f = _as_finite_real(v2, "v2")

    # Step 2: Physical domain checks
    if T_f <= 0.0:
        raise ValueError('Absolute temperature T must be greater than 0 K')
    if M_f <= 0.0:
        raise ValueError('Molar mass M must be greater than 0 kg/mol')
    if v1_f < 0.0 or v2_f < 0.0:
        raise ValueError('Velocity bounds must be non-negative')

    # Step 3: Normalize
    if v1_f > v2_f:
        warnings.warn(f"Automatically swapped v1 and v2: ({v1_f}, {v2_f}) -> ({v2_f}, {v1_f})")
        v1_f, v2_f = v2_f, v1_f
        
    if N is not None:
        if not isinstance(N, int) or N < 2:
            N = 2
        if N % 2 != 0:  # Tự động nâng lên số chẵn gần nhất
            N = N + 1
            
    # Step 4: Return
    return T_f, M_f, v1_f, v2_f, N

def validate_tolerance(tol) -> float:
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
        raise ValueError(f"tol must be a real number, got {tol!r}.")
    t = float(tol)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"tol must be finite and > 0, got {t}.")
    return t

# -----------------------------------------------------------------
# 4. Chuyển đổi khối lượng
# -----------------------------------------------------------------
def mass_from_molar(M) -> float:
    """M [kg/mol] → m [kg] của một phân tử."""
    return _positive_real(M, "Molar mass M") / N_A

def mass_from_gas(gas: str) -> float:
    """Tra khối lượng phân tử [kg] từ tên khí."""
    if not isinstance(gas, str):
        raise ValueError("Gas name must be a string.")
    key = gas.strip()
    if key not in MOLAR_MASS_G_PER_MOL:
        raise ValueError(f"Gas '{gas}' not supported. Options: {sorted(MOLAR_MASS_G_PER_MOL)}.")
    return MOLAR_MASS_G_PER_MOL[key] * 1e-3 / N_A

# -----------------------------------------------------------------
# 5. Hàm phân bố Maxwell–Boltzmann
# -----------------------------------------------------------------
def scale_parameter(T: float, m: float) -> float:
    """a = sqrt(k_B T / m) [m/s]"""
    ratio = K_B * T / m
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise InputValidationError(
            f"Scale parameter a = sqrt(k_B·T/m) invalid (T = {T}, m = {m})."
        )
    a = math.sqrt(ratio)
    if a <= 0.0 or not math.isfinite(a):
        raise InputValidationError("Scale parameter a is invalid.")
    return a

def maxwell_boltzmann_pdf(v: float, T: float, m: float) -> float:
    """f(v) evaluated in log-space."""
    if v <= 0.0:
        return 0.0
    a = scale_parameter(T, m)
    try:
        ratio = v / a
        log_val = (
            0.5 * math.log(2.0 / math.pi)
            + 2.0 * math.log(v)
            - 3.0 * math.log(a)
            - 0.5 * ratio * ratio
        )
    except (ValueError, OverflowError):
        return 0.0
    if log_val < -745.0:
        return 0.0
    if log_val > 700.0:
        return math.exp(700.0)
    return math.exp(log_val)

def _erf_cdf(x: float) -> float:
    """F(x) = erf(x/√2) − √(2/π)·x·exp(-x²/2)."""
    if x <= 0.0:
        return 0.0
    if x < 1.0e-4:  # Taylor branch for small x
        return math.sqrt(2.0 / math.pi) * x ** 3 / 3.0
    if x > 30.0:    # Saturation branch
        return 1.0
    val = (
        math.erf(x / math.sqrt(2.0))
        - math.sqrt(2.0 / math.pi) * x * math.exp(-0.5 * x * x)
    )
    return max(0.0, min(1.0, val))

def analytic_cdf(v: float, T: float, m: float) -> float:
    if v <= 0.0:
        return 0.0
    return _erf_cdf(v / scale_parameter(T, m))

def analytic_probability(v1, v2, T, m) -> float:
    T, m, v1, v2, _ = validate_inputs(T, m, v1, v2)
    if v1 == v2:
        return 0.0
    p = analytic_cdf(v2, T, m) - analytic_cdf(v1, T, m)
    return max(0.0, min(1.0, p))

# -----------------------------------------------------------------
# 6. Tích phân số: Simpson 1/3 ghép
# -----------------------------------------------------------------
def simpson_composite(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Quy tắc Simpson 1/3 ghép, tự động điều chỉnh nếu n lẻ hoặc n < 2."""
    if not isinstance(n, int):
        n = int(n)
    if n < 2:
        n = 2
    if n % 2 != 0:
        n += 1
        
    h = (b - a) / n
    s = f(a) + f(b)
    
    # Tách vòng lặp để tránh rẽ nhánh if/else trong vòng lặp (theo Docx)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * f(a + i * h)
        
    return s * h / 3.0

@dataclass(frozen=True)
class IntegrationResult:
    probability: float
    n_intervals: int
    error_estimate: float
    v_upper_used: float
    tail_truncated: bool
    converged: bool

def probability_numeric(v1, v2, T, m, tol: float = DEFAULT_TOL, n: Optional[int] = None) -> IntegrationResult:
    T, m, v1, v2, n = validate_inputs(T, m, v1, v2, n)
    tol = validate_tolerance(tol)
    
    if v1 == v2:
        return IntegrationResult(0.0, 0, 0.0, v1, False, True)

    a_scale = scale_parameter(T, m)
    v_cut = TAIL_CUT * a_scale

    if v1 >= v_cut:
        return IntegrationResult(0.0, 0, 0.0, v1, True, True)

    truncated = v2 > v_cut
    v_hi = min(v2, v_cut)

    def f(u):
        return maxwell_boltzmann_pdf(u, T, m)

    if n is not None:  # Fixed-n mode
        val = simpson_composite(f, v1, v_hi, n)
        return IntegrationResult(
            probability=max(0.0, min(1.0, val)),
            n_intervals=n,
            error_estimate=float("nan"),
            v_upper_used=v_hi,
            tail_truncated=truncated,
            converged=True,
        )

    # Adaptive Mode
    n_cur = INITIAL_N
    prev = simpson_composite(f, v1, v_hi, n_cur)
    err = float("inf")
    converged = False
    
    while n_cur < MAX_N:
        n_cur *= 2
        cur = simpson_composite(f, v1, v_hi, n_cur)
        err = abs(cur - prev) / 15.0
        if err <= tol:
            prev = cur
            converged = True
            break
        prev = cur

    return IntegrationResult(
        probability=max(0.0, min(1.0, prev)),
        n_intervals=n_cur,
        error_estimate=err,
        v_upper_used=v_hi,
        tail_truncated=truncated,
        converged=converged,
    )

# -----------------------------------------------------------------
# 7. API tiện dụng
# -----------------------------------------------------------------
def probability(v1, v2, T, m, method: str = "analytic", **kwargs) -> float:
    if method == "analytic":
        return analytic_probability(v1, v2, T, m)
    if method == "numeric":
        return probability_numeric(v1, v2, T, m, **kwargs).probability
    raise InputValidationError(f"method must be 'analytic' or 'numeric', got {method!r}.")

def probability_gas(gas, T, v1, v2, method="analytic", **kwargs) -> float:
    return probability(v1, v2, T, mass_from_gas(gas), method=method, **kwargs)

# -----------------------------------------------------------------
# 8. Tự kiểm chứng (dùng cho --check - Cập nhật theo Docx Ch. 3.2)
# -----------------------------------------------------------------
def self_check() -> bool:
    print("=" * 72)
    print("SELF-CHECK: analytic (erf) vs numeric (Simpson adaptive)")
    print("=" * 72)
    
    # 1. Empirical Test Case from Docx: N2, T=300, v_mp=422.06
    print("DOCX TEST CASE (Chapter 3.2):")
    m_doc = 28.0134e-3 / N_A  # M = 28.0134 g/mol
    p_exact = analytic_probability(0.0, 422.06, 300.0, m_doc)
    print(f"  Exact P (analytic): {p_exact:.9f} (Target: 0.427593576)")
    
    for n in [100, 1000, 10000]:
        res = probability_numeric(0.0, 422.06, 300.0, m_doc, n=n)
        print(f"  N={n:<6}: P={res.probability:.9f} | Err: {abs(p_exact-res.probability):.2e}")
    
    res_adapt = probability_numeric(0.0, 422.06, 300.0, m_doc)
    print(f"  Adaptive : P={res_adapt.probability:.9f} | Err: {abs(p_exact-res_adapt.probability):.2e}")
    print("-" * 72)

    # 2. General test cases
    cases = [
        ("N2", 300.0,   0.0,  500.0),
        ("N2", 300.0, 100.0,  600.0),
        ("O2", 300.0, 300.0,  700.0),
        ("He", 300.0, 500.0, 2500.0),
        ("Ar", 100.0,   0.0,  300.0),
    ]
    ok = True
    for gas, T, v1, v2 in cases:
        m = mass_from_gas(gas)
        pa = analytic_probability(v1, v2, T, m)
        res = probability_numeric(v1, v2, T, m)
        diff = abs(pa - res.probability)
        good = diff < 1e-8 and res.converged
        if not good:
            ok = False
        print(f"{gas:<5}{T:>8.0f}{v1:>8.0f}{v2:>8.0f}"
              f"{pa:>16.10f}{res.probability:>16.10f}{diff:>12.2e}"
              f"  {'OK' if good else 'FAIL'}")
    print("-" * 72)
    print("KẾT QUẢ:", "PASS" if ok else "FAIL")
    return ok

def demo():
    print("=" * 72)
    print("DEMO: P(v1 ≤ v ≤ v2) — Maxwell–Boltzmann")
    print("=" * 72)
    cases = [
        ("N2", 300.0,   0.0,  500.0),
        ("He", 300.0,   0.0, 1500.0),
        ("O2", 300.0, 300.0,  800.0),
    ]
    for gas, T, v1, v2 in cases:
        m = mass_from_gas(gas)
        pa = analytic_probability(v1, v2, T, m)
        pn = probability_numeric(v1, v2, T, m).probability
        print(f"{gas:<5} T={T:>3.0f}K | v=[{v1:>4.0f}, {v2:>4.0f}] | P_analytic={pa:.6f} | P_numeric={pn:.6f}")

# -----------------------------------------------------------------
# 9. CLI
# -----------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Xác suất P(v1 ≤ v ≤ v2) cho phân bố tốc độ Maxwell–Boltzmann."
    )
    mg = p.add_mutually_exclusive_group(required=False)
    mg.add_argument("--gas", choices=sorted(MOLAR_MASS_G_PER_MOL), help="Tên khí: He, N2, O2, Ar.")
    mg.add_argument("--mass", type=float, help="Khối lượng một phân tử [kg].")
    mg.add_argument("--molar-mass", type=float, help="Khối lượng mol [kg/mol].")
    p.add_argument("--T",  type=float, help="Nhiệt độ tuyệt đối [K].")
    p.add_argument("--v1", type=float, help="Cận dưới [m/s].")
    p.add_argument("--v2", type=float, help="Cận trên [m/s].")
    p.add_argument("--method", choices=["analytic", "numeric", "both"], default="both")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--n", type=int, default=None, help="Số khoảng Simpson cố định.")
    p.add_argument("--demo",  action="store_true", help="Chạy ví dụ.")
    p.add_argument("--check", action="store_true", help="Tự kiểm chứng.")
    return p

def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.demo:
        demo()
        return 0
    if args.check:
        return 0 if self_check() else 1

    try:
        provided = sum(x is not None for x in (args.gas, args.mass, args.molar_mass))
        if provided != 1:
            raise ValueError("Need exactly one of --gas / --mass / --molar-mass.")
        if args.T is None or args.v1 is None or args.v2 is None:
            raise ValueError("Need --T, --v1, --v2.")

        if args.gas is not None:
            m = mass_from_gas(args.gas)
            label = args.gas
        elif args.mass is not None:
            m = _positive_real(args.mass, "particle mass m")
            label = "(mass given)"
        else:
            m = mass_from_molar(args.molar_mass)
            label = "(molar mass given)"

        # Validation trả về bộ 5 giá trị
        T, m, v1, v2, _ = validate_inputs(args.T, m, args.v1, args.v2)

        print(f"Gas / source      : {label}")
        print(f"Particle mass m   : {m:.6e} kg")
        print(f"T                 : {T} K")
        print(f"Velocity interval : [{v1}, {v2}] m/s")

        # Tính toán các đặc trưng vận tốc theo Flowchart (Hình 3.1)
        a = scale_parameter(T, m)
        v_mp = a * math.sqrt(2.0)
        v_avg = a * math.sqrt(8.0 / math.pi)
        v_rms = a * math.sqrt(3.0)
        
        print(f"v_mp (most prob.) : {v_mp:.4f} m/s")
        print(f"v_avg (average)   : {v_avg:.4f} m/s")
        print(f"v_rms (RMS)       : {v_rms:.4f} m/s")
        print("-" * 30)

        pa = None
        if args.method in ("analytic", "both"):
            pa = analytic_probability(v1, v2, T, m)
            print(f"P (analytic/erf)  : {pa:.12f}  ({pa*100:.6f} %)")

        if args.method in ("numeric", "both"):
            res = probability_numeric(v1, v2, T, m, tol=args.tol, n=args.n)
            print(f"P (numeric)       : {res.probability:.12f}  ({res.probability*100:.6f} %)")
            print(f"  n_intervals     : {res.n_intervals}")
            print(f"  error estimate  : {res.error_estimate:.2e}")
            print(f"  converged       : {res.converged}")
            print(f"  tail truncated  : {res.tail_truncated}")
            if pa is not None:
                print(f"|ΔP|              : {abs(pa - res.probability):.2e}")

        return 0

    except ValueError as exc:
        print(f"[INPUT ERROR] {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
        print()
        self_check()
    else:
        sys.exit(main())