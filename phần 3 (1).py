#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 PROJECT 10 - PHYSICS: MAXWELL-BOLTZMANN VELOCITY DISTRIBUTION
 MODULE: Thành viên 3 - Tích phân xác định & Kiểm tra dữ liệu vào
=====================================================================

Bài toán:
    Tính xác suất  P(v1 ≤ v ≤ v2) = ∫_{v1}^{v2} f(v) dv,
    trong đó f(v) là hàm mật độ xác suất tốc độ Maxwell–Boltzmann:
        f(v) = 4π (m/(2π k_B T))^{3/2} v² exp(-m v²/(2 k_B T)),  v ≥ 0
    với m = khối lượng MỘT phân tử [kg], T = nhiệt độ tuyệt đối [K].

Nhiệm vụ (theo bảng phân công):
    1. Lập trình thuật toán tính tích phân xác định trên [v1, v2].
    2. Kiểm tra dữ liệu đầu vào (input validation) chặt chẽ.

Phương pháp:
    * Analytic  : CDF F(v) qua hàm lỗi erf (chính xác + nhanh).
    * Numeric   : Quy tắc Simpson 1/3 ghép, tự thích nghi với sai số.
    * Tail cut  : Cắt đuôi tại v = TAIL_CUT·a để tránh tích phân vô ích.

Nguồn tham khảo:
    [1] NIST CODATA 2022 — k_B, N_A (chính xác theo SI 2019)
    [2] IUPAC Atomic Weights — khối lượng mol He/N2/O2/Ar
    [3] Burden & Faires, Numerical Analysis, 10th ed. — Simpson's rule
    [4] NIST DLMF §7.2 — Error function
    [5] OpenStax Univ. Physics Vol.2 — Maxwell speed distribution

Đơn vị SI: v [m/s], T [K], m [kg], M [kg/mol].
Chạy thử:
    python mb_member3_unified.py --demo
    python mb_member3_unified.py --check
    python mb_member3_unified.py --gas N2 --T 300 --v1 0 --v2 500
"""

from __future__ import annotations

import argparse
import math
import numbers
import sys
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
    "He": 4.002602,          # Heli (đơn nguyên tử)
    "N2": 2 * 14.007,        # Nitơ = 2 × N
    "O2": 2 * 15.999,        # Oxy  = 2 × O
    "Ar": 39.95,             # Argon (đơn nguyên tử)
}

# Tham số thuật toán
TAIL_CUT   = 10.0            # Cắt đuôi tại v = TAIL_CUT·a (phần còn lại < 1e-15)
DEFAULT_TOL = 1e-10          # Sai số tuyệt đối cho xác suất P
INITIAL_N  = 200             # Số khoảng Simpson ban đầu (phải chẵn)
MAX_N      = 2_000_000       # Giới hạn trên số khoảng


# -----------------------------------------------------------------
# 2. Ngoại lệ dùng chung
# -----------------------------------------------------------------
class InputValidationError(ValueError):
    """Dữ liệu đầu vào không hợp lệ (thông báo bằng tiếng Việt)."""


# -----------------------------------------------------------------
# 3. Kiểm tra dữ liệu đầu vào
# -----------------------------------------------------------------
def _as_finite_real(value, name: str) -> float:
    """Ép về float; từ chối bool / non-number / NaN / ±Inf."""
    if isinstance(value, bool):
        raise InputValidationError(
            f"{name} phải là số thực, không phải bool ({value!r})."
        )
    if not isinstance(value, numbers.Real):
        raise InputValidationError(
            f"{name} phải là số thực, nhận {value!r} ({type(value).__name__})."
        )
    x = float(value)
    if not math.isfinite(x):
        raise InputValidationError(
            f"{name} phải hữu hạn (không NaN/Inf), nhận {x}."
        )
    return x


def _positive_real(value, name: str) -> float:
    x = _as_finite_real(value, name)
    if x <= 0.0:
        raise InputValidationError(f"{name} phải > 0, nhận {x}.")
    return x


def _non_negative_real(value, name: str) -> float:
    x = _as_finite_real(value, name)
    if x < 0.0:
        raise InputValidationError(f"{name} phải ≥ 0, nhận {x}.")
    return x


def validate_inputs(T, v1, v2, m) -> tuple[float, float, float, float]:
    """Kiểm tra & chuẩn hoá (T, v1, v2, m). Trả về bộ bốn float."""
    T_f  = _positive_real(T, "Nhiệt độ T")
    m_f  = _positive_real(m, "Khối lượng phân tử m")
    v1_f = _non_negative_real(v1, "Tốc độ v1")
    v2_f = _non_negative_real(v2, "Tốc độ v2")

    if v1_f >= v2_f:
        raise InputValidationError(
            f"Cần v1 < v2, nhận v1 = {v1_f}, v2 = {v2_f}."
        )
    if v2_f >= C_LIGHT:
        raise InputValidationError(
            f"v2 = {v2_f} m/s ≥ c; Maxwell–Boltzmann chỉ áp dụng khi v ≪ c."
        )
    return T_f, v1_f, v2_f, m_f


def validate_tolerance(tol) -> float:
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
        raise InputValidationError(f"tol phải là số thực, nhận {tol!r}.")
    t = float(tol)
    if not math.isfinite(t) or t <= 0.0:
        raise InputValidationError(f"tol phải hữu hạn và > 0, nhận {t}.")
    return t


# -----------------------------------------------------------------
# 4. Chuyển đổi khối lượng
# -----------------------------------------------------------------
def mass_from_molar(M) -> float:
    """M [kg/mol] → m [kg] của một phân tử."""
    return _positive_real(M, "Khối lượng mol M") / N_A


def mass_from_gas(gas: str) -> float:
    """Tra khối lượng phân tử [kg] từ tên khí."""
    if not isinstance(gas, str):
        raise InputValidationError("Tên khí phải là chuỗi ký tự.")
    key = gas.strip()
    if key not in MOLAR_MASS_G_PER_MOL:
        raise InputValidationError(
            f"Khí '{gas}' không được hỗ trợ. "
            f"Các lựa chọn: {sorted(MOLAR_MASS_G_PER_MOL)}."
        )
    return MOLAR_MASS_G_PER_MOL[key] * 1e-3 / N_A


# -----------------------------------------------------------------
# 5. Hàm phân bố Maxwell–Boltzmann
# -----------------------------------------------------------------
def scale_parameter(T: float, m: float) -> float:
    """a = sqrt(k_B T / m) [m/s]; ném lỗi nếu thang suy biến."""
    ratio = K_B * T / m
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise InputValidationError(
            f"Thang vận tốc a = sqrt(k_B·T/m) không hợp lệ "
            f"(T = {T}, m = {m}); kiểm tra lại đơn vị."
        )
    a = math.sqrt(ratio)
    if a <= 0.0 or not math.isfinite(a):
        raise InputValidationError("Thang vận tốc a không hợp lệ.")
    return a


def maxwell_boltzmann_pdf(v: float, T: float, m: float) -> float:
    """
    f(v) = √(2/π) · (v²/a³) · exp(-v²/(2a²)),  a = √(k_B T / m).

    Tính trong log-space để tránh tràn số với m/T cực đoan.
    """
    if v <= 0.0:
        return 0.0
    a = scale_parameter(T, m)
    try:
        ratio = v / a                                # = x
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
    if log_val > 700.0:                               # không xảy ra với input vật lý
        return math.exp(700.0)
    return math.exp(log_val)


def _erf_cdf(x: float) -> float:
    """
    F(x) = ∫₀^x √(2/π) u² exp(-u²/2) du
         = erf(x/√2) − √(2/π)·x·exp(-x²/2).

    Xử lý cancellation khi x nhỏ (Taylor bậc 3) và bão hoà khi x lớn.
    """
    if x <= 0.0:
        return 0.0
    if x < 1.0e-4:                                    # Taylor: F ≈ √(2/π)·x³/3
        return math.sqrt(2.0 / math.pi) * x ** 3 / 3.0
    if x > 30.0:                                      # exp(-x²/2) underflow
        return 1.0
    val = (
        math.erf(x / math.sqrt(2.0))
        - math.sqrt(2.0 / math.pi) * x * math.exp(-0.5 * x * x)
    )
    return max(0.0, min(1.0, val))


def analytic_cdf(v: float, T: float, m: float) -> float:
    """CDF F(v) = P(0 ≤ speed ≤ v)."""
    if v <= 0.0:
        return 0.0
    return _erf_cdf(v / scale_parameter(T, m))


def analytic_probability(v1, v2, T, m) -> float:
    """P(v1 ≤ v ≤ v2) = F(v2) − F(v1)."""
    T, v1, v2, m = validate_inputs(T, v1, v2, m)
    p = analytic_cdf(v2, T, m) - analytic_cdf(v1, T, m)
    return max(0.0, min(1.0, p))


# -----------------------------------------------------------------
# 6. Tích phân số: Simpson 1/3 ghép
# -----------------------------------------------------------------
def simpson_composite(f: Callable[[float], float],
                      a: float, b: float, n: int) -> float:
    """
    Quy tắc Simpson 1/3 ghép với n khoảng (n chẵn ≥ 2):
        I ≈ h/3·[f₀ + 4·Σ_{i lẻ} f_i + 2·Σ_{i chẵn trong} f_i + f_n],
        h = (b − a)/n.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2:
        raise InputValidationError(
            f"Số khoảng n phải là số nguyên chẵn ≥ 2, nhận {n!r}."
        )
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4.0 * f(a + i * h)
    for i in range(2, n, 2):
        s += 2.0 * f(a + i * h)
    return s * h / 3.0


@dataclass(frozen=True)
class IntegrationResult:
    probability: float       # P(v1 ≤ v ≤ v2)
    n_intervals: int         # số khoảng Simpson cuối cùng
    error_estimate: float    # ước lượng sai số |S₂ₙ − Sₙ|/15 (nan nếu fixed-n)
    v_upper_used: float      # cận trên thực sự dùng (sau cắt đuôi)
    tail_truncated: bool     # True nếu v2 bị cắt về TAIL_CUT·a
    converged: bool          # True nếu đạt tol trước khi chạm MAX_N


def probability_numeric(
    v1, v2, T, m,
    tol: float = DEFAULT_TOL,
    n: Optional[int] = None,
) -> IntegrationResult:
    """
    Tính P(v1 ≤ v ≤ v2) bằng Simpson ghép.

    n: nếu được cho → Simpson cố định với n khoảng.
       nếu None → tự thích nghi (nhân đôi n) đến khi |S₂ₙ − Sₙ|/15 ≤ tol.
    """
    T, v1, v2, m = validate_inputs(T, v1, v2, m)
    tol = validate_tolerance(tol)

    a_scale = scale_parameter(T, m)
    v_cut = TAIL_CUT * a_scale

    if v1 >= v_cut:                                  # toàn bộ trong đuôi
        return IntegrationResult(0.0, 0, 0.0, v1, True, True)

    truncated = v2 > v_cut
    v_hi = min(v2, v_cut)

    def f(u):
            return maxwell_boltzmann_pdf(u, T, m)

    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2:
            raise InputValidationError(
                f"n phải là số nguyên chẵn ≥ 2, nhận {n!r}."
            )
        val = simpson_composite(f, v1, v_hi, n)
        return IntegrationResult(
            probability=max(0.0, min(1.0, val)),
            n_intervals=n,
            error_estimate=float("nan"),
            v_upper_used=v_hi,
            tail_truncated=truncated,
            converged=True,
        )

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
    """
    Trả về P(v1 ≤ v ≤ v2).
      'analytic' → CDF qua erf (mặc định; chính xác + nhanh)
      'numeric'  → Simpson ghép (truyền tol / n vào probability_numeric)
    """
    if method == "analytic":
        return analytic_probability(v1, v2, T, m)
    if method == "numeric":
        return probability_numeric(v1, v2, T, m, **kwargs).probability
    raise InputValidationError(
        f"method phải là 'analytic' hoặc 'numeric', nhận {method!r}."
    )


def probability_gas(gas, T, v1, v2, method="analytic", **kwargs) -> float:
    """Tiện dụng: tra khối lượng phân tử từ tên khí rồi tính P."""
    return probability(v1, v2, T, mass_from_gas(gas), method=method, **kwargs)


# -----------------------------------------------------------------
# 8. Tự kiểm chứng (dùng cho --check)
# -----------------------------------------------------------------
def self_check() -> bool:
    """So sánh analytic và numeric trên nhiều bộ tham số (kiểm chứng chéo)."""
    print("=" * 72)
    print("SELF-CHECK: analytic (erf) vs numeric (Simpson thích nghi)")
    print("=" * 72)
    cases = [
        ("N2", 300.0,   0.0,  500.0),
        ("N2", 300.0, 100.0,  600.0),
        ("N2", 300.0, 300.0,  600.0),
        ("O2", 300.0, 300.0,  700.0),
        ("He", 300.0, 500.0, 2500.0),
        ("He",  50.0,   0.0,  500.0),
        ("Ar",1000.0, 200.0,  900.0),
        ("Ar", 100.0,   0.0,  300.0),
    ]
    print(f"{'Khí':<5}{'T[K]':>8}{'v1':>8}{'v2':>8}"
          f"{'P_analytic':>16}{'P_numeric':>16}{'|Δ|':>12}{'  ':<3}")
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
    """Ví dụ minh hoạ cho báo cáo."""
    print("=" * 72)
    print("DEMO: P(v1 ≤ v ≤ v2) — Maxwell–Boltzmann")
    print("=" * 72)
    cases = [
        ("N2", 300.0,   0.0,  500.0),
        ("N2", 600.0,   0.0,  500.0),
        ("He", 300.0,   0.0, 1500.0),
        ("O2", 300.0, 300.0,  800.0),
        ("Ar",1000.0, 200.0,  900.0),
    ]
    print(f"{'Khí':<5}{'T[K]':>8}{'v1':>10}{'v2':>10}"
          f"{'P(analytic)':>16}{'P(numeric)':>16}")
    for gas, T, v1, v2 in cases:
        m = mass_from_gas(gas)
        pa = analytic_probability(v1, v2, T, m)
        pn = probability_numeric(v1, v2, T, m).probability
        print(f"{gas:<5}{T:>8.0f}{v1:>10.0f}{v2:>10.0f}"
              f"{pa:>16.6f}{pn:>16.6f}")


# -----------------------------------------------------------------
# 9. CLI
# -----------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Xác suất P(v1 ≤ v ≤ v2) cho phân bố tốc độ "
                    "Maxwell–Boltzmann (tích phân + validation)."
    )
    mg = p.add_mutually_exclusive_group(required=False)
    mg.add_argument("--gas", choices=sorted(MOLAR_MASS_G_PER_MOL),
                    help="Tên khí: He, N2, O2, Ar.")
    mg.add_argument("--mass", type=float,
                    help="Khối lượng một phân tử [kg].")
    mg.add_argument("--molar-mass", type=float,
                    help="Khối lượng mol [kg/mol].")
    p.add_argument("--T",  type=float, help="Nhiệt độ tuyệt đối [K].")
    p.add_argument("--v1", type=float, help="Cận dưới [m/s].")
    p.add_argument("--v2", type=float, help="Cận trên [m/s].")
    p.add_argument("--method", choices=["analytic", "numeric", "both"],
                   default="both",
                   help="Phương pháp tính (mặc định: both).")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL,
                   help=f"Sai số tuyệt đối cho numeric (mặc định {DEFAULT_TOL:g}).")
    p.add_argument("--n", type=int, default=None,
                   help="Số khoảng Simpson cố định (mặc định: thích nghi).")
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
        provided = sum(x is not None for x in
                       (args.gas, args.mass, args.molar_mass))
        if provided != 1:
            raise InputValidationError(
                "Cần cung cấp đúng một trong --gas / --mass / --molar-mass."
            )
        if args.T is None or args.v1 is None or args.v2 is None:
            raise InputValidationError("Cần cung cấp --T, --v1, --v2.")

        if args.gas is not None:
            m = mass_from_gas(args.gas)
            label = args.gas
        elif args.mass is not None:
            m = _positive_real(args.mass, "particle mass m")
            label = "(mass given)"
        else:
            m = mass_from_molar(args.molar_mass)
            label = "(molar mass given)"

        T, v1, v2, m = validate_inputs(args.T, args.v1, args.v2, m)

        print(f"Khí / nguồn m     : {label}")
        print(f"m                 : {m:.6e} kg")
        print(f"T                 : {T} K")
        print(f"Khoảng tốc độ     : [{v1}, {v2}] m/s")

        pa = None
        if args.method in ("analytic", "both"):
            pa = analytic_probability(v1, v2, T, m)
            print(f"P (analytic/erf)  : {pa:.12f}  ({pa*100:.6f} %)")

        if args.method in ("numeric", "both"):
            res = probability_numeric(v1, v2, T, m, tol=args.tol, n=args.n)
            print(f"P (numeric)       : {res.probability:.12f}  "
                  f"({res.probability*100:.6f} %)")
            print(f"  n_intervals     : {res.n_intervals}")
            print(f"  error estimate  : {res.error_estimate:.2e}")
            print(f"  converged       : {res.converged}")
            print(f"  tail truncated  : {res.tail_truncated}")
            if pa is not None:
                print(f"|ΔP|              : {abs(pa - res.probability):.2e}")

        return 0

    except InputValidationError as exc:
        print(f"[LỖI ĐẦU VÀO] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
        print()
        self_check()
    else:
        sys.exit(main())