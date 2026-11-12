import math
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Iterable

# ==========================================
# CONSTANTS & CONFIGURATIONS
# ==========================================
# Sử dụng hằng số khí lý tưởng R (J/(mol*K)) vì dữ liệu đầu vào dùng Molar Mass (kg/mol)
R = 8.31446261815324 
LOG_TWO_OVER_PI = math.log(2.0 / math.pi)
SQRT2 = math.sqrt(2.0)
TAIL_CUTOFF_FACTOR = 10.0
MAX_N_SUBINTERVALS = 2_000_000

# ==========================================
# CUSTOM EXCEPTIONS & DATA STRUCTURES
# ==========================================
class InputValidationError(ValueError):
    """Exception raised for invalid physical or numerical parameters."""
    pass

@dataclass
class IntegrationResult:
    probability: float
    n_intervals: int
    error_estimate: float
    v_upper_used: float
    tail_truncated: bool
    converged: bool
    elapsed_ms: float

# ==========================================
# 1. INPUT VALIDATION (Chapter 1 & 4.1)
# ==========================================
def validate_inputs(T: float, m: float, v1: float, v2: float, tol: float = 1e-10, N: Optional[int] = None) -> Tuple[float, float, float, float, float]:
    """Validate physical parameters and numerical parameters."""
    # Check type and finiteness
    for name, val in [("T", T), ("m", m), ("v1", v1), ("v2", v2), ("tol", tol)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InputValidationError(f"Parameter '{name}' must be a numeric float/int, got {type(val).__name__}.")
        if not math.isfinite(val):
            raise InputValidationError(f"Parameter '{name}' must be a finite number, got {val}.")

    if T <= 0:
        raise InputValidationError(f"Temperature T must be strictly positive (T > 0 K), got T = {T}.")
    if m <= 0:
        raise InputValidationError(f"Molar mass M must be strictly positive (M > 0 kg/mol), got M = {m}.")
    if v1 < 0 or v2 < 0:
        raise InputValidationError(f"Speed bounds v1, v2 must be non-negative, got v1 = {v1}, v2 = {v2}.")
    if tol < 1e-15 or tol >= 1.0:
        raise InputValidationError(f"Tolerance 'tol' must be in range [1e-15, 1.0), got {tol}.")
    if N is not None:
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise InputValidationError(f"Subinterval count N must be a positive integer >= 1, got {N}.")

    return float(T), float(m), float(v1), float(v2), float(tol)

# ==========================================
# 2. SCALE & DENSITY FUNCTIONS (Chapter 4.2 & 4.3)
# ==========================================
def scale_parameter(T: float, m: float) -> float:
    """Compute characteristic speed scale a = sqrt(R * T / m)."""
    try:
        a2 = R * T / m
        if a2 <= 0 or not math.isfinite(a2):
            raise InputValidationError(f"Invalid physical parameter ratio R*T/m = {a2}.")
        return math.sqrt(a2)
    except (OverflowError, ZeroDivisionError) as e:
        raise InputValidationError(f"Physical scale computation overflowed/failed: {e}")

def most_probable_speed(T: float, m: float) -> float:
    """Compute most probable speed v_p = a * sqrt(2)."""
    return scale_parameter(T, m) * SQRT2

def _make_integrand(T: float, m: float) -> Tuple[Callable[[float], float], float]:
    """Pre-compute constant closure for ultra-fast adaptive quadrature loops."""
    a = scale_parameter(T, m)
    a2_inv_half = 0.5 / (a * a)
    log_c = 0.5 * LOG_TWO_OVER_PI - 3.0 * math.log(a)

    def integrand(v: float) -> float:
        if v <= 0.0:
            return 0.0
        log_val = log_c + 2.0 * math.log(v) - (v * v) * a2_inv_half
        if log_val < -745.0: # IEEE 754 float64 underflow threshold
            return 0.0
        return math.exp(log_val)

    return integrand, a

# ==========================================
# 3. NUMERICAL INTEGRATION METHODS (Chapter 4.4 & 4.5)
# ==========================================
def _neumaier_sum(values: Iterable[float]) -> float:
    """Neumaier compensated summation algorithm for high-precision summing."""
    s = 0.0
    c = 0.0
    for x in values:
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c

def trapezoidal_composite(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Evaluate composite trapezoidal rule T_N over interval [a, b] with N subintervals."""
    if n < 1:
        raise ValueError("Subinterval count N must be >= 1.")
    if a == b:
        return 0.0

    h = (b - a) / n
    endpoints_sum = 0.5 * (f(a) + f(b))
    interior_sum = _neumaier_sum(f(a + i * h) for i in range(1, n))
    
    return h * (endpoints_sum + interior_sum)

def trapezoid_refine(f: Callable[[float], float], a: float, b: float, n_prev: int, t_prev: float) -> float:
    """Refine trapezoidal sum from N_prev to 2*N_prev by evaluating only new midpoints."""
    h_new = (b - a) / (2 * n_prev)
    # Evaluate new midpoints: a + (2i + 1)*h_new
    midpoints_sum = _neumaier_sum(f(a + (2 * i + 1) * h_new) for i in range(n_prev))
    return 0.5 * t_prev + h_new * midpoints_sum

# ==========================================
# 4. MASTER CONTROLLER (Chapter 4.6)
# ==========================================
def probability_numeric(v1: float, v2: float, T: float, m: float, tol: float = 1e-10, n: Optional[int] = None) -> IntegrationResult:
    """Master controller for computing Maxwell-Boltzmann speed interval probability."""
    t_start = time.perf_counter()
    
    # 1. Input Validation
    T, m, v1, v2, tol = validate_inputs(T, m, v1, v2, tol, n)

    # 2. Handle boundary swap
    if v1 > v2:
        warnings.warn(f"Lower bound v1 ({v1}) > v2 ({v2}). Swapping boundaries.", RuntimeWarning)
        v1, v2 = v2, v1

    # 3. Handle degenerate interval
    if v1 == v2:
        elapsed = (time.perf_counter() - t_start) * 1000.0
        return IntegrationResult(0.0, 0, 0.0, v2, False, True, elapsed)

    # 4. Setup Integrand
    f, a = _make_integrand(T, m)
    v_cut = TAIL_CUTOFF_FACTOR * a

    # 5. Tail truncation check
    if v1 >= v_cut:
        elapsed = (time.perf_counter() - t_start) * 1000.0
        return IntegrationResult(0.0, 0, 0.0, v2, True, True, elapsed)

    v_hi = min(v2, v_cut)
    tail_truncated = (v2 > v_cut)

    # 6. Fixed-N Mode
    if n is not None:
        val = trapezoidal_composite(f, v1, v_hi, n)
        val = max(0.0, min(1.0, val)) # Clamping [0, 1]
        elapsed = (time.perf_counter() - t_start) * 1000.0
        return IntegrationResult(val, n, float('nan'), v_hi, tail_truncated, True, elapsed)

    # 7. Adaptive Mode
    n_cur = 100
    t_cur = trapezoidal_composite(f, v1, v_hi, n_cur)
    converged = False
    est_error = float('inf')

    while 2 * n_cur <= MAX_N_SUBINTERVALS:
        t_next = trapezoid_refine(f, v1, v_hi, n_cur, t_cur)
        est_error = abs(t_next - t_cur) / 3.0
        
        n_cur *= 2
        t_cur = t_next

        if est_error <= tol:
            converged = True
            break

    final_p = max(0.0, min(1.0, t_cur))
    elapsed = (time.perf_counter() - t_start) * 1000.0
    
    return IntegrationResult(final_p, n_cur, est_error, v_hi, tail_truncated, converged, elapsed)

# ==========================================
# 5. EXECUTION / UNIT TEST (Chapter 5)
# ==========================================
if __name__ == "__main__":
    print("=== MAXWELL-BOLTZMANN INTEGRATION ENGINE ===\n")

    # --- TEST 1: Input Validation Check (Error Handling) ---
    print("--- Testing Input Validation Guard Layer ---")
    try:
        probability_numeric(v1=0, v2=500, T=-10, m=0.004)
    except InputValidationError as e:
        print(f"[PASS] Caught expected error: {e}")
        
    try:
        probability_numeric(v1=0, v2=500, T=300, m=0)
    except InputValidationError as e:
        print(f"[PASS] Caught expected error: {e}\n")

    # --- TEST 2: Case A - Scale Invariance (Helium) ---
    print("--- Case A: Scale Invariance (Helium at 300K) ---")
    M_helium = 0.0040026 # kg/mol
    T_test = 300.0       # K
    
    vp = most_probable_speed(T_test, M_helium)
    print(f"Most Probable Speed (v_p) = {vp:.2f} m/s")
    
    res_A = probability_numeric(v1=0.0, v2=vp, T=T_test, m=M_helium, tol=1e-10)
    print(f"Probability P [0, v_p]: {res_A.probability:.12f}")
    print(f"Converged: {res_A.converged} | Intervals: {res_A.n_intervals} | Time: {res_A.elapsed_ms:.2f} ms\n")

    # --- TEST 3: Case B - Generic Interval [0, 500] m/s ---
    print("--- Case B: Generic Interval [0, 500] m/s ---")
    res_B = probability_numeric(v1=0.0, v2=500.0, T=T_test, m=M_helium, tol=1e-10)
    print(f"Probability P [0, 500]: {res_B.probability:.12f}")
    print(f"Converged: {res_B.converged} | Intervals: {res_B.n_intervals} | Time: {res_B.elapsed_ms:.2f} ms\n")

    # --- TEST 4: Boundary Swap Warning ---
    print("--- Testing Boundary Swap (v1 > v2) ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_swap = probability_numeric(v1=500, v2=0, T=300, m=M_helium)
        if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
            print(f"[PASS] Warning issued: {w[-1].message}")
            print(f"Probability P [0, 500]: {res_swap.probability:.12f}\n")

    # --- TEST 5: Degenerate Bounds ---
    print("--- Testing Degenerate Bounds (v1 == v2) ---")
    res_deg = probability_numeric(v1=300, v2=300, T=300, m=M_helium)
    print(f"Probability P [300, 300]: {res_deg.probability}")
    print(f"Converged: {res_deg.converged} | Time: {res_deg.elapsed_ms:.2f} ms\n")

    # --- TEST 6: Tail Truncation ---
    print("--- Testing Tail Truncation (v1 >= 10*a) ---")
    res_tail = probability_numeric(v1=5000, v2=10000, T=300, m=M_helium)
    print(f"Probability P [5000, 10000]: {res_tail.probability}")
    print(f"Tail Truncated: {res_tail.tail_truncated}")