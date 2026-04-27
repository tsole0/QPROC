import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------
# 1. CONSTANTS & PARAMETERS (Ammonia)
# ---------------------------------------------------------
R = 8.31446    # Universal Gas Constant [J/(mol K)]
Tc = 405.4     # Critical Temperature [K]
Pc = 11.33e6   # Critical Pressure [Pa]
omega = 0.253  # Acentric factor

# Calculate fixed Peng-Robinson Constants
m = 0.37464 + 1.54226 * omega - 0.26992 * (omega**2)
b = 0.07780 * R * Tc / Pc
ac = 0.45724 * (R**2) * (Tc**2) / Pc

# ---------------------------------------------------------
# 2. TEMPERATURE-DEPENDENT FUNCTIONS
# ---------------------------------------------------------
def a(T):
    """Temperature-dependent attractive parameter."""
    return ac * (1 + m * (1 - np.sqrt(T / Tc)))**2

# ---------------------------------------------------------
# 3. THE ROOT SOLVER
# ---------------------------------------------------------
def get_v(T, P):
    """Finds the volume root of the PR cubic equation."""
    A = a(T) * P / (R**2 * T**2)
    B = b * P / (R * T)
    
    # PR Cubic Polynomial: Z^3 - (1-B)Z^2 + (A - 2B - 3B^2)Z - (AB - B^2 - B^3) = 0
    coeffs = [1, -(1 - B), A - 2*B - 3*B**2, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    
    # Filter for real roots and take the largest
    real_roots = roots[np.isreal(roots)].real
    Z = np.max(real_roots)
    
    # Return volume (v = ZRT/P)
    return Z * R * T / P

# ---------------------------------------------------------
# 4. NUMERICAL DERIVATIVES & REAL CP
# ---------------------------------------------------------
def dv_dT(T, P):
    """First derivative of raw volume w.r.t Temperature."""
    dT_step = 0.01 
    return (get_v(T + dT_step, P) - get_v(T - dT_step, P)) / (2 * dT_step)

def d2v_dT2(P_int, T):
    """
    Second derivative of RAW volume w.r.t Temp.
    P_int is the first argument for scipy.integrate.quad.
    """
    dT_step = 0.1
    
    v_plus  = get_v(T + dT_step, P_int)
    v_base  = get_v(T, P_int)
    v_minus = get_v(T - dT_step, P_int)
    
    return (v_plus - 2*v_base + v_minus) / (dT_step**2)

# The Anchor Point (From tables for NH3 at 200 MPa and 180 C)
CP_START = 75.19 # J/(mol K)
P_ANCHOR = 200e6 # Pa

def cp_real(T, P_current):
    """Calculates real-gas Cp by integrating from the anchor pressure down to current P."""
    # Integrate from 200 MPa down to P_current
    integral_val, _ = quad(d2v_dT2, P_ANCHOR, P_current, args=(T,))
    
    # Cp(P) = Cp(Anchor) - T * integral(d2v/dT2 dP)
    return CP_START - T * integral_val

# ---------------------------------------------------------
# 5. THE INTEGRATION LOOP
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initial Conditions
    T_start_C = 180.0
    T_current = T_start_C + 273.15 # Convert to Kelvin
    P_start = 200e6                # 200 MPa to Pa
    P_end = 50e6                   # 50 MPa to Pa
    
    steps = 150                    # Number of integration slices
    dP = (P_end - P_start) / steps # This will be a negative value (-1e6 Pa/step)
    
    print(f"Starting isenthalpic expansion from {P_start/1e6:.1f} MPa at {T_start_C:.2f} °C...")
    
    P_current = P_start
    
    for i in range(steps):
        # 1. Find Volume
        v = get_v(T_current, P_current)
        
        # 2. Find Derivatives and Cp
        dv_dT_val = dv_dT(T_current, P_current)
        cp = cp_real(T_current, P_current)
        
        # 3. Calculate Joule-Thomson Coefficient
        mu_JT = (1 / cp) * (T_current * dv_dT_val - v)
        
        # 4. Update Temperature and Pressure for this step
        dT_step = mu_JT * dP
        T_current += dT_step
        P_current += dP
        
    # Final Output
    T_final_C = T_current - 273.15
    delta_T = T_final_C - T_start_C
    
    print("-" * 40)
    print(f"Final Pressure:    {P_current/1e6:.1f} MPa")
    print(f"Final Temperature: {T_final_C:.2f} °C")
    print(f"Total Temp Change: {delta_T:+.2f} °C")