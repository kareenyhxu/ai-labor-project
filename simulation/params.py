import numpy as np

# -----------------------------
# Baseline parameters
# -----------------------------
Lbar = 1.0       # total measure of workers supplied
Kbar = 1.0       # total capital stock
lam  = 4.0       # elasticity of substitution across occupations
theta = 1.5      # measure of generic tasks
eta  = 15.0      # capital (relative) productivity

# Occupations to track in plots (Figure 2)
phi_list = [0.2, 0.4, 0.6, 0.8]

# Grid of automation cutoffs
I_grid = np.linspace(0.1, 0.65, 300)  # needs to satisfy Assum 6

# Numerical integration settings
QUAD_LIMIT = 300     # max subintervals for adaptive quadrature, default in safe.quad
