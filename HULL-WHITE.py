import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Market data
maturities = np.array([0.0, 0.2493150684931507, 0.5013698630136987, 0.7534246575342466, 
                       1.0, 3.0082191780821916, 5.002739726027397, 8.005479452054795, 
                       10.008219178082191, 15.013698630136986, 20.019178082191782, 
                       25.016438356164382])
discount_factors = np.array([1.0, 0.9828472951503892, 0.9655676031402194, 0.9481221554793985, 
                             0.9309067565567731, 0.7864070426924387, 0.6586262149130966, 
                             0.4731428899670019, 0.36771603681856185, 0.20626168272503248, 
                             0.1201872976177416, 0.0830308835811612])

# Interpolated market discount factors
P0T = interp1d(maturities, discount_factors, kind='quadratic', fill_value="extrapolate")

# Forward curve and its derivative
def forward_curve(t, P0T):
    dt = 0.01    
    expr = - (np.log(P0T(t+dt)) - np.log(P0T(t-dt))) / (2*dt)
    return expr

def forward_derivative(t, P0T, dt=0.01):
    df_dt = (forward_curve(t + dt, P0T) - forward_curve(t - dt, P0T)) / (2 * dt)
    return df_dt

# Theta function
def theta(t, lambd, eta, P0T):
    f = forward_curve(t, P0T)
    df_dt = forward_derivative(t, P0T)
    theta_t = df_dt + (lambd * f) + (eta**2) / (2 * lambd) * (1 - np.exp(-2 * lambd * t))
    return theta_t

# Analytical A(t, T) and B(t, T)
def A(t, T, lambd, eta, P0T):
    if t == 0:
        return P0T(T)  
    
    Pt = P0T(t)
    PT = P0T(T)
    
    B_tt = B(t, T, lambd)
    
    f_t = forward_curve(t, P0T)
    
    term1 = np.log(PT / Pt)
    term2 = B_tt * f_t
    term3 = (eta**2 / (4 * lambd**3)) * (1 - np.exp(-2 * lambd * t)) * B_tt**2
    
    return np.exp(term1 + term2 - term3)

def B(t, T, lambd):
    return (1 - np.exp(-lambd * (T - t))) / lambd

# Parameters
lambd = 0.15  # Mean reversion speed
eta = 0.1   # Volatility of short rate

# Compute analytical ZCB prices
analytical_prices = []
for T in maturities:
    price = A(0, T, lambd, eta, P0T)
    analytical_prices.append(price)

# Convert to numpy array
analytical_prices = np.array(analytical_prices)

# Calculate differences between market discount factors and Hull-White ZCB prices
differences = np.abs(discount_factors - analytical_prices)

# Store values and differences
results = []
for i, T in enumerate(maturities):
    results.append({
        "Maturity": T,
        "Market Discount Factor": discount_factors[i],
        "Hull-White ZCB Price": analytical_prices[i],
        "Difference": differences[i]
    })

# Print results in a tabular format
print(f"{'Maturity (Years)':<15} {'Market DF':<15} {'HW ZCB Price':<15} {'Difference':<15}")
print("-" * 60)
for result in results:
    print(f"{result['Maturity']:<15.2f} {result['Market Discount Factor']:<15.6f} "
          f"{result['Hull-White ZCB Price']:<15.6f} {result['Difference']:<15.6f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(maturities, discount_factors, label="Market Discount Curve", marker='o', linestyle='--')
plt.plot(maturities, analytical_prices, label="Hull-White Analytical ZCB Prices", marker='x', linestyle='-')
plt.title("Comparison of Market Discount Curve and Hull-White ZCB Prices")
plt.xlabel("Maturity (Years)")
plt.ylabel("Discount Factor")
plt.legend()
plt.grid(True)
plt.show()
