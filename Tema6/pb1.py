import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

def f_derivata(x):
    return 4*x**3 - 36*x**2 + 60*x

x0 = float(input("Introdu x0: "))
xn = float(input("Introdu xn: "))

if x0 >= xn:
    raise ValueError("Date invalide: x0 nu poate fi mai mare decat xn.")

da = f_derivata(x0)  
db = f_derivata(xn) 
x_bar = 1.5


n = 10   # nr subinterbale
m = 4     # gradul polinomului pentru prima metoda

x_nodes = [x0]

random_points = sorted([random.uniform(x0, xn) for _ in range(n-1)])
x_nodes.extend(random_points)
x_nodes.append(xn)
x_nodes = np.array(x_nodes)

y_nodes = f(x_nodes)


print(f"Nodurile generate x: {np.round(x_nodes, 4)}")
print(f"Valorile f(x):       {np.round(y_nodes, 4)}")
print(f"Punctul de evaluare x_bar: {x_bar}\n")

# Construirea sistemului B * a = f_vec
B = np.zeros((m+1, m+1))
f_vec = np.zeros(m+1)

for i in range(m+1):
    for j in range(m+1):
        B[i, j] = np.sum(x_nodes**(i+j))
    f_vec[i] = np.sum(y_nodes * (x_nodes**i))

a_coeffs = np.linalg.solve(B, f_vec)

def horner(coeffs, val):
    grad = len(coeffs) - 1
    d = coeffs[grad]
    for i in range(grad-1, -1, -1):
        d = coeffs[i] + d * val
    return d


P_xbar = horner(a_coeffs, x_bar)
val_exacta_xbar = f(x_bar)

suma_erori_noduri = np.sum(np.abs([horner(a_coeffs, xi) - yi for xi, yi in zip(x_nodes, y_nodes)]))

print(f"Pm(x_bar)            = {P_xbar:.6f}")
print(f"|Pm(x_bar) - f(x_bar)| = {abs(P_xbar - val_exacta_xbar):.6f}")
print(f"Suma |Pm(xi) - yi|   = {suma_erori_noduri:.6f}\n")


# spline
h = np.diff(x_nodes) # h[i] = x[i+1] - x[i]
H = np.zeros((n+1, n+1))
F = np.zeros(n+1)

# H * A = F
H[0, 0] = 2 * h[0]
H[0, 1] = h[0]
F[0] = 6 * ((y_nodes[1] - y_nodes[0])/h[0] - da)

for i in range(1, n):
    H[i, i-1] = h[i-1]
    H[i, i]   = 2 * (h[i-1] + h[i])
    H[i, i+1] = h[i]
    F[i] = 6 * ((y_nodes[i+1] - y_nodes[i])/h[i] - (y_nodes[i] - y_nodes[i-1])/h[i-1])

H[n, n-1] = h[n-1]
H[n, n]   = 2 * h[n-1]
F[n] = 6 * (db - (y_nodes[n] - y_nodes[n-1])/h[n-1])

A = np.linalg.solve(H, F)

def eval_spline(val):
    
    if val <= x_nodes[0]:
        i0 = 0
    elif val >= x_nodes[-1]:
        i0 = n - 1
    else:
        for i in range(n):
            if x_nodes[i] <= val <= x_nodes[i+1]:
                i0 = i
                break
                
    hi = h[i0]
    Ai = A[i0]
    Ai1 = A[i0+1]
    xi = x_nodes[i0]
    xi1 = x_nodes[i0+1]
    yi = y_nodes[i0]
    yi1 = y_nodes[i0+1]

    # calc coeficientii
    bi = (yi1 - yi)/hi - hi*(Ai1 - Ai)/6
    ci = (xi1*yi - xi*yi1)/hi - hi*(xi1*Ai - xi*Ai1)/6

    termen1 = ((val - xi)**3 * Ai1) / (6 * hi)
    termen2 = ((xi1 - val)**3 * Ai) / (6 * hi)
    
    return termen1 + termen2 + bi * val + ci

S_xbar = eval_spline(x_bar)

print("=== Aproximarea prin Funcții Spline Cubice C2 ===")
print(f"Sf(x_bar)            = {S_xbar:.6f}")
print(f"|Sf(x_bar) - f(x_bar)| = {abs(S_xbar - val_exacta_xbar):.6f}\n")


x_plot = np.linspace(x0, xn, 300)
y_exact = f(x_plot)
y_lsa = [horner(a_coeffs, xp) for xp in x_plot]
y_spline = [eval_spline(xp) for xp in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_exact, 'k-', linewidth=2, label='f(x) - Funcția exactă')
plt.plot(x_plot, y_lsa, 'b--', linewidth=1.5, label=f'Pm(x) - Cel mai mici pătrate (m={m})')
plt.plot(x_plot, y_spline, 'r-.', linewidth=1.5, label='Sf(x) - Spline Cubic C2')
plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Nodurile de interpolare')
plt.axvline(x=x_bar, color='gray', linestyle=':', label=f'x_bar = {x_bar}')

plt.title('Aproximarea funcției prin LSA și Spline Cubice')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()