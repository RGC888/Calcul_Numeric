import math
import time
import random

#Metoda Fractiilor Continue

def my_tan_lentz(x, eps=1e-10):
    # reducem argumentul la intervalul (-pi/2, pi/2)
    x = x % math.pi
    if x > math.pi / 2:
        x -= math.pi
        
    # tratarea separata a multiplilor de pi/2
    if math.isclose(abs(x), math.pi / 2, rel_tol=1e-9):
        return float('inf') if x > 0 else float('-inf')

    # initializare variabile pentru metoda Lentz
    mic = 1e-12
    b0 = 0.0
    f = b0
    if f == 0.0:
        f = mic
    C = f
    D = 0.0
    j = 1
    
    while True:
        # Conform recurenței pentru tangenta: a_1 = x, a_j = -x^2 pt j > 1
        if j == 1:
            a_j = x
        else:
            a_j = -x * x
            
        b_j = 2 * j - 1 # b_1 = 1, b_2 = 3, b_3 = 5...
        
        # Pasul j
        D = b_j + a_j * D
        if D == 0.0:
            D = mic
            
        C = b_j + a_j / C
        if C == 0.0:
            C = mic
            
        D = 1.0 / D
        Delta = C * D
        f = f * Delta
        
        # Condiția de oprire
        if abs(Delta - 1.0) < eps:
            break
        j += 1
        
    return f

# Metoda Aproximarii Polinomiale

def my_tan_poly_base(x):
    # Funcția de bază pentru polinom, asumată a fi apelată cu x în [0, pi/4]
    c1 = 0.3333333333333333
    c2 = 0.1333333333333333
    c3 = 0.0539682539682540
    c4 = 0.0218694885361552
    
    x_2 = x * x
    x_4 = x_2 * x_2
    x_6 = x_4 * x_2
    
    P = c1 + c2 * x_2 + c3 * x_4 + c4 * x_6
    return x + (x_2 * x) * P # x + x^3 * P(x^2)

def my_tan_poly(x):
    # reducem argumentul la intervalul (-pi/2, pi/2
    x = x % math.pi
    if x > math.pi / 2:
        x -= math.pi
        
    # multiplii de pi/2
    if math.isclose(abs(x), math.pi / 2, rel_tol=1e-9):
        return float('inf') if x > 0 else float('-inf')

    # tan(-x) = -tan(x)
    sign = 1
    if x < 0:
        x = -x
        sign = -1
        
    # Reducerea la intervalul recomandat [0, pi/4]  
    if x > math.pi / 4:
        return sign * (1.0 / my_tan_poly_base(math.pi / 2 - x))
    else:
        return sign * my_tan_poly_base(x)

# generam valori si comparam metodele

def run_comparison():
    N = 10000
    # generam 10.000 numere aleatoare n (-pi/2, pi/2)
    valori_x = [random.uniform(-math.pi/2 + 1e-5, math.pi/2 - 1e-5) for _ in range(N)]
    
    # timp si eroare pentru Math (Referinta)
    start_math = time.perf_counter()
    rezultate_math = [math.tan(x) for x in valori_x]
    timp_math = time.perf_counter() - start_math
    
    # timp si eroare pentru Metoda Lentz
    start_lentz = time.perf_counter()
    rezultate_lentz = [my_tan_lentz(x) for x in valori_x]
    timp_lentz = time.perf_counter() - start_lentz
    
    eroare_max_lentz = max(abs(rezultate_math[i] - rezultate_lentz[i]) for i in range(N))
    
    #timp si eroare pentru Metoda Polinomiala
    start_poly = time.perf_counter()
    rezultate_poly = [my_tan_poly(x) for x in valori_x]
    timp_poly = time.perf_counter() - start_poly
    
    eroare_max_poly = max(abs(rezultate_math[i] - rezultate_poly[i]) for i in range(N))
    
    print(f"--- Comparație pe {N} de valori ---")
    print(f"Timp executie math.tan():    {timp_math:.6f} secunde")
    print("-" * 40)
    print(f"Metoda Fracțiilor Continue (Lentz):")
    print(f"  Timp executie: {timp_lentz:.6f} secunde")
    print(f"  Eroare maxima: {eroare_max_lentz:.2e}")
    print("-" * 40)
    print(f"Metoda Polinomiala:")
    print(f"  Timp executie: {timp_poly:.6f} secunde")
    print(f"  Eroare maxima: {eroare_max_poly:.2e}")

# Rulare program
if __name__ == "__main__":
    run_comparison()     