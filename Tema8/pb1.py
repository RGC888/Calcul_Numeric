import numpy as np

def numerical_gradient(F, x, h=1e-5):
    """Calculează gradientul numeric folosind formula aproximativă din text."""
    # h este pasul mic pentru aproximarea derivatelor parțiale
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        # Copiem x pentru a nu modifica vectorul original
        x1 = np.copy(x); x1[i] += 2 * h
        x2 = np.copy(x); x2[i] += h
        x3 = np.copy(x); x3[i] -= h
        x4 = np.copy(x); x4[i] -= 2 * h
        
        F1 = F(x1)
        F2 = F(x2)
        F3 = F(x3)
        F4 = F(x4)
        
        grad[i] = (-F1 + 8*F2 - 8*F3 + F4) / (12 * h)
    return grad

def backtracking_line_search(F, x, grad_x, beta=0.8):
    """Ajustare de tip backtracking pentru rata de învățare."""
    #grad_x este gradientul calculat la punctul x
    eta = 1.0 #pasul initial
    p = 1 #contor pentru numarul de ajustari

    #lungimea patratica a gradientului, folosita pentru a verifica conditia de backtracking
    grad_norm_sq = np.linalg.norm(grad_x) ** 2
    
    # Prevenim erori de overflow/evaluare la pași prea mari inițiali
    try:
        while F(x - eta * grad_x) > F(x) - (eta / 2) * grad_norm_sq and p < 8: # daca valoarea este mai mare decat cea a punctului curent, atunci pasul este prea mare
            eta *= beta
            p += 1  
    except OverflowError:
        # În caz de overflow la evaluare, forțăm un pas mai mic
        eta *= beta
        
    return eta

def gradient_descent(F, grad_F_analytical, x0, lr_strategy='backtracking', 
                     grad_strategy='analytical', epsilon=1e-5, kmax=30000, fixed_lr=1e-3):
    """
    Implementarea algoritmului principal conform schemei de calcul.
    """
    x = np.array(x0, dtype=float)
    k = 0
    
    while True:
        # 1. Calculează gradientul
        if grad_strategy == 'analytical':
            grad = grad_F_analytical(x)
        else:
            grad = numerical_gradient(F, x)
            
        # 2. Calculează rata de învățare
        if lr_strategy == 'backtracking':
            eta = backtracking_line_search(F, x, grad)
        else:
            eta = fixed_lr
            
        # Condiție de oprire evaluată direct pe lungimea pasului: eta * ||grad(x)||
        step_len = eta * np.linalg.norm(grad) # lungimea pasului de actualizare, folosita pentru a verifica convergenta
        
        # Actualizare x
        x = x - eta * grad
        k += 1
        
        # 3. Verifică condițiile de oprire
        if step_len <= epsilon:
            return x, k, "Convergent"
        if k > kmax:
            return x, k, "Max Iteratii (Posibil divergent sau lent)"
        if step_len > 1e10 or np.isnan(step_len):
            return x, k, "Divergent"

def sigmoid(z):
    # Folosim clip pentru a evita overflow în np.exp
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# Funcția 1: l(w0, w1)
def F1(x):
    w0, w1 = x[0], x[1]
    # Evităm log(0) prin adunarea unei valori foarte mici (1e-15)
    return -np.log(1 - sigmoid(w0 - w1) + 1e-15) - np.log(sigmoid(w0 + w1) + 1e-15)

def grad_F1(x):
    w0, w1 = x[0], x[1]
    df_dw0 = sigmoid(w0 - w1) + sigmoid(w0 + w1) - 1
    df_dw1 = sigmoid(w0 + w1) - sigmoid(w0 - w1) - 1
    return np.array([df_dw0, df_dw1])

# Funcția 2
def F2(x):
    return x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1] - 1

def grad_F2(x):
    return np.array([2*x[0] - 2, 2*x[1] - 4])

# Funcția 3
def F3(x):
    return 3*x[0]**2 - 12*x[0] + 2*x[1]**2 + 16*x[1] - 10

def grad_F3(x):
    return np.array([6*x[0] - 12, 4*x[1] + 16])

# Funcția 4
def F4(x):
    return x[0]**2 - 4*x[0]*x[1] + 4.5*x[1]**2 - 4*x[1] + 3

def grad_F4(x):
    return np.array([2*x[0] - 4*x[1], -4*x[0] + 9*x[1] - 4])

# Funcția 5
def F5(x):
    return x[0]**2 * x[1] - 2*x[0]*x[1]**2 + 3*x[0]*x[1] + 4

def grad_F5(x):
    return np.array([2*x[0]*x[1] - 2*x[1]**2 + 3*x[1], x[0]**2 - 4*x[0]*x[1] + 3*x[0]])

# Lista de funcții pentru testare
functions = [
    {"name": "F1 (Logistic Loss)", "F": F1, "grad": grad_F1},
    {"name": "F2", "F": F2, "grad": grad_F2},
    {"name": "F3", "F": F3, "grad": grad_F3},
    {"name": "F4", "F": F4, "grad": grad_F4},
    {"name": "F5", "F": F5, "grad": grad_F5}
]

if __name__ == "__main__":
    np.random.seed(42) # Setăm un seed pentru reproductibilitate
    
    for i, func in enumerate(functions):
        print(f"--- Testare: {func['name']} ---")
        
        # Alegem un punct de start aleator relativ aproape de origine
        x0 = np.random.uniform(-1, 1, size=2)
        print(f"Punct start (x0): [{x0[0]:.4f}, {x0[1]:.4f}]")
        
        # Test 1: Gradient Analitic + Backtracking
        x_ana, k_ana, status_ana = gradient_descent(
            func['F'], func['grad'], x0, 
            lr_strategy='backtracking', grad_strategy='analytical'
        )
        
        # Test 2: Gradient Numeric + Backtracking
        x_num, k_num, status_num = gradient_descent(
            func['F'], func['grad'], x0, 
            lr_strategy='backtracking', grad_strategy='numerical'
        )
        
        # Test 3: Gradient Analitic + Rata de învățare constantă
        x_const, k_const, status_const = gradient_descent(
            func['F'], func['grad'], x0, 
            lr_strategy='constant', grad_strategy='analytical', fixed_lr=1e-3
        )
        
        print(f"  > Grad. Analitic + Backtracking: {k_ana} iterații | Soluție: {x_ana} | {status_ana}")
        print(f"  > Grad. Numeric  + Backtracking: {k_num} iterații | Soluție: {x_num} | {status_num}")
        print(f"  > Grad. Analitic + Eta Constant: {k_const} iterații | Soluție: {x_const} | {status_const}")
        print("\n")

#la f4 da divergent pentru backtracking deoarece avem conditia p<8 si pasul de backtracking nu reuseste sa scada suficient de mult pentru a indeplini conditia de oprire, ceea ce duce la o crestere exponentiala a pasului si, in final, la divergent.