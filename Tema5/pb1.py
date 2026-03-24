import numpy as np
import scipy.linalg
import math

def sign(x):
    return 1 if x >= 0 else -1

# Jacobi (p = n)

def metoda_jacobi(A_init, epsilon=1e-8, kmax=1000):

    n = A_init.shape[0]
    A = A_init.copy().astype(float)
    U = np.eye(n)
    k = 0   
    
    while k <= kmax:
        # sub diagonala principala
        max_val = -1.0
        p, q = -1, -1
        for i in range(1, n):
            for j in range(i):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
                    
        if max_val <= epsilon:
            break
            
        # aici calc teta
        a_pp = A[p, p]
        a_qq = A[q, q]
        a_pq = A[p, q]
        
        alpha = (a_pp - a_qq) / (2.0 * a_pq)
        t = -alpha + sign(alpha) * math.sqrt(alpha**2 + 1)
        
        c = 1.0 / math.sqrt(1 + t**2)
        s = t / math.sqrt(1 + t**2)
        
        # actualizam matricea
        for j in range(n):
            if j != p and j != q:
                a_pj = A[p, j]
                a_qj = A[q, j]
                
                A[p, j] = c * a_pj + s * a_qj
                A[j, p] = A[p, j]
                
                A[q, j] = -s * a_pj + c * a_qj
                A[j, q] = A[q, j] 
                
        A[p, p] = a_pp + t * a_pq
        A[q, q] = a_qq - t * a_pq
        
        A[p, q] = 0.0
        A[q, p] = 0.0
        
        # actualizam matricea de transformare U
        for i in range(n):
            u_ip_vechi = U[i, p]
            u_iq_vechi = U[i, q]
            
            U[i, p] = c * u_ip_vechi + s * u_iq_vechi
            U[i, q] = -s * u_ip_vechi + c * u_iq_vechi
            
        k += 1

    return A, U


# Cholesky
def sir_matrice_cholesky(A_init, epsilon=1e-8, kmax=1000):

    A_k = A_init.copy().astype(float)
    k = 0
    
    while k < kmax:
        try:
            # L * L^T = A_k
            L = np.linalg.cholesky(A_k)
            # A_(k+1) = L^T * L
            A_next = np.dot(L.T, L)
        except np.linalg.LinAlgError:
            # daca esueaza Cholesky, incercam LU
            P, L, U = scipy.linalg.lu(A_k)
            # A_next = U * L (Algoritmul LR clasic)
            A_next = np.dot(U, L)
            
        if np.linalg.norm(A_next - A_k) < epsilon:
            A_k = A_next
            break
            
        A_k = A_next
        k += 1
        
    return A_k

#  p > n (SVD)
def svd_p_mai_mare_ca_n(A):
    p, n = A.shape
    
    # SVD: A = U * S * V^T
    U, sing_vals, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T
    
    # afizes valorile singulare
    print("-> Valorile singulare ale matricei A:", sing_vals)
    
    # rang
    tol = 1e-10
    rang_formule = np.sum(sing_vals > tol)
    rang_lib = np.linalg.matrix_rank(A)
    print(f"-> Rangul matricei A: {rang_formule} (Verificare biblioteca: {rang_lib})")
    
    # nr de conditionare
    if rang_formule > 0:
        sigma_max = np.max(sing_vals)
        sigma_min_poz = np.min(sing_vals[sing_vals > tol])
        cond_formule = sigma_max / sigma_min_poz
    else:
        cond_formule = float('inf')
    
    cond_lib = np.linalg.cond(A)
    print(f"-> Numarul de conditionare: {cond_formule:.6f} (Verificare biblioteca: {cond_lib:.6f})")
    
    # pseudoinversa Moore-Penrose A_I = V * S_I * U^T
    S_I = np.zeros((n, p))
    for i in range(rang_formule):
        S_I[i, i] = 1.0 / sing_vals[i]
    
    A_I = np.dot(V, np.dot(S_I, U.T))
    
    try:
        A_J = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
        norma_dif = np.linalg.norm(A_I - A_J, 1) # norma 1 
        print(f"-> Norma matriciala ||A_I - A_J||_1: {norma_dif:.6e}")
    except np.linalg.LinAlgError:
        print("-> Atentie: Matricea (A^T * A) nu este inversabila. A_J nu poate fi calculata clasic.")

if __name__ == "__main__":
    print("Cazul p = n")
    A_sim = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=float)
    
    # Jacobi
    Lambda_aprox, U_aprox = metoda_jacobi(A_sim)
    valori_proprii_jacobi = np.diag(Lambda_aprox)
    
    print("Metoda Jacobi:")
    print("Valorile proprii aproximate (de pe diagonala):\n", valori_proprii_jacobi)
    
    # verificam A_init * U ≈ U * Lambda
    Lambda_diag = np.diag(valori_proprii_jacobi)
    verificare = np.linalg.norm(np.dot(A_sim, U_aprox) - np.dot(U_aprox, Lambda_diag))
    print(f"Verificare ||A_init * U - U * Lambda||: {verificare:.6e}")
    
    # 2. Șirul de matrice
    print("\nSirul de matrice (Cholesky/LU):")
    A_final_sir = sir_matrice_cholesky(A_sim)
    print("Ultima matrice calculata:\n", np.diag(A_final_sir))
    
    print("\nCazul p > n (SVD):")
    # Generam o matrice dreptunghiulara (p=5, n=3)
    A_drept = np.array([
        [1, 2, 0],
        [2, 0, 2],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ], dtype=float)
    
    svd_p_mai_mare_ca_n(A_drept)