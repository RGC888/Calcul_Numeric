#Butnaru Stefan - 310910401RSL231036 - butnarustefan04@gmail.com - butnarustefan
#Rusu George-Constantin - 310910401RSL221191 - rusugeorge333@gmail.com - George
# 15% AI
# Toate formulele si algoritmii au fost explicati in clasa de catre laborant; avem link-ul cu explicatiile mai jos
# https://onedrive.live.com/personal/401895cf656dd8f0/_layouts/15/Doc.aspx?sourcedoc={6247276d-0cb1-4847-a468-ddaad611c641}&action=view&redeem=aHR0cHM6Ly8xZHJ2Lm1zL28vYy80MDE4OTVjZjY1NmRkOGYwL0lnQnRKMGRpc1F4SFNLUm8zYXJXRWNaQkFVMmVseWFocXpUM05vREg4NUxrWjU4P2U9OHBUMmFr&wd=target%28Tema2.one%7C5da459bd-5a8d-41a7-b7b3-7d61e34b0a2f%2FTema2%7C3609a860-3d07-498c-9c0e-574625b9c8f3%2F%29&wdorigin=NavigationUrl

import numpy as np
import sys

# fct pt Rx=b, unde R este triunghiular supeiror
def back_substitution(R, b, eps):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma_elemente = np.sum(R[i, i+1:] * x[i+1:])
        if abs(R[i, i]) > eps:
            x[i] = (b[i] - suma_elemente) / R[i, i]
        else:
            raise ValueError(f"Nu se poate face împărțirea la R[{i},{i}].")
    return x

# Qr cu transformarea simultana a lui b
def householder_qr(A_init, b_init, eps):
    n = A_init.shape[0]
    A = A_init.copy().astype(float)
    b = b_init.copy().astype(float)
    Q_t = np.eye(n).astype(float)

    for r in range(n - 1):
        sigma = np.sum(A[r:, r]**2)

        if sigma <= eps:
            break 

        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k

        beta = sigma - k * A[r, r]

        u = np.zeros(n)
        u[r] = A[r, r] - k
        u[r+1:] = A[r+1:, r]

        if abs(beta) > eps:
            # transformarea coloanelor de dupa ce curenta
            for j in range(r + 1, n):
                gamma = np.dot(u[r:], A[r:, j]) / beta
                A[r:, j] = A[r:, j] - gamma * u[r:]

            # transformarea coloanei curente
            A[r, r] = k
            A[r+1:, r] = 0

            # transformarea vectorului b
            gamma_b = np.dot(u[r:], b[r:]) / beta
            b[r:] = b[r:] - gamma_b * u[r:]

            # transformarea matricei Q_t -> o folosim mai tarziu pt inversa
            for j in range(n):
                gamma_q = np.dot(u[r:], Q_t[r:, j]) / beta
                Q_t[r:, j] = Q_t[r:, j] - gamma_q * u[r:]

    return Q_t, A, b # aici b e transformat cu Q_t

def main():
    n = int(input("Introduceți dimensiunea sistemului n: "))
    t = int(input("Introduceți puterea t pentru precizie: "))
    eps = 10**(-t)

    A_init = np.random.rand(n, n) * 10
    s = np.random.rand(n) * 10
    b_init = np.dot(A_init, s)

    # descompunerea QR -> b va fi transformat direrect in functie
    Q_t, R, b_transformat = householder_qr(A_init, b_init, eps)

    for i in range(n):
        if abs(R[i, i]) <= eps:
            print(f"Matricea este singulara")
            sys.exit(0)

    # Ax = b
    # cu biblioteca
    Q_lib, R_lib = np.linalg.qr(A_init)
    y_lib = np.dot(Q_lib.T, b_init)
    x_QR = back_substitution(R_lib, y_lib, eps)

    # cu Householder calculat de noi
    x_Householder = back_substitution(R, b_transformat, eps)

    print("--- Diferența între soluții ---")
    diff_sols = np.linalg.norm(x_QR - x_Householder)
    print(f"|| x_QR - x_Householder ||_2 = {diff_sols:.6e}")

    #erorile
    print("\n--- Erorile ---")
    
    def euclidean_norm(v):
        suma = 0.0
        for i in range(len(v)):
            suma += v[i]**2
        return suma**0.5

    def euclidean_norm_diff(v1, v2):
        suma = 0.0
        for i in range(len(v1)):
            suma += (v1[i] - v2[i])**2
        return suma**0.5

    eroare_1 = euclidean_norm_diff(np.dot(A_init, x_Householder), b_init)
    eroare_2 = euclidean_norm_diff(np.dot(A_init, x_QR), b_init)
    eroare_3 = euclidean_norm_diff(x_Householder, s) / euclidean_norm(s)
    eroare_4 = euclidean_norm_diff(x_QR, s) / euclidean_norm(s)

    print(f"|| A_init * x_Householder - b_init ||_2 = {eroare_1:.6e}")
    print(f"|| A_init * x_QR - b_init ||_2          = {eroare_2:.6e}")
    print(f"|| x_Householder - s ||_2 / || s ||_2   = {eroare_3:.6e}")
    print(f"|| x_QR - s ||_2 / || s ||_2            = {eroare_4:.6e}")

    # inversa matricei
    print("\n--- Inversa matricei ---")
    A_inv_Householder = np.zeros((n, n))

    # aici avem nevoie de Q_t
    for j in range(n):
        b_col = Q_t[:, j] 
        x_col = back_substitution(R, b_col, eps)
        A_inv_Householder[:, j] = x_col

    A_inv_bibl = np.linalg.inv(A_init)

    def matrix_diff_norm(M1, M2):
        diff_flat = (M1 - M2).flatten()
        return euclidean_norm(diff_flat)

    eroare_inv = matrix_diff_norm(A_inv_Householder, A_inv_bibl)
    print(f"|| A_inv_Householder - A_inv_bibl ||    = {eroare_inv:.6e}")

if __name__ == "__main__":
    main()