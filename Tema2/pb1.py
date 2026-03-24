#Butnaru Stefan - 310910401RSL231036 - butnarustefan04@gmail.com - butnarustefan
#Rusu George-Constantin - 310910401RSL221191 - rusugeorge333@gmail.com - George
# 25% AI
# Toate formulele si algoritmii au fost explicati in clasa de catre laborant; avem link-ul cu explicatiile mai jos
# https://onedrive.live.com/personal/401895cf656dd8f0/_layouts/15/Doc.aspx?sourcedoc={6247276d-0cb1-4847-a468-ddaad611c641}&action=view&redeem=aHR0cHM6Ly8xZHJ2Lm1zL28vYy80MDE4OTVjZjY1NmRkOGYwL0lnQnRKMGRpc1F4SFNLUm8zYXJXRWNaQkFVMmVseWFocXpUM05vREg4NUxrWjU4P2U9OHBUMmFr&wd=target%28Tema2.one%7C5da459bd-5a8d-41a7-b7b3-7d61e34b0a2f%2FTema2%7C3609a860-3d07-498c-9c0e-574625b9c8f3%2F%29&wdorigin=NavigationUrl


import numpy as np

#citim datele de intrare
n = int(input("n = "))
t = int(input("t (eps = 10^(-t)) = "))
eps = 10 ** (-t)

#gegneram matricea A si vectorul b
B = np.random.rand(n, n)
A = B @ B.T                 # matrice simetrica pozitiv definita
A_init = A.copy()

b = np.random.rand(n)

#calculam solutia cu biblioteca np
xlib = np.linalg.solve(A_init, b)

print("\nSolutia cu biblioteca (numpy):")
print(xlib)

#facem descompunerea LDLT
#memoram d in vector separat
#L va fi calculat in matricea A (tinem cont de faptul ca matricea va avea valori de 1 pe diagonala, deci nu le stocam explicit)
d = np.zeros(n)

for p in range(n):

    # calcul dp
    suma = 0.0
    for k in range(p):
        suma += d[k] * (A[p, k] ** 2)

    d[p] = A[p, p] - suma

    if abs(d[p]) < eps:
        raise ValueError("Descompunerea LDLT nu poate continua.")

    # calcul elemente L sub diagonala
    for i in range(p + 1, n):
        suma = 0.0
        for k in range(p):
            suma += d[k] * A[i, k] * A[p, k]

        A[i, p] = (A[i, p] - suma) / d[p]

print("\nVectorul d (diagonala lui D):")
print(d)

#calculam determinantul lui A
detA = np.prod(d)
# nu punem det L deoarece ar fi valorile initiale ale matricei A, dar noi stim ca in mod normal ar trebui ca toate valorile sa fie 1

print("\nDeterminant A =", detA)

#rezolvam sistemul folosind descompunerea LDLT

# Lz = b
z = np.zeros(n)
for i in range(n):
    suma = 0.0
    for j in range(i):
        suma += A[i, j] * z[j]
    z[i] = b[i] - suma

# Dy = z
y = np.zeros(n)
for i in range(n):
    if abs(d[i]) < eps:
        raise ValueError("Impartire la zero.")
    y[i] = z[i] / d[i]

# L^T x = y
xChol = np.zeros(n)
for i in reversed(range(n)):
    suma = 0.0
    for j in range(i + 1, n):
        suma += A[j, i] * xChol[j]
    xChol[i] = y[i] - suma

print("\nSolutia cu LDLT:")
print(xChol)

#facem inmultirea manuala pentru a calcula A_init * xChol
Ax = np.zeros(n)

for i in range(n):
    suma = 0.0
    for j in range(n):
        suma += A_init[i, j] * xChol[j]
    Ax[i] = suma

#calculam normele manual cu formula din tema

# ||A_init xChol - b||_2 manual
suma1 = 0.0
for i in range(n):
    suma1 += (Ax[i] - b[i]) ** 2
norm1_manual = np.sqrt(suma1)

# ||xChol - xlib||_2 manual
suma2 = 0.0
for i in range(n):
    suma2 += (xChol[i] - xlib[i]) ** 2
norm2_manual = np.sqrt(suma2)

print("\nNorme calculate manual:")
print("||A_init xChol - b|| =", norm1_manual)
print("||xChol - xlib|| =", norm2_manual)

#calculam normele pentru a compara solutiile (folosind biblioteca numpy)
norm1 = np.linalg.norm(Ax - b, 2)
norm2 = np.linalg.norm(xChol - xlib, 2)

print("\n||A_init xChol - b|| =", norm1)
print("||xChol - xlib|| =", norm2)