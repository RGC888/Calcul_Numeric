import os

def citeste_vector(nume_fisier):
    """Citește un vector de numere reale dintr-un fișier text."""
    try:
        with open(nume_fisier, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Eroare: Nu am putut găsi fișierul {nume_fisier}")
        return []

def rezolva_sistem(index_sistem, epsilon=1e-6, kmax=10000):
    print(f"\n{'='*40}")
    print(f"--- Rezolvare sistemul {index_sistem} ---")
    
    director_curent = os.path.dirname(os.path.abspath(__file__))

    f_d0 = os.path.join(director_curent, f"d0_{index_sistem}.txt")
    f_d1 = os.path.join(director_curent, f"d1_{index_sistem}.txt")
    f_d2 = os.path.join(director_curent, f"d2_{index_sistem}.txt")
    f_b  = os.path.join(director_curent, f"b_{index_sistem}.txt")

    d0 = citeste_vector(f_d0)
    d1 = citeste_vector(f_d1)
    d2 = citeste_vector(f_d2)
    b  = citeste_vector(f_b)

    if not d0 or not d1 or not d2 or not b:
        return

    # 1. calc dimensiunea sistemului
    n = len(d0)
    if len(b) != n:
        print("Eroare: Vectorul b nu are aceeași dimensiune cu d0.")
        return
        
    print(f"1. Dimensiunea sistemului (n): {n}")

    # 2. calc ordinul diagonalelor 
    p = n - len(d1)
    q = n - len(d2)
    print(f"2. Diagonalele secundare sunt de ordin: p = {p}, q = {q}")

    # 3. verificarea elementelor de pe diagonala principală
    toate_nenule = True
    for val in d0:
        if abs(val) <= epsilon:
            toate_nenule = False
            break
            
    if not toate_nenule:
        print("3. Eroare: Există elemente nule (sau < epsilon) pe diagonala principală. Metoda se oprește.")
        return
    else:
        print("3. Toate elementele de pe diagonala principală sunt nenule.")

    # 4. metoda Gauss-Seidel
    xc = [0.0] * n  # vectorul curent
    xp = [0.0] * n  # vectorul precedent
    k = 0

    while True:
        #iteratia anterioara
        for i in range(n):
            xp[i] = xc[i]
            
        delta_x = 0.0

        for i in range(n):
            suma = 0.0
            
            # elementele din stg deja calculate (deci xc)
            if i - q >= 0:
                suma += d2[i - q] * xc[i - q]
            if i - p >= 0:
                suma += d1[i - p] * xc[i - p]
                
            # elementele din dreapta ce vor fi calculate (deci xp)
            if i + p < n:
                suma += d1[i] * xp[i + p]
            if i + q < n:   
                suma += d2[i] * xp[i + q]
                
            # calc noul xc[i]
            noua_valoare = (b[i] - suma) / d0[i]
            
            # delta reprezinta cea mai mare eroare
            diff = abs(noua_valoare - xp[i])
            if diff > delta_x:
                delta_x = diff
                
            xc[i] = noua_valoare
                
        k += 1
        
        if delta_x < epsilon or k > kmax or delta_x > 1e10:
            break

    if delta_x < epsilon:
        print(f"4. Soluția a fost aproximată cu succes după {k} iterații.")
        
        # 5. y = A * x_GS
        y = [0.0] * n
        for i in range(n):
            valoare_y = d0[i] * xc[i]
            
            if i - p >= 0:
                valoare_y += d1[i - p] * xc[i - p]
            if i + p < n:
                valoare_y += d1[i] * xc[i + p]
            if i - q >= 0:
                valoare_y += d2[i - q] * xc[i - q]
            if i + q < n:
                valoare_y += d2[i] * xc[i + q]
                
            y[i] = valoare_y
            
        # 6. norma infinita
        norma_inf = 0.0
        for i in range(n):
            eroare = abs(y[i] - b[i])
            if eroare > norma_inf:
                norma_inf = eroare
                
        print(f"5 & 6. Norma ||A*x_GS - b||_inf = {norma_inf}")
        
        # Afișăm primele 5 componente ale soluției ca să verificăm cu exemplele din temă
        print(f"   Primele 5 elemente ale soluției x_GS: {[round(x, 4) for x in xc[:5]]}")
        
    else:
        print(f"4. Divergență: Algoritmul s-a oprit (delta_x = {delta_x}, iterații = {k}).")

if __name__ == "__main__":
    for i in range(1, 6):
        rezolva_sistem(i, epsilon=1e-6)