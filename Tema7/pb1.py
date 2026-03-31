import random

def horner(coefs, val):
    b = coefs[0]
    for a in coefs[1:]:
        b = a + b * val
    return b

def derivata_polinom(coefs):
    n = len(coefs) - 1
    return [coefs[i] * (n - i) for i in range(n)]

def raza_radacinilor(coefs):
    a0 = abs(coefs[0])
    A = max(abs(c) for c in coefs[1:])
    return (a0 + A) / a0

def metoda_newton(coefs, coefs_d1, x0, epsilon, kmax=1000):
    x = x0
    k = 0
    while True:
        p_d1_val = horner(coefs_d1, x)
        
        # dacă derivata este prea mică, metoda poate diverge sau se poate bloca, deci oprim căutarea
        if abs(p_d1_val) <= epsilon:
            return None, k
        
        p_val = horner(coefs, x)
        delta_x = p_val / p_d1_val
        x = x - delta_x
        k += 1
        
        if abs(delta_x) < epsilon:
            return x, k
        if k > kmax or abs(delta_x) >= 10**8:
            return None, k

def metoda_olver(coefs, coefs_d1, coefs_d2, x0, epsilon, kmax=1000):
    x = x0
    k = 0
    while True:
        p_d1_val = horner(coefs_d1, x)
        
        if abs(p_d1_val) <= epsilon:
            return None, k
            
        p_val = horner(coefs, x)
        p_d2_val = horner(coefs_d2, x)
        
        c_k = ((p_val ** 2) * p_d2_val) / (p_d1_val ** 3)
        delta_x = (p_val / p_d1_val) + 0.5 * c_k
        
        x = x - delta_x
        k += 1
        
        if abs(delta_x) < epsilon:
            return x, k
        if k > kmax or abs(delta_x) >= 10**8:
            return None, k


def gaseste_radacinile():
    #  (x) = x^3 - 6x^2 + 11x - 6
    coefs = [1.0, -6.0, 11.0, -6.0]
    epsilon = 1e-6
    nr_incercari = 100 
    
    coefs_d1 = derivata_polinom(coefs)
    coefs_d2 = derivata_polinom(coefs_d1)
    
    R = raza_radacinilor(coefs)
    print(f"Intervalul de căutare a rădăcinilor reale: [{-R:.4f}, {R:.4f}]\n")
    
    # Structură pentru a stoca rezultatele: 
    # [
    #   {
    #       'val': 1.000002,         # Valoarea aproximativă a rădăcinii
    #       'newton': [5, 6, 5, 7],  # Pașii făcuți de metoda Newton când a ajuns la ea
    #       'olver': [3, 4, 3]       # Pașii făcuți de metoda Olver când a ajuns la ea
    #   },
    #   {
    #       'val': 1.999998,         # O altă rădăcină distinctă (diferență > epsilon)
    #       'newton': [8, 9],
    #       'olver': [4, 5, 5, 4]
    #   }
    # ]

    radacini_gasite = []

    def adauga_la_statistici(radacina, pasi, metoda):
        if radacina is None:
            return
        
        for r_dict in radacini_gasite:
            if abs(r_dict['val'] - radacina) <= epsilon:
                r_dict[metoda].append(pasi)
                r_dict['val'] = (r_dict['val'] + radacina) / 2.0 
                return
        
        # daca rad e noua
        noua_radacina = {'val': radacina, 'newton': [], 'olver': []}
        noua_radacina[metoda].append(pasi)
        radacini_gasite.append(noua_radacina)

    # rulam ambele metode
    for _ in range(nr_incercari):
        x0 = random.uniform(-R, R)
        
        rad_n, pasi_n = metoda_newton(coefs, coefs_d1, x0, epsilon)
        adauga_la_statistici(rad_n, pasi_n, 'newton')
        
        rad_o, pasi_o = metoda_olver(coefs, coefs_d1, coefs_d2, x0, epsilon)
        adauga_la_statistici(rad_o, pasi_o, 'olver')


    radacini_gasite.sort(key=lambda x: x['val'])

    print(f"{'Rădăcină (aprox)':<20} | {'Medie Pași Newton':<20} | {'Medie Pași Olver':<20}")
    print("-" * 65)
    
    # Variabile pentru a calcula overall-ul
    total_pasi_newton = 0
    total_succese_newton = 0
    total_pasi_olver = 0
    total_succese_olver = 0
    
    for r_dict in radacini_gasite:
        val = r_dict['val']
        
        # Calcule pentru Newton
        nr_succese_n = len(r_dict['newton'])
        suma_pasi_n = sum(r_dict['newton'])
        medie_n = suma_pasi_n / nr_succese_n if nr_succese_n else 0
        
        total_pasi_newton += suma_pasi_n
        total_succese_newton += nr_succese_n
        
        # Calcule pentru Olver
        nr_succese_o = len(r_dict['olver'])
        suma_pasi_o = sum(r_dict['olver'])
        medie_o = suma_pasi_o / nr_succese_o if nr_succese_o else 0
        
        total_pasi_olver += suma_pasi_o
        total_succese_olver += nr_succese_o
        
        print(f"{val:<20.6f} | {medie_n:<20.2f} | {medie_o:<20.2f}")

    # calc si afisare
    print("=" * 65)
    medie_generala_n = total_pasi_newton / total_succese_newton if total_succese_newton else 0
    medie_generala_o = total_pasi_olver / total_succese_olver if total_succese_olver else 0
    
    print(f"{'MEDIE GENERALĂ':<20} | {medie_generala_n:<20.2f} | {medie_generala_o:<20.2f}")
    print("=" * 65)

    # scriere în fișier
    nume_fisier = "radacini.txt"
    with open(nume_fisier, "w") as f:
        f.write("Radacini reale distincte gasite:\n")
        f.write("-" * 35 + "\n")
        for r_dict in radacini_gasite:
            f.write(f"{r_dict['val']:.6f}\n")
            
    print(f"\nRădăcinile distincte au fost salvate cu succes în '{nume_fisier}'.")

if __name__ == "__main__":
    gaseste_radacinile()