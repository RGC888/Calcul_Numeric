from ex1 import gaseste_precizia_masina

def rezolva_problema_2():
    m, u = gaseste_precizia_masina()
    print(f"-> Precizia mașină preluată: u = {u} (m = {m})\n")
    
    print("Verificare neasociativitate adunare")
    x_add = 1.0
    y_add = u / 10.0
    z_add = u / 10.0
    
    stanga_add = (x_add + y_add) + z_add
    dreapta_add = x_add + (y_add + z_add)
    
    print(f"x = {x_add}, y = {y_add}, z = {z_add}\n")
    print(f"(x + y) + z = {stanga_add}")
    print(f"x + (y + z) = {dreapta_add}")
    
    if stanga_add != dreapta_add:
        print("Rezultat: Adunarea NU este asociativă (valorile diferă).\n")
    else:
       print("Rezultat: Valorile sunt egale.\n")
       
    print("Căutare neasociativitate înmulțire")
    
    #cautam prin aceste valori deoarece pt unele numere este greu sa aporximam puteri ale lui 2
    for i in range(1, 100):
        for j in range(1, 100):
            for k in range(1, 100):
                x_mul = i / 10.0
                y_mul = j / 10.0
                z_mul = k / 10.0
                
                stanga_mul = (x_mul * y_mul) * z_mul
                dreapta_mul = x_mul * (y_mul * z_mul)
                
                if stanga_mul != dreapta_mul:
                    print(f"Am găsit valorile: x = {x_mul}, y = {y_mul}, z = {z_mul}\n")
                    print(f"(x * y) * z = {stanga_mul}")
                    print(f"x * (y * z) = {dreapta_mul}")
                    print(f"Rezultat: Înmulțirea NU este asociativă!\n")
                    return

if __name__ == "__main__":
    rezolva_problema_2()
   