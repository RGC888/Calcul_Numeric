def gaseste_precizia_masina():
    m = 0
    u = 1.0
    
    # Verificăm inegalitatea 1.0 + u/10 != 1.0
    # Cât timp adunarea încă are efect, incrementăm m și actualizăm u
    while (1.0 + (u / 10.0)) != 1.0: # u/10 ca sa ne oprim direct la m-ul nostru (sa nu mai dau dupa m--)
        m += 1
        u = 10.0 ** (-m)
        
    return m, u

m_rezultat, u_rezultat = gaseste_precizia_masina()

print(f"Cel mai mic m pentru care 1.0 + 10^(-m) != 1.0 este: m = {m_rezultat}")
print(f"Precizia mașină (u) în puteri ale lui 10 este: {u_rezultat}")