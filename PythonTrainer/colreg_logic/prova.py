from rtamt_yml_parser import RTAMTYmlParser

def run_until_test():
    # 1. Definizione dello YAML custom con le tre logiche
    yaml_content = """
dense: True
timestep: 1.0
variables:
  - name: x
    type: float
    location: obs
    identifier: id_x
  - name: y
    type: float
    location: obs
    identifier: id_y
specifications:
  - name: R6_stand_on
    spec: R6_stand_on = always[0,2]((y <= 0.0) or ((x >= 0.0) until (y <= 0.0)))
    weight: 1.0
"""
    with open("test_until.yaml", "w") as f:
        f.write(yaml_content)

    print("Inizializzazione parser...")
    parser = RTAMTYmlParser("test_until.yaml")

  # Simulazione: 
    # t=0..2: Navigazione corretta (y positivo, rotta x mantenuta)
    # t=3: Sblocco corretto (y scende a -0.5, x può virare a -1.0)
    # t=4: Navigazione corretta
    # t=5: Errore! Vira (x = -1.0) prima che la situazione sia risolta (y = 1.0)
    # t=6: Sblocco in ritardo
    
    dummy_x = [ 1.0,  1.0,  1.0, -1.0,   1.0, -1.0, -1.0, -1.0]
    dummy_y = [ 2.0,  1.0,  0.5, -0.5,   2.0,  1.0, -0.5, -1.0]
    tau_state = []
    for x_val, y_val in zip(dummy_x, dummy_y):
        tau_state.append({'id_x': x_val, 'id_y': y_val})

    print("Calcolo robustezza in corso...\n")
    total_rho, single_rho = parser.compute_robustness_dense(tau_state)

    # 3. Stampa dei risultati formattati
    print("--- DATI IN INGRESSO ---")
    print(f"Step: {[s for s in range(len(tau_state))]}")
    print(f"x:    {[d['id_x'] for d in tau_state]}")
    print(f"y:    {[d['id_y'] for d in tau_state]}")
    
    print("\n--- RISULTATI ROBUSTEZZA (RHO) ---")
    print(f"R6_stand_on: {[round(r, 2) for r in single_rho['R6_stand_on']]}")

if __name__ == "__main__":
    run_until_test()