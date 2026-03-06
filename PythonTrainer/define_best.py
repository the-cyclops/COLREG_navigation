import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_runs(log_dir="runs/Grid_Search_initial_reward", top_k=5):
    runs_data = []

    print(f"Analisi della cartella {log_dir} in corso...")
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                ea = EventAccumulator(os.path.join(root, file))
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                
                if "Training/Mean_Return" not in tags:
                    continue

                # Estraiamo gli ultimi 50 valori per calcolare la media stabile di fine addestramento
                returns = [s.value for s in ea.Scalars("Training/Mean_Return")][-50:]
                entropy = [s.value for s in ea.Scalars("Policy/Entropy")][-50:] if "Policy/Entropy" in tags else [999]
                steer_std = [s.value for s in ea.Scalars("Policy/Steering_Std")][-50:] if "Policy/Steering_Std" in tags else [999]
                throttle_std = [s.value for s in ea.Scalars("Policy/Throttle_Std")][-50:] if "Policy/Throttle_Std" in tags else [999]

                runs_data.append({
                    "run": os.path.basename(root),
                    "return": np.mean(returns),
                    "entropy": np.mean(entropy),
                    "steer_std": np.mean(steer_std),
                    "throttle_std": np.mean(throttle_std),
                    "total_std": np.mean(steer_std) + np.mean(throttle_std)
                })

    if not runs_data:
        print("Nessun dato trovato. Controlla il path di log_dir.")
        return

    # 1. Troviamo il return massimo raggiunto a fine addestramento
    max_return = max(r["return"] for r in runs_data)
    
    # 2. Teniamo solo i modelli che performano in modo simile (es. entro 1.5 punti dal max)
    valid_runs = [r for r in runs_data if r["return"] >= max_return - 1.5]
    
    # 3. Li ordiniamo per deviazione standard totale crescente (cercando le azioni più stabili)
    valid_runs.sort(key=lambda x: x["total_std"])

    print(f"\n--- TOP {top_k} SETUP SUGGERITI ---")
    print(f"Filtrati per Return vicino al max (>= {max_return - 1.5:.2f}) e ordinati per massima stabilità di controllo:\n")
    
    for i, r in enumerate(valid_runs[:top_k], 1):
        print(f"{i}. {r['run']}")
        print(f"   Return: {r['return']:.2f} | Entropy: {r['entropy']:.4f} | Steer Std: {r['steer_std']:.4f} | Throttle Std: {r['throttle_std']:.4f}\n")

if __name__ == "__main__":
    # Assicurati di lanciare lo script dalla directory PythonTrainer
    analyze_runs(log_dir="runs/Grid_Search_initial_reward")