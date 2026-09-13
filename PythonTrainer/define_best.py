import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_runs(log_dir="runs/GRID_SEARCH_EMPTY_SCENE_FIXED_REWARD_SCALE_1.0", top_k=5, min_final_entropy=-3.0):
    runs_data = []

    print(f"Analisi di {log_dir} in corso...")
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                ea = EventAccumulator(os.path.join(root, file))
                ea.Reload()
                tags = ea.Tags().get('scalars', [])
                
                # Usa Smoothed_Return o Mean_Return come fallback
                ret_tag = "Training/Smoothed_Return" if "Training/Smoothed_Return" in tags else "Training/Mean_Return"
                if ret_tag not in tags:
                    continue

                # Ultimi 50 punti per valutare asintoto e stabilità
                raw_returns = [s.value for s in ea.Scalars(ret_tag)][-50:]
                raw_entropy = [s.value for s in ea.Scalars("Policy/Entropy")][-50:] if "Policy/Entropy" in tags else [float('nan')]

                if len(raw_returns) < 5:
                    continue

                mean_ret = float(np.mean(raw_returns))
                std_ret = float(np.std(raw_returns))
                mean_ent = float(np.mean(raw_entropy))

                runs_data.append({
                    "run": os.path.basename(root),
                    "mean_return": mean_ret,
                    "std_return": std_ret,
                    "entropy": mean_ent
                })

    if not runs_data:
        print("Nessun dato valido trovato in log_dir.")
        return

    # 1. Filtro: escludi run con entropia collassata a valori estremi negativi (se presenti)
    valid_runs = [r for r in runs_data if np.isnan(r["entropy"]) or r["entropy"] > min_final_entropy]
    if not valid_runs:
        valid_runs = runs_data

    # 2. Ordina primariamente per ritorno medio decrescente, secondariamente per deviazione standard crescente
    valid_runs.sort(key=lambda x: (-x["mean_return"], x["std_return"]))

    print(f"\n--- TOP {top_k} CONFIGURAZIONI MIGLIORI ---\n")
    for i, r in enumerate(valid_runs[:top_k], 1):
        print(f"{i}. {r['run']}")
        print(f"   Return Finale: {r['mean_return']:.2f} (± {r['std_return']:.2f}) | Entropia: {r['entropy']:.3f}\n")

if __name__ == "__main__":
    analyze_runs(top_k=10)