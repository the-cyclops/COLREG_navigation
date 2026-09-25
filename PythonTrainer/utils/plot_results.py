import os
import glob
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Parametri grafici richiesti
FONT_SIZE = 22
TICK_LEN = 10
TICK_WDT = 3
NUM_INTERP_STEPS = 500  # Numero di punti per allineare i seed lungo l'asse X
X_SCALE = 50000.0       # Fattore di scala asse X (5 * 10^4)

pio.templates.default = "simple_white"

# Mappa metodi, percorsi e colori (percorsi relativi da utils/)
METHODS_CONFIG = {
    "Our": {
        "dir": "../runs/boat_R6_FIXREWARD_GAMMA_0.995_lr_0.0003_ent_0.001_batchsize_256_costscale_0.1_reward_scale_1.0",
        "color": "#D81B60",
        "highlight": True
    },
    "Reward shaping": {
        "dir": "../runs/boat_R6_REWARDSHAPING_GAMMA_0.995_lr_0.0003_ent_0.001_batchsize_256_costscale_0.1_reward_scale_1.0",
        "color": "#6B4F3A",
        "highlight": False
    },
    "CMORL original": {
        "dir": "../runs/boat_R6_CMORL_GAMMA_0.995_lr_0.0003_ent_0.001_batchsize_256_costscale_0.1_reward_scale_1.0",
        "color": "#ff7f0e",
        "highlight": False
    }
}

TAGS = {
    "return_mean": "Training/Mean_Return",
    "cost_r1": "Training/Smoothed_Ep_Cost_R1",
    "cost_r2": "Training/Smoothed_Ep_Cost_R2",
    "cost_r6": "Training/Smoothed_Ep_Cost_R6",
    "pos_cost_r1": "Training/Smoothed_Ep_Pos_Cost_R1",
    "pos_cost_r2": "Training/Smoothed_Ep_Pos_Cost_R2",
    "pos_cost_r6": "Training/Smoothed_Ep_Pos_Cost_R6",
    "eval_return": "Eval/Mean_Eval_Return",
    "rho_r1": "Training/Robustness_R1_Physics",
    "rho_r2": "Training/Robustness_R2_Physics",
    "rho_r6": "Training/Robustness_R6_Physics"
}


def hex_to_rgb_tuple_str(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    return ",".join(str(int(hex_color[i:i + 2], 16)) for i in (0, 2, 4))


def ema_smooth(arr: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """EMA coerente con l'implementazione del professore."""
    smoothed = np.zeros_like(arr, dtype=float)
    smoothed[0] = arr[0]
    for t in range(1, len(arr)):
        smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * arr[t]
    return smoothed


def extract_scalar_from_seed(seed_dir: str, tag: str):
    """Estrae (step, valori) da una cartella di log TensorBoard."""
    ea = EventAccumulator(seed_dir, size_guidance={"scalars": 0})
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_and_aggregate_metrics(methods_config: dict, tot_steps: int = None, start_safety: int = None):
    """Estrae i dati dei 5 seed per ciascun metodo e li allinea su un unico asse X scalato."""
    parsed_data = {}

    max_step_found = 0
    raw_runs = {}

    for label, conf in methods_config.items():
        base_dir = conf["dir"]
        seed_dirs = sorted(glob.glob(os.path.join(base_dir, "seed_*")))
        raw_runs[label] = []

        for s_dir in seed_dirs:
            seed_metrics = {}
            for metric_key, tag in TAGS.items():
                steps, vals = extract_scalar_from_seed(s_dir, tag)
                if steps is not None and len(steps) > 0:
                    seed_metrics[metric_key] = (steps, vals)
                    if steps[-1] > max_step_found:
                        max_step_found = steps[-1]
            if seed_metrics:
                raw_runs[label].append(seed_metrics)

    target_max_step = tot_steps if tot_steps is not None else max_step_found
    common_steps = np.linspace(0, target_max_step, NUM_INTERP_STEPS)
    # Asse X per evaluation (da start_safety a fine)
    start_eval = start_safety if start_safety is not None else 0
    common_eval_steps = np.linspace(start_eval, target_max_step, NUM_INTERP_STEPS)

    for label, runs in raw_runs.items():
        parsed_data[label] = {
            "return": [],
            "eval_return": [],
            "total_cost": [],
            "total_pos_cost": [],
            "rho_r1": [],
            "rho_r2": [],
            "rho_r6": []
        }

        for seed_data in runs:
            # 1. Training Return: Mean Return grezzo con smoothing EMA del prof (alpha=0.9)
            if "return_mean" in seed_data:
                st, vl = seed_data["return_mean"]
                interp_val = np.interp(common_steps, st, vl)
                parsed_data[label]["return"].append(ema_smooth(interp_val, alpha=0.9))

            # 2. Evaluation Return
            if "eval_return" in seed_data:
                st, vl = seed_data["eval_return"]
                interp_val = np.interp(common_eval_steps, st, vl)
                parsed_data[label]["eval_return"].append(interp_val)

            # 3. Costi totali standard
            if all(k in seed_data for k in ["cost_r1", "cost_r2", "cost_r6"]):
                st1, c1 = seed_data["cost_r1"]
                st2, c2 = seed_data["cost_r2"]
                st6, c6 = seed_data["cost_r6"]

                c1_i = np.interp(common_steps, st1, c1)
                c2_i = np.interp(common_steps, st2, c2)
                c6_i = np.interp(common_steps, st6, c6)
                total_c = c1_i + c2_i + c6_i
                parsed_data[label]["total_cost"].append(total_c)

            # 4. Costi positivi totali (solo violazioni)
            if all(k in seed_data for k in ["pos_cost_r1", "pos_cost_r2", "pos_cost_r6"]):
                st1, c1 = seed_data["pos_cost_r1"]
                st2, c2 = seed_data["pos_cost_r2"]
                st6, c6 = seed_data["pos_cost_r6"]

                c1_i = np.interp(common_steps, st1, c1)
                c2_i = np.interp(common_steps, st2, c2)
                c6_i = np.interp(common_steps, st6, c6)
                total_pos_c = c1_i + c2_i + c6_i
                parsed_data[label]["total_pos_cost"].append(total_pos_c)

            # 5. Robustness grezza con EMA (alpha=0.7)
            for r_key in ["rho_r1", "rho_r2", "rho_r6"]:
                if r_key in seed_data:
                    st, vl = seed_data[r_key]
                    r_interp = np.interp(common_steps, st, vl)
                    parsed_data[label][r_key].append(ema_smooth(r_interp, alpha=0.7))

        for metric_k in parsed_data[label]:
            if len(parsed_data[label][metric_k]) > 0:
                parsed_data[label][metric_k] = np.array(parsed_data[label][metric_k])
            else:
                parsed_data[label][metric_k] = None

    scaled_steps = common_steps / X_SCALE
    scaled_eval_steps = common_eval_steps / X_SCALE
    return scaled_steps, scaled_eval_steps, parsed_data


def add_metric_trace(fig, steps, arr, label, color, highlight=False, showlegend=True):
    if arr is None or len(arr) == 0:
        return
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    sem = arr.std(axis=0) / np.sqrt(n)
    ci95 = 1.96 * sem

    rgb_str = hex_to_rgb_tuple_str(color)
    opacity = 0.15 if highlight else 0.10
    width = 2 if highlight else 1

    # Banda di incertezza (IC 95%)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([mean - ci95, (mean + ci95)[::-1]]),
            fill="toself",
            fillcolor=f"rgba({rgb_str},{opacity})",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False
        )
    )
    # Linea della media
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean,
            mode="lines",
            name=label,
            line=dict(color=f"rgb({rgb_str})", width=width),
            showlegend=showlegend,
            legendrank=0 if highlight else 1
        )
    )


def apply_custom_layout(fig, y_title: str, start_safety_scaled: float = None):
    fig.update_layout(
        height=600,
        width=1000,
        font=dict(family="Helvetica, sans-serif", size=FONT_SIZE),
        xaxis_title="Timesteps 5 × 10<sup>4</sup>",
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)
    fig.update_yaxes(ticks="outside", ticklen=TICK_LEN, tickwidth=TICK_WDT)

    if start_safety_scaled is not None:
        fig.add_vline(
            x=start_safety_scaled,
            line=dict(color="green", dash="dash", width=2),
            annotation_text="start safety",
            annotation_position="bottom right",
            annotation_font_color="green",
            annotation_ax=30
        )


def generate_plots(scaled_steps, scaled_eval_steps, parsed_data, start_safety: int = None, output_dir: str = "../results_plots"):
    os.makedirs(output_dir, exist_ok=True)
    start_safety_scaled = (start_safety / X_SCALE) if start_safety is not None else None

    # 1. Grafico Training Return
    fig_ret = go.Figure()
    for label, conf in METHODS_CONFIG.items():
        add_metric_trace(
            fig_ret, scaled_steps, parsed_data[label]["return"],
            label=label, color=conf["color"], highlight=conf["highlight"]
        )
    apply_custom_layout(fig_ret, "Return", start_safety_scaled)
    fig_ret.write_image(os.path.join(output_dir, "training_return.png"), scale=2)
    fig_ret.write_html(os.path.join(output_dir, "training_return.html"))

    # 2. Eval Return (inizia da start_safety)
    fig_eval_ret = go.Figure()
    for label, conf in METHODS_CONFIG.items():
        add_metric_trace(fig_eval_ret, scaled_eval_steps, parsed_data[label]["eval_return"], label=label, color=conf["color"], highlight=conf["highlight"])
    apply_custom_layout(fig_eval_ret, "Eval Return", start_safety_scaled)
    fig_eval_ret.write_image(os.path.join(output_dir, "eval_return.png"), scale=2)
    fig_eval_ret.write_html(os.path.join(output_dir, "eval_return.html"))

    # 3. Grafico Costi Totali
    fig_cost = go.Figure()
    for label, conf in METHODS_CONFIG.items():
        add_metric_trace(
            fig_cost, scaled_steps, parsed_data[label]["total_cost"],
            label=label, color=conf["color"], highlight=conf["highlight"]
        )
    apply_custom_layout(fig_cost, "Total cost", start_safety_scaled)
    fig_cost.write_image(os.path.join(output_dir, "training_total_cost.png"), scale=2)
    fig_cost.write_html(os.path.join(output_dir, "training_total_cost.html"))

    # 4. Grafico Positive Cost Totale
    fig_pos_cost = go.Figure()
    for label, conf in METHODS_CONFIG.items():
        add_metric_trace(
            fig_pos_cost, scaled_steps, parsed_data[label]["total_pos_cost"],
            label=label, color=conf["color"], highlight=conf["highlight"]
        )
    apply_custom_layout(fig_pos_cost, "Total positive cost", start_safety_scaled)
    fig_pos_cost.write_image(os.path.join(output_dir, "training_positive_cost.png"), scale=2)
    fig_pos_cost.write_html(os.path.join(output_dir, "training_positive_cost.html"))

    # 5. Grafici separati per ogni Robustness
    robustness_metrics = [
        ("rho_r1", "robustness_R1", "R1"),
        ("rho_r2", "robustness_R2", "R2"),
        ("rho_r6", "robustness_R6", "R6")
    ]

    for metric_key, file_name, title in robustness_metrics:
        fig_rho = go.Figure()
        for label, conf in METHODS_CONFIG.items():
            add_metric_trace(
                fig_rho, scaled_steps, parsed_data[label][metric_key],
                label=label, color=conf["color"], highlight=conf["highlight"]
            )
        apply_custom_layout(fig_rho, f"Robustness {title}", start_safety_scaled)
        fig_rho.write_image(os.path.join(output_dir, f"{file_name}.png"), scale=2)
        fig_rho.write_html(os.path.join(output_dir, f"{file_name}.html"))

    print(f"Grafici esportati con successo in: {output_dir}/")


if __name__ == "__main__":
    START_SAFETY_STEP = 1_024_000

    steps, eval_steps, data = load_and_aggregate_metrics(METHODS_CONFIG, start_safety=START_SAFETY_STEP)
    generate_plots(steps, eval_steps, data, start_safety=START_SAFETY_STEP)