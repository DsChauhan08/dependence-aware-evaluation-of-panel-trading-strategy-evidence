"""
run_prod_parallel.py — Parallel production simulation
======================================================
Runs each DGP as a separate subprocess for robustness.
Uses SDDM_N_SIM=500, SDDM_N_BOOT=2000 (sufficient for coverage SEs < 1pp).
"""
import subprocess, sys, os, time, json

N_SIM = int(os.environ.get("SDDM_N_SIM", 500))
N_BOOT = int(os.environ.get("SDDM_N_BOOT", 2000))
OUTDIR = os.environ.get("AUDIT_OUTPUT_DIR", "output_prod")

# Each DGP runs as a subprocess calling this worker
WORKER_SCRIPT = """
import sys, os, json, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SDDM_N_SIM', '{n_sim}')
os.environ.setdefault('SDDM_N_BOOT', '{n_boot}')

import numpy as np
import pandas as pd
from sddm_bootstrap import PanelData, sddm_inference, _sharpe, effective_sample_size
from simulation_study import generate_panel, DGPConfig, true_sharpe, coverage_experiment
import warnings
warnings.filterwarnings("ignore")

N_SIM = int(os.environ['SDDM_N_SIM'])
N_BOOT = int(os.environ['SDDM_N_BOOT'])
OUTDIR = os.environ.get('AUDIT_OUTPUT_DIR', 'output_prod')
METHODS = ["iid", "blocked", "stationary"]

dgp_name = sys.argv[1]
dgp_params = json.loads(sys.argv[2])

cfg = DGPConfig(**dgp_params)
print(f"Running {{dgp_name}}: {{cfg.name}}, N_SIM={{N_SIM}}, N_BOOT={{N_BOOT}}", flush=True)

t0 = time.time()
df = coverage_experiment(cfg, methods=METHODS, n_simulations=N_SIM, n_boot=N_BOOT)
df["DGP"] = dgp_name
elapsed = time.time() - t0

os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(f"{{OUTDIR}}/coverage_{{dgp_name}}.csv", index=False)

for _, row in df.iterrows():
    print(f"  {{row['Method']}}: coverage={{row['Coverage']:.3f}} ± {{row['Coverage_SE']:.3f}}, CI_width={{row['Mean_CI_Width']:.3f}}", flush=True)
print(f"  Done in {{elapsed:.0f}}s", flush=True)
"""

DGPS = {
    "01_iid":          {"name": "IID baseline",          "T": 1000, "N": 50, "ar1_serial": 0.0, "rho_cross": 0.0},
    "02_ser_mild":     {"name": "Serial mild",           "T": 1000, "N": 50, "ar1_serial": 0.2, "rho_cross": 0.0},
    "03_ser_strong":   {"name": "Serial strong",         "T": 1000, "N": 50, "ar1_serial": 0.5, "rho_cross": 0.0},
    "04_xs_mild":      {"name": "Cross mild",            "T": 1000, "N": 50, "ar1_serial": 0.0, "rho_cross": 0.2},
    "05_xs_strong":    {"name": "Cross strong",          "T": 1000, "N": 50, "ar1_serial": 0.0, "rho_cross": 0.5},
    "06_both_mod":     {"name": "Both moderate",         "T": 1000, "N": 50, "ar1_serial": 0.2, "rho_cross": 0.2},
    "07_both_strong":  {"name": "Both strong",           "T": 1000, "N": 50, "ar1_serial": 0.5, "rho_cross": 0.5},
    "08_realistic":    {"name": "Realistic",             "T": 2500, "N": 100, "ar1_serial": 0.15, "rho_cross": 0.35},
    "09_garch_mild":   {"name": "GARCH mild",            "T": 1000, "N": 50, "ar1_serial": 0.2, "rho_cross": 0.2,
                        "garch_alpha": 0.05, "garch_beta": 0.90},
    "10_garch_strong": {"name": "GARCH strong",          "T": 1000, "N": 50, "ar1_serial": 0.3, "rho_cross": 0.4,
                        "garch_alpha": 0.10, "garch_beta": 0.85},
    "11_multifactor":  {"name": "Multi-factor K=3",      "T": 1000, "N": 50, "ar1_serial": 0.2, "rho_cross": 0.3,
                        "n_factors": 3},
    "12_regime":       {"name": "Regime-switching",       "T": 2000, "N": 50, "ar1_serial": 0.15, "rho_cross": 0.25,
                        "regime_switch": True, "regime_p_stay": 0.98},
    "13_high_ar1":     {"name": "High-persistence AR(1)", "T": 1000, "N": 50, "ar1_serial": 0.7, "rho_cross": 0.3},
    "14_garch_highvol": {"name": "High-volatility GARCH", "T": 1000, "N": 50, "ar1_serial": 0.2, "rho_cross": 0.3,
                        "garch_alpha": 0.15, "garch_beta": 0.80},
}

def run_dgp(name, params):
    """Run a single DGP as subprocess."""
    script = WORKER_SCRIPT.format(n_sim=N_SIM, n_boot=N_BOOT)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    worker_dir = os.path.join(code_dir, OUTDIR, "_workers")
    os.makedirs(worker_dir, exist_ok=True)
    script_path = os.path.join(worker_dir, f"_worker_{name}.py")
    with open(script_path, "w") as f:
        f.write(script)
    cmd = [sys.executable, script_path, name, json.dumps(params)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           cwd=code_dir,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})

def already_done(name):
    path = os.path.join(OUTDIR, f"coverage_{name}.csv")
    return os.path.exists(path) and os.path.getsize(path) > 100

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    # Skip already-completed DGPs
    todo = {k: v for k, v in DGPS.items() if not already_done(k)}
    done = {k for k in DGPS if already_done(k)}
    print(f"Already done: {sorted(done)}")
    print(f"To run: {sorted(todo.keys())}")
    print(f"Settings: N_SIM={N_SIM}, N_BOOT={N_BOOT}, OUTDIR={OUTDIR}")
    print()

    import pandas as pd
    defaults = {
        "true_mu": 0.0004,
        "sigma": 0.015,
        "garch_alpha": 0.0,
        "garch_beta": 0.0,
        "n_factors": 1,
        "regime_switch": False,
        "regime_mu_low": -0.0002,
        "regime_p_stay": 0.98,
    }
    pd.DataFrame([
        {"DGP": name, **defaults, **params}
        for name, params in DGPS.items()
    ]).to_csv(os.path.join(OUTDIR, "dgp_configs.csv"), index=False)

    # Run 2 at a time (avoid OOM on 16GB)
    MAX_PARALLEL = 2
    pending = list(todo.items())
    active = {}
    finished = 0

    while pending or active:
        # Start new jobs
        while pending and len(active) < MAX_PARALLEL:
            name, params = pending.pop(0)
            print(f"Starting {name}...", flush=True)
            active[name] = run_dgp(name, params)

        # Check for completion
        for name in list(active.keys()):
            proc = active[name]
            ret = proc.poll()
            if ret is not None:
                out = proc.stdout.read().decode()
                print(out, flush=True)
                if ret != 0:
                    print(f"  *** {name} FAILED (exit {ret})", flush=True)
                finished += 1
                del active[name]

        if active:
            time.sleep(5)

    # Merge all coverage CSVs
    all_dfs = []
    for name in sorted(DGPS.keys()):
        path = os.path.join(OUTDIR, f"coverage_{name}.csv")
        if os.path.exists(path):
            all_dfs.append(pd.read_csv(path))
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(os.path.join(OUTDIR, "coverage_all_merged.csv"), index=False)
        pivot = merged.pivot_table(values="Coverage", index="DGP", columns="Method")
        pivot.to_csv(os.path.join(OUTDIR, "coverage_pivot.csv"))
        print("\n" + "="*70)
        print("FINAL COVERAGE TABLE")
        print("="*70)
        print(pivot.to_string(float_format="%.3f"))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
