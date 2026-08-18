#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run_command(cmd: list[str], cwd: Path, logs: Path, name: str, stop_on_error: bool) -> bool:
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    print(f"\n[{ts()}] === {name} ===")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# {name}\n# started: {ts()}\n# cwd: {cwd}\n# command: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        code = proc.wait()
        f.write(f"\n# finished: {ts()}\n# exit_code: {code}\n")
    if code != 0:
        print(f"[ERROR] {name} failed with exit code {code}. Log: {log_path}")
        if stop_on_error:
            raise SystemExit(code)
        return False
    print(f"[OK] {name}")
    return True


def require(project: Path, relative: str, missing: list[str]) -> None:
    if not (project / relative).exists():
        missing.append(relative)


def zip_results(project: Path, output_name: str) -> None:
    paths = [project / "experiments" / "results", project / "outputs"]
    zip_path = project / output_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root in paths:
            if not root.exists():
                continue
            for file in root.rglob("*"):
                if file.is_file():
                    z.write(file, arcname=str(file.relative_to(project)))
    print(f"[OK] packed results to {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thesis experiment analyses.")
    parser.add_argument("--project-root", default=".", help="Repository root containing experiments/run_experiments.py.")
    parser.add_argument("--run-main", choices=["none", "evaluate", "all"], default="none",
                        help="Optionally run the main rating experiment before the analyses.")
    parser.add_argument("--profile", default="max", help="Profile passed to run_experiments.py when --run-main is used.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when a script fails.")
    parser.add_argument("--skip-latent-pool-structure", action="store_true", help="Skip analyses requiring games.parquet and games_unique.parquet.")
    parser.add_argument("--skip-heavy-shocks", action="store_true", help="Skip shock replay analyses.")
    parser.add_argument("--zip-name", default="thesis_experiment_outputs.zip")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    code_root = Path(__file__).resolve().parent
    analysis = code_root / "analysis"
    logs = project / "outputs" / "logs"
    py = sys.executable
    main_script = code_root / "run_experiments.py"

    print(f"[INFO] project root: {project}")
    print(f"[INFO] analysis code: {analysis}")
    print(f"[INFO] python: {py}")

    missing = []
    require(project, "experiments/run_experiments.py", missing)
    require(project, "experiments/results/best_model_parameters.json", missing)
    require(project, "experiments/results/final_test_comparison.csv", missing)
    require(project, "experiments/results/final_scope_metrics.csv", missing)
    require(project, "experiments/results/final_monthly_metrics.csv", missing)
    require(project, "experiments/cache", missing)

    if missing and args.run_main == "none":
        print("[ERROR] Missing required core files:")
        for item in missing:
            print(f"  - {item}")
        print("\nRun the main experiment first, or rerun this command with --run-main evaluate/all.")
        raise SystemExit(1)

    if args.run_main != "none":
        cmd = [py, str(main_script), "--mode", args.run_main, "--profile", args.profile]
        run_command(cmd, project, logs, f"00_main_experiment_{args.run_main}", args.stop_on_error)

    out_pred = project / "outputs" / "prediction_sensitivity"
    out_rob = project / "outputs" / "robustness_ablation"
    out_pool = project / "outputs" / "latent_pool_structure"
    out_plots = project / "outputs" / "thesis_plots"

    r1 = analysis / "prediction_sensitivity"
    commands = [
        ("01_expected_score_curves", [py, str(r1 / "expected_score_curves.py"),
                                      "--best", str(project / "experiments/results/best_model_parameters.json"),
                                      "--out", str(out_pred)]),
        ("02_monthly_improvement_summary", [py, str(r1 / "monthly_improvement_summary.py"),
                                            "--monthly", str(project / "experiments/results/final_monthly_metrics.csv"),
                                            "--out", str(out_pred)]),
        ("03_pool_weight_sensitivity", [py, str(r1 / "pool_weight_sensitivity.py"),
                                        "--module", str(main_script),
                                        "--best", str(project / "experiments/results/best_model_parameters.json"),
                                        "--out", str(out_pred)]),
        ("04_provisional_entry_sensitivity", [py, str(r1 / "provisional_entry_sensitivity.py"),
                                              "--module", str(main_script),
                                              "--best", str(project / "experiments/results/best_model_parameters.json"),
                                              "--out", str(out_pred)]),
        ("05_calibration_bins", [py, str(r1 / "calibration_bins.py"),
                                 "--module", str(main_script),
                                 "--best", str(project / "experiments/results/best_model_parameters.json"),
                                 "--out", str(out_pred)]),
    ]
    if not args.skip_heavy_shocks:
        commands.append(("06_global_rating_shock_recovery", [py, str(r1 / "global_rating_shock_recovery.py"),
                                                             "--module", str(main_script),
                                                             "--best", str(project / "experiments/results/best_model_parameters.json"),
                                                             "--out", str(out_pred),
                                                             "--shock-month", "2022-01",
                                                             "--pool-id", "-1"]))

    r2 = analysis / "robustness_ablation"
    commands.extend([
        ("07_ablation_component_report", [py, str(r2 / "ablation_component_report.py"),
                                          "--comparison", str(project / "experiments/results/final_test_comparison.csv"),
                                          "--out", str(out_rob)]),
        ("08_generalization_gap", [py, str(r2 / "generalization_gap.py"),
                                   "--scope", str(project / "experiments/results/final_scope_metrics.csv"),
                                   "--out", str(out_rob)]),
        ("09_pool_offsets_interpretation", [py, str(r2 / "pool_offsets_interpretation.py"),
                                            "--pool-offsets", str(project / "experiments/results/final_pool_offsets.csv"),
                                            "--pair-offsets", str(project / "experiments/results/final_pool_pair_offsets.csv"),
                                            "--summary", str(project / "experiments/results/latent_pool_summary.csv"),
                                            "--out", str(out_rob)]),
        ("10_rating_scale_diagnostics", [py, str(r2 / "rating_scale_diagnostics.py"),
                                         "--monthly-ratings", str(project / "experiments/results/monthly_rating_distributions.csv"),
                                         "--out", str(out_rob)]),
        ("13_k_scale_sensitivity", [py, str(r2 / "k_scale_sensitivity.py"),
                                     "--module", str(main_script),
                                     "--best", str(project / "experiments/results/best_model_parameters.json"),
                                     "--out", str(out_rob)]),
        ("14_white_advantage_sensitivity", [py, str(r2 / "white_advantage_sensitivity.py"),
                                            "--module", str(main_script),
                                            "--best", str(project / "experiments/results/best_model_parameters.json"),
                                            "--out", str(out_rob)]),
        ("15_model_simplification_report", [py, str(r2 / "model_simplification_report.py"),
                                            "--comparison", str(project / "experiments/results/final_test_comparison.csv"),
                                            "--out", str(out_rob)]),
    ])
    if not args.skip_heavy_shocks:
        commands.append(("11_pool_specific_shock_batch", [py, str(r2 / "pool_specific_shock_batch.py"),
                                                          "--module", str(main_script),
                                                          "--best", str(project / "experiments/results/best_model_parameters.json"),
                                                          "--summary", str(project / "experiments/results/latent_pool_summary.csv"),
                                                          "--out", str(out_rob),
                                                          "--shock-month", "2022-01",
                                                          "--top-n", "5"]))
        commands.append(("12_sustained_recovery_summary", [py, str(r2 / "sustained_shock_recovery.py"),
                                                           "--monthly", str(out_pred / "results/rating_shock_recovery_monthly.csv"),
                                                           "--pool-monthly", str(out_rob / "results/pool_specific_shock_monthly.csv"),
                                                           "--out", str(out_rob),
                                                           "--shock-month", "2022-01"]))

    if not args.skip_latent_pool_structure:
        pool_missing = []
        require(project, "games.parquet", pool_missing)
        require(project, "games_unique.parquet", pool_missing)
        if pool_missing:
            print("[WARN] Skipping latent-pool structure analyses because these inputs are missing:")
            for item in pool_missing:
                print(f"  - {item}")
        else:
            r3 = analysis / "latent_pool_structure"
            commands.extend([
                ("16_pool_federation_profile", [py, str(r3 / "pool_federation_profile.py"),
                                                "--module", str(main_script),
                                                "--games", str(project / "games.parquet"),
                                                "--out", str(out_pool)]),
                ("17_pool_age_junior_profile", [py, str(r3 / "pool_age_junior_profile.py"),
                                                "--module", str(main_script),
                                                "--games", str(project / "games.parquet"),
                                                "--out", str(out_pool)]),
                ("18_elite_top_players_snapshot", [py, str(r3 / "elite_top_players_snapshot.py"),
                                                   "--module", str(main_script),
                                                   "--games", str(project / "games.parquet"),
                                                   "--out", str(out_pool),
                                                   "--top-n", "200",
                                                   "--min-rating", "2400"]),
                ("19_federation_pool_alignment", [py, str(r3 / "federation_pool_alignment.py"),
                                                  "--module", str(main_script),
                                                  "--games", str(project / "games.parquet"),
                                                  "--out", str(out_pool),
                                                  "--min-fed-players", "50"]),
                ("20_cross_pool_interactions", [py, str(r3 / "cross_pool_interactions.py"),
                                                "--module", str(main_script),
                                                "--games-unique", str(project / "games_unique.parquet"),
                                                "--out", str(out_pool),
                                                "--from-month", "2015-01",
                                                "--top-pools", "15"]),
                ("21_junior_participation_over_time", [py, str(r3 / "junior_participation_over_time.py"),
                                                       "--games", str(project / "games.parquet"),
                                                       "--games-unique", str(project / "games_unique.parquet"),
                                                       "--out", str(out_pool),
                                                       "--junior-age", "21"]),
                ("22_sex_field_pool_profile_optional", [py, str(r3 / "sex_field_pool_profile.py"),
                                                        "--module", str(main_script),
                                                        "--games", str(project / "games.parquet"),
                                                        "--out", str(out_pool)]),
            ])

    commands.append(("90_make_thesis_plots", [py, str(code_root / "make_thesis_plots.py"),
                                             "--project-root", str(project),
                                             "--out", str(out_plots.relative_to(project)),
                                             "--start-month", "2015-01"]))

    successes = 0
    failures = 0
    for name, cmd in commands:
        ok = run_command(cmd, project, logs, name, args.stop_on_error)
        successes += int(ok)
        failures += int(not ok)

    zip_results(project, args.zip_name)

    print("\n=== SUMMARY ===")
    print(f"successful scripts: {successes}")
    print(f"failed scripts:     {failures}")
    print(f"logs:              {logs}")
    print(f"results zip:       {project / args.zip_name}")

    if failures:
        print("\nSome scripts failed. Inspect logs and partial outputs.")
    else:
        print("\nAll selected scripts completed.")


if __name__ == "__main__":
    main()
