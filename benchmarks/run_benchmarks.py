#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is required. Install it with: python3 -m pip install PyYAML"
        )

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def sanitize_token(value):
    text = str(value).strip()
    text = text.replace("-", "_m")
    text = text.replace("+", "")
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "x"


def number_token(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return sanitize_token(value)

    if math.isfinite(numeric) and numeric.is_integer():
        text = str(int(numeric))
    else:
        text = f"{numeric:g}"
    return sanitize_token(text)


def bool_prm(value):
    return "true" if bool(value) else "false"


def prm_value(value):
    if isinstance(value, bool):
        return bool_prm(value)
    return str(value)


def reynolds_to_nu(defaults, config, reynolds):
    table = defaults.get("reynolds_viscosity", {})
    for key, value in table.items():
        if float(key) == float(reynolds):
            return value

    reference_velocity = float(config.get("reference_velocity", 1.0))
    reference_length = float(config.get("reference_length", 0.1))
    return reference_velocity * reference_length / float(reynolds)


def apply_sweep_value(config, key, value, defaults, variant):
    variant["sweep_keys"].append(key)

    if key == "stabilization":
        config.update({k: v for k, v in value.items() if k != "name"})
        variant["stabilization_name"] = value.get("name", "stabilization")
        return

    if key == "nonlinear":
        config.update({k: v for k, v in value.items() if k != "name"})
        variant["nonlinear_name"] = value.get("name", "nonlinear")
        return

    if key == "Re":
        if isinstance(value, dict):
            config["Re"] = value["Re"]
            config["nu"] = value.get("nu", reynolds_to_nu(defaults, config, value["Re"]))
        else:
            config["Re"] = value
            config["nu"] = reynolds_to_nu(defaults, config, value)
        return

    if key == "pcd_inner_tolerance":
        config["pcd_velocity_relative_tolerance"] = value
        config["pcd_pressure_relative_tolerance"] = value
        variant["pcd_inner_tolerance"] = value
        return

    if key == "tolerance_profile":
        config.update({k: v for k, v in value.items() if k != "name"})
        variant["tolerance_profile_name"] = value.get("name", "tolerance")
        return

    if key == "config_profile":
        config.update({k: v for k, v in value.items() if k != "name"})
        variant["config_profile_name"] = value.get("name", "profile")
        return

    config[key] = value


def make_run_id(benchmark_id, config, variant):
    parts = [
        benchmark_id,
        str(config.get("preconditioner", "simple")),
        str(config.get("mesh", config.get("mesh_name", "mesh"))),
        "dt" + number_token(config.get("dt")),
        "Re" + number_token(config.get("Re")),
    ]

    if "simple_pressure_relaxation" in variant["sweep_keys"]:
        parts.append("alpha" + number_token(config.get("simple_pressure_relaxation")))
    if "mpi_ranks" in variant["sweep_keys"]:
        parts.append("np" + number_token(config.get("mpi_ranks")))
    if "pcd_inner_tolerance" in variant["sweep_keys"]:
        parts.append("pcdtol" + number_token(variant.get("pcd_inner_tolerance")))
    if "pcd_pressure_relative_tolerance" in variant["sweep_keys"]:
        parts.append("pcdptol" + number_token(config.get("pcd_pressure_relative_tolerance")))

    tolerance_keys = [
        ("simple_velocity_relative_tolerance", "svtol"),
        ("simple_schur_relative_tolerance", "sstol"),
        ("block_triangular_velocity_relative_tolerance", "bvtol"),
        ("block_triangular_schur_relative_tolerance", "bstol"),
        ("pcd_velocity_relative_tolerance", "pcdvtol"),
        ("yosida_relative_tolerance", "ytol"),
    ]
    for key, prefix in tolerance_keys:
        if key in variant["sweep_keys"]:
            parts.append(prefix + number_token(config.get(key)))

    tolerance_profile_name = variant.get("tolerance_profile_name")
    if tolerance_profile_name:
        parts.append(sanitize_token(tolerance_profile_name))

    config_profile_name = variant.get("config_profile_name")
    if config_profile_name:
        parts.append(sanitize_token(config_profile_name))

    stabilization_name = variant.get("stabilization_name")
    if stabilization_name and stabilization_name != "baseline":
        parts.append(sanitize_token(stabilization_name))

    nonlinear_name = variant.get("nonlinear_name")
    if nonlinear_name and nonlinear_name != "oseen":
        parts.append(sanitize_token(nonlinear_name))

    return "_".join(sanitize_token(part) for part in parts)


def materialize_run(benchmark_id, config, defaults, output_root, variant):
    config = deepcopy(config)
    mesh_catalog = defaults.get("mesh_catalog", {})
    mesh_name = str(config.get("mesh", config.get("mesh_name", "unknown")))
    if "mesh_file" in config:
        mesh_file = config["mesh_file"]
    elif mesh_name in mesh_catalog:
        mesh_file = mesh_catalog[mesh_name]
    else:
        mesh_path = Path(mesh_name)
        if mesh_path.parent != Path(".") or mesh_path.suffix:
            mesh_file = mesh_name
        else:
            known_meshes = ", ".join(sorted(str(key) for key in mesh_catalog))
            raise SystemExit(
                f"Unknown mesh key '{mesh_name}'. Add it to defaults.mesh_catalog "
                f"or set mesh_file explicitly. Known mesh keys: {known_meshes}"
            )

    if "nu" not in config and "Re" in config:
        config["nu"] = reynolds_to_nu(defaults, config, config["Re"])

    config["mesh_name"] = mesh_name
    config["mesh_file"] = mesh_file
    config["benchmark_id"] = benchmark_id
    config.setdefault("statistics_start_time", 0.0)
    config.setdefault("write_solution_output", True)
    config["run_id"] = make_run_id(benchmark_id, config, variant)
    config["output_directory"] = str((output_root / config["run_id"]).resolve())
    return config


def expand_benchmark(benchmark_id, benchmark, defaults, args, output_root, respect_enabled=True):
    if respect_enabled and benchmark.get("enabled", True) is False:
        print(f"Skipping disabled benchmark {benchmark_id}: {benchmark.get('todo', '')}")
        return []

    base_config = defaults.get("base_config", {})
    config = deep_merge(base_config, benchmark.get("config", {}))
    sweep = deepcopy(benchmark.get("sweep", {}))

    if args.preconditioners:
        sweep["preconditioner"] = args.preconditioners
    if args.dts:
        sweep["dt"] = args.dts
    if args.meshes:
        sweep["mesh"] = args.meshes
    if args.reynolds:
        sweep["Re"] = args.reynolds
    if args.mpi_ranks_list:
        sweep["mpi_ranks"] = args.mpi_ranks_list
    if args.final_time is not None:
        config["final_time"] = args.final_time

    keys = list(sweep.keys())
    values = [as_list(sweep[key]) for key in keys]
    combos = itertools.product(*values) if keys else [()]

    runs = []
    for combo in combos:
        run_config = deepcopy(config)
        variant = {"sweep_keys": []}
        for key, value in zip(keys, combo):
            apply_sweep_value(run_config, key, value, defaults, variant)
        runs.append(materialize_run(benchmark_id, run_config, defaults, output_root, variant))

    return runs


def apply_enabled_file(benchmarks, enabled_data):
    if not enabled_data:
        return

    enabled_map = enabled_data.get("benchmarks", enabled_data)
    for benchmark_id, enabled in enabled_map.items():
        if benchmark_id not in benchmarks:
            raise SystemExit(f"Unknown benchmark id in enabled file: {benchmark_id}")
        benchmarks[benchmark_id]["enabled"] = bool(enabled)


def expand_aliases(selected, benchmarks):
    expanded = []

    def visit(benchmark_id, stack):
        if benchmark_id not in benchmarks:
            raise SystemExit(f"Unknown benchmark id: {benchmark_id}")
        if benchmark_id in stack:
            chain = " -> ".join(stack + [benchmark_id])
            raise SystemExit(f"Benchmark alias cycle: {chain}")

        aliases = benchmarks[benchmark_id].get("aliases")
        if aliases:
            for alias in aliases:
                visit(alias, stack + [benchmark_id])
        else:
            expanded.append(benchmark_id)

    for benchmark_id in selected:
        visit(benchmark_id, [])

    return expanded


def list_benchmarks(benchmarks):
    for benchmark_id, benchmark in benchmarks.items():
        marker = ""
        if benchmark.get("enabled", True) is False:
            marker = " [disabled]"
        elif benchmark.get("aliases"):
            marker = " [group]"
        print(f"{benchmark_id:>4}  {benchmark.get('description', '')}{marker}")


def write_prm(path, config):
    boundary_ids = config.get("boundary_ids", {})
    dimension = int(config.get("dimension", 2))

    lines = []
    lines.extend([
        "subsection Mesh and discretization",
        f"  set Dimension = {dimension}",
        f"  set Mesh file = {config['mesh_file']}",
        f"  set Velocity degree = {config['velocity_degree']}",
        f"  set Pressure degree = {config['pressure_degree']}",
        f"  set Final time = {config['final_time']}",
        f"  set Theta = {config['theta']}",
        f"  set Time step = {config['dt']}",
        "end",
        "",
        "subsection Solver",
        f"  set Nonlinear method = {config['nonlinear_method']}",
        f"  set Nonlinear iterations = {config['nonlinear_iterations']}",
        f"  set Nonlinear tolerance = {config['nonlinear_tolerance']}",
        f"  set Picard relaxation = {config['picard_relaxation']}",
        f"  set GMRES restart length = {config['gmres_restart_length']}",
        f"  set Pressure regularization = {config['pressure_regularization']}",
        f"  set Linear max iterations = {config['linear_max_iterations']}",
        f"  set Linear relative tolerance = {config['linear_relative_tolerance']}",
        f"  set Linear absolute tolerance = {config['linear_absolute_tolerance']}",
        f"  set Preconditioner = {config['preconditioner']}",
        f"  set SIMPLE pressure relaxation = {config['simple_pressure_relaxation']}",
        f"  set Block triangular velocity max iterations = {config['block_triangular_velocity_max_iterations']}",
        f"  set Block triangular Schur max iterations = {config['block_triangular_schur_max_iterations']}",
        f"  set Block triangular velocity relative tolerance = {config['block_triangular_velocity_relative_tolerance']}",
        f"  set Block triangular Schur relative tolerance = {config['block_triangular_schur_relative_tolerance']}",
        f"  set SIMPLE velocity max iterations = {config['simple_velocity_max_iterations']}",
        f"  set SIMPLE Schur max iterations = {config['simple_schur_max_iterations']}",
        f"  set SIMPLE velocity relative tolerance = {config['simple_velocity_relative_tolerance']}",
        f"  set SIMPLE Schur relative tolerance = {config['simple_schur_relative_tolerance']}",
        f"  set PCD velocity max iterations = {config['pcd_velocity_max_iterations']}",
        f"  set PCD pressure max iterations = {config['pcd_pressure_max_iterations']}",
        f"  set PCD velocity relative tolerance = {config['pcd_velocity_relative_tolerance']}",
        f"  set PCD pressure relative tolerance = {config['pcd_pressure_relative_tolerance']}",
        f"  set Yosida velocity max iterations = {config['yosida_velocity_max_iterations']}",
        f"  set Yosida Schur max iterations = {config['yosida_schur_max_iterations']}",
        f"  set Yosida correction max iterations = {config['yosida_correction_max_iterations']}",
        f"  set Yosida relative tolerance = {config['yosida_relative_tolerance']}",
        f"  set Preconditioner absolute tolerance = {config['preconditioner_absolute_tolerance']}",
        f"  set Yosida absolute tolerance = {config['yosida_absolute_tolerance']}",
        "end",
        "",
        "subsection Stabilization",
        f"  set Temam = {prm_value(config['temam'])}",
        f"  set Grad-div = {prm_value(config['grad_div'])}",
        f"  set Grad-div coefficient = {config['gamma_grad_div']}",
        f"  set SUPG = {prm_value(config['supg'])}",
        "end",
        "",
        "subsection Physics",
        f"  set Viscosity = {config['nu']}",
        f"  set Inlet velocity = {config['inlet_velocity']}",
        f"  set Inlet channel height = {config['inlet_channel_height']}",
    ])

    if dimension == 3:
        lines.append(f"  set Inlet channel width = {config['inlet_channel_width']}")

    lines.extend([
        f"  set Inlet ramp time = {config['inlet_ramp_time']}",
        f"  set Outlet pressure = {config['outlet_pressure']}",
        "end",
        "",
        "subsection Force coefficients",
        f"  set Reference velocity = {config['reference_velocity']}",
        f"  set Reference length = {config['reference_length']}",
    ])

    if dimension == 3:
        lines.append(f"  set Reference span = {config['reference_span']}")

    lines.extend([
        "end",
        "",
        "subsection Boundary ids",
        f"  set Inlet = {boundary_ids.get('inlet', 1)}",
        f"  set Outlet = {boundary_ids.get('outlet', 2)}",
        f"  set Walls = {boundary_ids.get('walls', 3)}",
        f"  set Cylinder = {boundary_ids.get('cylinder', 5)}",
        "end",
        "",
        "subsection Output",
        f"  set Output directory = {config['output_directory']}",
        f"  set Run id = {config['run_id']}",
        f"  set Benchmark id = {config['benchmark_id']}",
        f"  set Mesh name = {config['mesh_name']}",
        f"  set Write solution output = {prm_value(config['write_solution_output'])}",
        f"  set Statistics start time = {config['statistics_start_time']}",
        "end",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_config_json(path, config, command):
    data = deepcopy(config)
    data["command"] = command
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def csv_float(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def csv_int(row, key):
    value = csv_float(row, key)
    if not math.isfinite(value):
        return None
    return int(value)


def strong_scaling_group_key(row):
    return (
        row.get("benchmark_id", ""),
        row.get("preconditioner", ""),
        row.get("mesh_name", ""),
        row.get("dt", ""),
        row.get("nu", ""),
        row.get("Re", ""),
        row.get("nonlinear_method", ""),
    )


def enrich_scaling_metrics(rows):
    baselines = {}
    for row in rows:
        if row.get("benchmark_id") != "P8":
            continue
        if csv_int(row, "mpi_ranks") != 1:
            continue

        runtime = csv_float(row, "total_runtime")
        gmres_mean = csv_float(row, "gmres_mean")
        if math.isfinite(runtime) and runtime > 0.0:
            baselines[strong_scaling_group_key(row)] = {
                "runtime": runtime,
                "gmres_mean": gmres_mean,
            }

    for row in rows:
        row.setdefault("speedup", "")
        row.setdefault("efficiency", "")
        row.setdefault("iteration_growth", "")
        if row.get("benchmark_id") != "P8":
            continue

        baseline = baselines.get(strong_scaling_group_key(row))
        mpi_ranks = csv_int(row, "mpi_ranks")
        runtime = csv_float(row, "total_runtime")
        gmres_mean = csv_float(row, "gmres_mean")
        if baseline is None or mpi_ranks is None or mpi_ranks <= 0:
            continue
        if not math.isfinite(runtime) or runtime <= 0.0:
            continue

        speedup = baseline["runtime"] / runtime
        row["speedup"] = f"{speedup:.16g}"
        row["efficiency"] = f"{speedup / float(mpi_ranks):.16g}"

        baseline_gmres = baseline.get("gmres_mean", math.nan)
        if math.isfinite(gmres_mean) and math.isfinite(baseline_gmres) and baseline_gmres > 0.0:
            row["iteration_growth"] = f"{gmres_mean / baseline_gmres:.16g}"


def aggregate_summaries(output_root):
    summaries = sorted(output_root.glob("*/summary.csv"))
    target = output_root / "all_summaries.csv"

    if not summaries:
        return

    rows = []
    fieldnames = []
    for path in summaries:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                for field in row:
                    if field not in fieldnames:
                        fieldnames.append(field)

    enrich_scaling_metrics(rows)
    for field in ["speedup", "efficiency", "iteration_growth"]:
        if field not in fieldnames:
            fieldnames.append(field)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_one(config, solver, solver_cwd, mpirun, dry_run):
    output_dir = Path(config["output_directory"])
    prm_path = output_dir / "run.prm"
    command = [mpirun, "-np", str(config.get("mpi_ranks", 1)), str(solver), str(prm_path)]

    if dry_run:
        print(f"[dry-run] cd {solver_cwd} && {' '.join(command)}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_prm(prm_path, config)
    write_config_json(output_dir / "config.json", config, command)

    failed_path = output_dir / "FAILED"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=str(solver_cwd), stdout=stdout, stderr=stderr)

    if result.returncode != 0:
        failed_path.write_text(
            f"Command failed with return code {result.returncode}\n"
            f"Command: {' '.join(command)}\n",
            encoding="utf-8",
        )
    elif failed_path.exists():
        failed_path.unlink()

    return result.returncode


def validation_setup_label(config):
    benchmark_id = str(config.get("benchmark_id", ""))
    if not benchmark_id.startswith("V"):
        return ""

    fields = []
    if "inlet_velocity" in config:
        fields.append(f"Uin={config['inlet_velocity']}")
    if "Re" in config:
        fields.append(f"Re={config['Re']}")
    if "inlet_ramp_time" in config:
        try:
            ramp_time = float(config["inlet_ramp_time"])
        except (TypeError, ValueError):
            ramp_time = 0.0
        if ramp_time > 0.0:
            fields.append(f"ramp=yes, Tramp={config['inlet_ramp_time']}")
        else:
            fields.append("ramp=no")

    return f" ({', '.join(fields)})" if fields else ""


def parse_args():
    parser = argparse.ArgumentParser(description="Run Navier-Stokes benchmark sweeps.")
    parser.add_argument("--yaml", default="benchmarks/benchmarks.yaml", help="Benchmark YAML file.")
    parser.add_argument("--enabled-file", default="benchmarks/enabled.yaml", help="Benchmark enable/disable YAML file.")
    parser.add_argument("--list", action="store_true", help="List available benchmarks.")
    parser.add_argument("--all", action="store_true", help="Run all enabled non-group benchmarks.")
    parser.add_argument("--benchmarks", nargs="+", help="Benchmark ids to run.")
    parser.add_argument("--preconditioners", nargs="+", help="Override preconditioner sweep.")
    parser.add_argument("--dts", nargs="+", type=float, help="Override timestep sweep.")
    parser.add_argument("--meshes", nargs="+", help="Override mesh sweep.")
    parser.add_argument("--reynolds", nargs="+", type=float, help="Override Reynolds sweep.")
    parser.add_argument("--mpi-ranks-list", nargs="+", type=int, help="Override MPI ranks sweep.")
    parser.add_argument("--final-time", type=float, help="Override final time for all selected runs.")
    parser.add_argument("--solver", help="Path to solver executable.")
    parser.add_argument("--mpirun", help="MPI launcher command.")
    parser.add_argument("--output-root", help="Output root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with summary.csv and no FAILED marker.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first failed run.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_yaml(resolve_path(args.yaml))
    defaults = data.get("defaults", {})
    benchmarks = data.get("benchmarks", {})
    enabled_path = resolve_path(args.enabled_file)
    if enabled_path.exists():
        apply_enabled_file(benchmarks, load_yaml(enabled_path))

    if args.list:
        list_benchmarks(benchmarks)
        return 0

    if args.all:
        selected = [
            benchmark_id
            for benchmark_id, benchmark in benchmarks.items()
            if benchmark.get("enabled", True) is not False and not benchmark.get("aliases")
        ]
    elif args.benchmarks:
        selected = expand_aliases(args.benchmarks, benchmarks)
    else:
        print("Specify --list, --all, or --benchmarks.", file=sys.stderr)
        return 2

    output_root = resolve_path(args.output_root or defaults.get("output_root", "benchmark_results"))
    solver = resolve_path(args.solver or defaults.get("executable", "build/main"))
    solver_cwd = solver.parent if solver.parent.exists() else REPO_ROOT
    mpirun = args.mpirun or defaults.get("mpirun", "mpirun")

    if not args.dry_run and not solver.exists():
        print(f"Solver executable not found: {solver}", file=sys.stderr)
        return 2

    runs = []
    for benchmark_id in selected:
        runs.extend(
            expand_benchmark(
                benchmark_id,
                benchmarks[benchmark_id],
                defaults,
                args,
                output_root,
                respect_enabled=args.all,
            )
        )

    seen = set()
    unique_runs = []
    for run in runs:
        if run["run_id"] in seen:
            continue
        seen.add(run["run_id"])
        unique_runs.append(run)

    failures = 0
    for index, config in enumerate(unique_runs, start=1):
        output_dir = Path(config["output_directory"])
        summary_path = output_dir / "summary.csv"
        failed_path = output_dir / "FAILED"

        if args.skip_existing and summary_path.exists() and not failed_path.exists():
            print(f"[{index}/{len(unique_runs)}] skip existing {config['run_id']}{validation_setup_label(config)}")
            continue

        print(f"[{index}/{len(unique_runs)}] run {config['run_id']}{validation_setup_label(config)}")
        returncode = run_one(config, solver, solver_cwd, mpirun, args.dry_run)
        if returncode != 0:
            failures += 1
            print(f"  failed with return code {returncode}")
            if args.stop_on_failure:
                break

    if not args.dry_run:
        aggregate_summaries(output_root)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
