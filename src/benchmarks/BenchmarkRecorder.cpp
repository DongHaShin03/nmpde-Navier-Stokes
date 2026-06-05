#include "BenchmarkRecorder.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fs = std::filesystem;

void BenchmarkRecorder::configure(const BenchmarkOutputOptions &options_)
{
    options = options_;
    if (options.output_directory.empty())
        options.output_directory = "benchmark_results/default_run";
    if (options.run_id.empty())
        options.run_id = "default_run";
    if (options.benchmark_id.empty())
        options.benchmark_id = "unknown";
    if (options.mesh_name.empty())
        options.mesh_name = "unknown";
}

void BenchmarkRecorder::initialize(const BenchmarkRunMetadata &metadata_, const bool master_rank_)
{
    metadata = metadata_;
    master_rank = master_rank_;
    run_id = make_run_id(options.run_id);
    lift_history.clear();
    last_strouhal_number = std::numeric_limits<double>::quiet_NaN();
    summary_steps = 0;
    summary_reynolds_number = std::numeric_limits<double>::quiet_NaN();
    summary_cd_max = std::numeric_limits<double>::quiet_NaN();
    summary_cl_max = std::numeric_limits<double>::quiet_NaN();
    summary_cl_abs_max = std::numeric_limits<double>::quiet_NaN();
    summary_delta_pressure_final = std::numeric_limits<double>::quiet_NaN();
    summary_div_l2_sum = 0.0;
    summary_div_l2_max = 0.0;
    summary_gmres_sum = 0.0;
    summary_gmres_max = 0;
    summary_step_time_sum = 0.0;
    summary_failed_step = -1;
    summary_stable = true;

    if (!master_rank)
        return;

    write_config_copy();
    const fs::path metrics_path = fs::path(options.output_directory) / "timeseries.csv";
    std::ofstream file(metrics_path.string(), std::ios::trunc);
    file << "step,time,dt,Re,nu,Cd,Cl,Cs,DeltaP,divL2,divLinf,velocity_L2,velocity_Linf,pressure_L2,pressure_mean,kinetic_energy,gmres_iterations,gmres_final_residual,gmres_converged,assembly_time,preconditioner_setup_time,linear_solve_time,total_step_time\n";
}

void BenchmarkRecorder::record_step(const BenchmarkStepRecord &record)
{
    last_strouhal_number = update_strouhal_number(record);
    update_summary_metrics(record);

    if (!master_rank)
        return;

    const fs::path metrics_path = fs::path(options.output_directory) / "timeseries.csv";
    std::ofstream file(metrics_path.string(), std::ios::app);
    file << std::setprecision(16);
    file << record.step << ',' << record.time << ',' << record.dt << ',' << record.reynolds_number << ',' << record.nu << ',' << record.drag_coefficient << ',' << record.lift_coefficient << ',' << record.side_coefficient << ',' << record.delta_pressure << ',' << record.field_metrics.div_l2 << ',' << record.field_metrics.div_linf << ',' << record.field_metrics.velocity_l2 << ',' << record.field_metrics.velocity_linf << ',' << record.field_metrics.pressure_l2 << ',' << record.field_metrics.pressure_mean << ',' << record.field_metrics.kinetic_energy << ',' << record.solver_metrics.gmres_iterations << ',' << record.solver_metrics.gmres_final_residual << ',' << (record.solver_metrics.gmres_converged ? "true" : "false") << ',' << record.solver_metrics.assembly_time << ',' << record.solver_metrics.preconditioner_setup_time << ',' << record.solver_metrics.linear_solve_time << ',' << record.solver_metrics.total_step_time << '\n';
}

void BenchmarkRecorder::finalize(const double total_runtime) const
{
    if (!master_rank)
        return;

    const double div_l2_mean = summary_steps > 0 ? summary_div_l2_sum / static_cast<double>(summary_steps) : std::numeric_limits<double>::quiet_NaN();
    const double gmres_mean = summary_steps > 0 ? summary_gmres_sum / static_cast<double>(summary_steps) : std::numeric_limits<double>::quiet_NaN();
    const double mean_step_time = summary_steps > 0 ? summary_step_time_sum / static_cast<double>(summary_steps) : std::numeric_limits<double>::quiet_NaN();
    const bool stable = summary_stable && summary_steps > 0;

    const fs::path summary_path = fs::path(options.output_directory) / "summary.csv";
    std::ofstream file(summary_path.string(), std::ios::trunc);
    file << std::setprecision(16);
    file << "run_id,benchmark_id,dimension,mesh_name,dofs_velocity,dofs_pressure,dofs_total,num_cells,dt,nu,Re,preconditioner,nonlinear_method,mpi_ranks,Cd_max,Cl_max,Cl_abs_max,DeltaP_final,St,divL2_mean,divL2_max,gmres_mean,gmres_max,total_runtime,mean_step_time,stable,failed_step\n";
    file << run_id << ',' << options.benchmark_id << ',' << metadata.dimension << ',' << metadata.mesh_name << ',' << metadata.dofs_velocity << ',' << metadata.dofs_pressure << ',' << metadata.dofs_total << ',' << metadata.num_cells << ',' << metadata.dt << ',' << metadata.nu << ',' << summary_reynolds_number << ',' << metadata.preconditioner << ',' << metadata.nonlinear_method << ',' << metadata.mpi_ranks << ',' << summary_cd_max << ',' << summary_cl_max << ',' << summary_cl_abs_max << ',' << summary_delta_pressure_final << ',' << last_strouhal_number << ',' << div_l2_mean << ',' << summary_div_l2_max << ',' << gmres_mean << ',' << summary_gmres_max << ',' << total_runtime << ',' << mean_step_time << ',' << (stable ? "true" : "false") << ',' << summary_failed_step << '\n';
}

const std::string &BenchmarkRecorder::output_directory() const
{
    return options.output_directory;
}

std::string BenchmarkRecorder::effective_mesh_name(const std::string &configured_mesh_name, const std::string &mesh_file_name)
{
    if (!configured_mesh_name.empty() && configured_mesh_name != "unknown")
        return configured_mesh_name;

    const fs::path mesh_path(mesh_file_name);
    const std::string stem = mesh_path.stem().generic_string();
    return stem.empty() ? "unknown" : stem;
}

void BenchmarkRecorder::write_config_copy() const
{
    std::map<std::string, std::string> values = options.config_values;
    values["run_id"] = run_id;
    values["benchmark_id"] = options.benchmark_id;
    values["dimension"] = std::to_string(metadata.dimension);
    values["mesh_file"] = metadata.mesh_file;
    values["mesh_name"] = metadata.mesh_name;
    values["output_directory"] = metadata.output_directory;
    values["mpi_ranks"] = std::to_string(metadata.mpi_ranks);
    values["source_parameter_file"] = options.source_parameter_file;

    const fs::path json_path = fs::path(options.output_directory) / "config.json";
    std::ofstream json(json_path.string(), std::ios::trunc);
    json << "{\n";
    unsigned int written = 0;
    for (const auto &entry : values)
    {
        json << "  \"" << json_escape(entry.first) << "\": \"" << json_escape(entry.second) << "\"";
        ++written;
        json << (written < values.size() ? "," : "") << "\n";
    }
    json << "}\n";

    if (options.source_parameter_file.empty())
        return;

    std::ifstream input(options.source_parameter_file);
    if (!input)
        return;

    const fs::path prm_path = fs::path(options.output_directory) / "config.prm";
    std::ofstream prm(prm_path.string(), std::ios::trunc);
    prm << input.rdbuf();
}

double BenchmarkRecorder::update_strouhal_number(const BenchmarkStepRecord &record)
{
    lift_history.push_back({record.time, record.lift_coefficient});

    if (record.reference_velocity <= 0.0 || record.reference_length <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> peak_times;
    for (unsigned int i = 1; i + 1 < lift_history.size(); ++i)
    {
        const double c_prev = lift_history[i - 1].second;
        const double c = lift_history[i].second;
        const double c_next = lift_history[i + 1].second;
        if (c > c_prev && c >= c_next)
            peak_times.push_back(lift_history[i].first);
    }

    if (peak_times.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    double period_sum = 0.0;
    for (unsigned int i = 1; i < peak_times.size(); ++i)
        period_sum += peak_times[i] - peak_times[i - 1];

    const double mean_period = period_sum / static_cast<double>(peak_times.size() - 1);
    if (mean_period <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();

    return record.reference_length / (record.reference_velocity * mean_period);
}

void BenchmarkRecorder::update_summary_metrics(const BenchmarkStepRecord &record)
{
    if (summary_steps == 0)
    {
        summary_cd_max = record.drag_coefficient;
        summary_cl_max = record.lift_coefficient;
        summary_cl_abs_max = std::abs(record.lift_coefficient);
    }
    else
    {
        summary_cd_max = std::max(summary_cd_max, record.drag_coefficient);
        summary_cl_max = std::max(summary_cl_max, record.lift_coefficient);
        summary_cl_abs_max = std::max(summary_cl_abs_max, std::abs(record.lift_coefficient));
    }

    summary_reynolds_number = record.reynolds_number;
    summary_delta_pressure_final = record.delta_pressure;
    summary_div_l2_sum += record.field_metrics.div_l2;
    summary_div_l2_max = std::max(summary_div_l2_max, record.field_metrics.div_l2);
    summary_gmres_sum += record.solver_metrics.gmres_iterations;
    summary_gmres_max = std::max(summary_gmres_max, record.solver_metrics.gmres_iterations);
    summary_step_time_sum += record.solver_metrics.total_step_time;
    ++summary_steps;

    const bool step_stable = std::isfinite(record.drag_coefficient) && std::isfinite(record.lift_coefficient) && std::isfinite(record.delta_pressure) && std::isfinite(record.reynolds_number) && std::isfinite(record.field_metrics.div_l2) && std::isfinite(record.field_metrics.div_linf) && std::isfinite(record.field_metrics.velocity_l2) && std::isfinite(record.field_metrics.velocity_linf) && std::isfinite(record.field_metrics.pressure_l2) && std::isfinite(record.field_metrics.pressure_mean) && std::isfinite(record.field_metrics.kinetic_energy) && std::isfinite(record.solver_metrics.gmres_final_residual) && record.solver_metrics.linear_solves > 0 && record.solver_metrics.gmres_converged;
    summary_stable = summary_stable && step_stable;
    if (!step_stable && summary_failed_step < 0)
        summary_failed_step = static_cast<int>(record.step);
}

std::string BenchmarkRecorder::make_run_id(const std::string &configured_run_id)
{
    if (!configured_run_id.empty())
        return configured_run_id;

    const std::time_t now = std::time(nullptr);
    std::tm *time_info = std::localtime(&now);
    if (time_info == nullptr)
        return "run";

    char buffer[32];
    if (std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", time_info) == 0)
        return "run";

    return buffer;
}

std::string BenchmarkRecorder::json_escape(const std::string &value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char c : value)
    {
        switch (c)
        {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += c;
                break;
        }
    }
    return escaped;
}
