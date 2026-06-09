#ifndef BENCHMARKRECORDER_HPP
#define BENCHMARKRECORDER_HPP

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct BenchmarkOutputOptions
{
    std::string output_directory = "benchmark_results/default_run";
    std::string run_id = "default_run";
    std::string benchmark_id = "unknown";
    std::string mesh_name = "unknown";
    std::string source_parameter_file;
    std::map<std::string, std::string> config_values;
    double statistics_start_time = 0.0;
};

struct BenchmarkRunMetadata
{
    unsigned int dimension = 0;
    std::string mesh_file;
    std::string mesh_name;
    std::string output_directory;
    unsigned int mpi_ranks = 1;
    std::uint64_t dofs_velocity = 0;
    std::uint64_t dofs_pressure = 0;
    std::uint64_t dofs_total = 0;
    std::uint64_t num_cells = 0;
    double dt = 0.0;
    double nu = 0.0;
    std::string preconditioner;
    std::string nonlinear_method;
};

struct BenchmarkFieldMetrics
{
    double div_l2 = 0.0;
    double div_linf = 0.0;
    double velocity_l2 = 0.0;
    double velocity_linf = 0.0;
    double pressure_l2 = 0.0;
    double pressure_mean = 0.0;
    double kinetic_energy = 0.0;
};

struct BenchmarkSolverStepMetrics
{
    unsigned int linear_solves = 0;
    unsigned int gmres_iterations = 0;
    unsigned int preconditioner_inner_solves = 0;
    unsigned int preconditioner_inner_iterations = 0;
    unsigned int preconditioner_inner_failures = 0;
    unsigned int nonlinear_iterations = 0;
    double gmres_final_residual = 0.0;
    bool gmres_converged = true;
    double assembly_time = 0.0;
    double preconditioner_setup_time = 0.0;
    double linear_solve_time = 0.0;
    double total_step_time = 0.0;
};

struct BenchmarkStepRecord
{
    unsigned int step = 0;
    double time = 0.0;
    double dt = 0.0;
    double nu = 0.0;
    double reynolds_number = 0.0;
    double drag_coefficient = 0.0;
    double lift_coefficient = 0.0;
    double side_coefficient = 0.0;
    double delta_pressure = 0.0;
    double reference_velocity = 0.0;
    double reference_length = 0.0;
    BenchmarkFieldMetrics field_metrics;
    BenchmarkSolverStepMetrics solver_metrics;
};

class BenchmarkRecorder
{
    public:
        void configure(const BenchmarkOutputOptions &options_);
        void initialize(const BenchmarkRunMetadata &metadata_, const bool master_rank_);
        void record_step(const BenchmarkStepRecord &record);
        void finalize(const double total_runtime) const;
        const std::string &output_directory() const;
        static std::string effective_mesh_name(const std::string &configured_mesh_name, const std::string &mesh_file_name);

    private:
        void write_config_copy() const;
        double update_strouhal_number(const BenchmarkStepRecord &record);
        void update_summary_metrics(const BenchmarkStepRecord &record);
        void update_stability_metrics(const BenchmarkStepRecord &record);
        static std::string make_run_id(const std::string &configured_run_id);
        static std::string json_escape(const std::string &value);

        BenchmarkOutputOptions options;
        BenchmarkRunMetadata metadata;
        bool master_rank = false;
        std::string run_id = "default_run";
        std::vector<std::pair<double, double>> lift_history;
        double last_strouhal_number = 0.0;
        double last_lift_period = 0.0;
        double last_lift_frequency = 0.0;
        unsigned int recorded_steps = 0;
        unsigned int summary_steps = 0;
        double summary_reynolds_number = 0.0;
        double summary_cd_final = 0.0;
        double summary_cl_final = 0.0;
        double summary_cs_final = 0.0;
        double summary_cd_max = 0.0;
        double summary_cl_max = 0.0;
        double summary_cl_abs_max = 0.0;
        double summary_cs_max = 0.0;
        double summary_delta_pressure_final = 0.0;
        double summary_div_l2_sum = 0.0;
        double summary_div_l2_max = 0.0;
        double summary_gmres_sum = 0.0;
        unsigned int summary_gmres_max = 0;
        unsigned int summary_gmres_failures = 0;
        double summary_preconditioner_inner_solves_sum = 0.0;
        double summary_preconditioner_inner_iterations_sum = 0.0;
        unsigned int summary_preconditioner_inner_failures = 0;
        double summary_assembly_time_sum = 0.0;
        double summary_preconditioner_setup_time_sum = 0.0;
        double summary_linear_solve_time_sum = 0.0;
        double summary_step_time_sum = 0.0;
        int summary_failed_step = -1;
        bool summary_stable = true;
};

#endif
