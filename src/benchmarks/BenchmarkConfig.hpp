#ifndef BENCHMARKCONFIG_HPP
#define BENCHMARKCONFIG_HPP

#include "NavierStokesOptions.hpp"

#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>

class BenchmarkConfig
{
    public:
        template <typename Problem, typename Parameters>
        static void apply_to(Problem &problem, const Parameters &parameters, const std::string &parameter_file)
        {
            problem.set_output_options(parameters.output_directory, parameters.run_id, parameters.benchmark_id, parameters.mesh_name);
            problem.set_write_solution_output(parameters.write_solution_output);
            problem.set_benchmark_statistics_start_time(parameters.statistics_start_time);
            problem.set_run_config_file(parameter_file);
            problem.set_run_config_values(values(parameters));
        }

        template <typename Parameters>
        static std::map<std::string, std::string> values(const Parameters &parameters)
        {
            const double reynolds_number = parameters.nu > 0.0 ? parameters.force_coefficient_reference_velocity * parameters.force_coefficient_reference_length / parameters.nu : std::numeric_limits<double>::quiet_NaN();
            std::map<std::string, std::string> config_values;
            config_values["run_id"] = parameters.run_id;
            config_values["benchmark_id"] = parameters.benchmark_id;
            config_values["write_solution_output"] = to_config_string(parameters.write_solution_output);
            config_values["statistics_start_time"] = to_config_string(parameters.statistics_start_time);
            config_values["dimension"] = to_config_string(Parameters::dimension);
            config_values["mesh_file"] = parameters.mesh_file_name;
            config_values["mesh_name"] = parameters.mesh_name;
            config_values["velocity_degree"] = to_config_string(parameters.degree_velocity);
            config_values["pressure_degree"] = to_config_string(parameters.degree_pressure);
            config_values["final_time"] = to_config_string(parameters.T);
            config_values["dt"] = to_config_string(parameters.delta_t);
            config_values["theta"] = to_config_string(parameters.theta);
            config_values["nu"] = to_config_string(parameters.nu);
            config_values["Re"] = to_config_string(reynolds_number);
            config_values["inlet_velocity"] = to_config_string(parameters.inlet_velocity);
            config_values["inlet_channel_height"] = to_config_string(parameters.inlet_channel_height);
            config_values["inlet_ramp_time"] = to_config_string(parameters.inlet_ramp_time);
            config_values["reference_velocity"] = to_config_string(parameters.force_coefficient_reference_velocity);
            config_values["reference_length"] = to_config_string(parameters.force_coefficient_reference_length);
            config_values["preconditioner"] = to_config_string(to_string(parameters.preconditioner));
            config_values["simple_pressure_relaxation"] = to_config_string(parameters.simple_pressure_relaxation);
            config_values["nonlinear_method"] = to_config_string(to_string(parameters.nonlinear_method));
            config_values["nonlinear_iterations"] = to_config_string(parameters.nonlinear_max_iterations);
            config_values["nonlinear_tolerance"] = to_config_string(parameters.nonlinear_tolerance);
            config_values["picard_relaxation"] = to_config_string(parameters.picard_relaxation);
            config_values["gmres_restart_length"] = to_config_string(parameters.gmres_restart_length);
            config_values["pressure_regularization"] = to_config_string(parameters.pressure_regularization);
            config_values["linear_max_iterations"] = to_config_string(parameters.linear_max_iterations);
            config_values["linear_relative_tolerance"] = to_config_string(parameters.linear_relative_tolerance);
            config_values["linear_absolute_tolerance"] = to_config_string(parameters.linear_absolute_tolerance);
            config_values["temam"] = to_config_string(parameters.stabilization.temam);
            config_values["grad_div"] = to_config_string(parameters.stabilization.grad_div);
            config_values["gamma_grad_div"] = to_config_string(parameters.stabilization.gamma_grad_div);
            config_values["supg"] = to_config_string(parameters.stabilization.supg);

            if constexpr (Parameters::dimension == 3)
            {
                config_values["inlet_channel_width"] = to_config_string(parameters.inlet_channel_width);
                config_values["reference_span"] = to_config_string(parameters.force_coefficient_reference_span);
            }

            return config_values;
        }

    private:
        template <typename T>
        static std::string to_config_string(const T &value)
        {
            std::ostringstream out;
            out << std::boolalpha << std::setprecision(16) << value;
            return out.str();
        }

        static std::string to_config_string(const std::string &value)
        {
            return value;
        }
};

#endif
