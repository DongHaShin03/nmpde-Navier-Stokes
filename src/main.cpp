#include "FlowPastCylinder2D.hpp"
#include "FlowPastCylinder3D.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace
{
    std::string trim(const std::string &text)
    {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";

        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    unsigned int read_problem_dimension(const std::string &parameter_file)
    {
        std::ifstream input(parameter_file);
        std::string line;

        while (std::getline(input, line))
        {
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line.erase(comment);

            const auto set_position = line.find("set");
            const auto dimension_position = line.find("Dimension");
            const auto equals_position = line.find('=');

            if (set_position == std::string::npos ||
                dimension_position == std::string::npos ||
                equals_position == std::string::npos)
                continue;

            const std::string value = trim(line.substr(equals_position + 1));
            if (value == "2")
                return 2;
            if (value == "3")
                return 3;

            throw std::runtime_error("Unsupported problem dimension: " + value);
        }

        return 2;
    }

    template <typename Problem, typename Parser, typename Case>
    void run_flow_past_cylinder_case(const std::string &parameter_file)
    {
        const auto parameters = Parser::read(parameter_file);

        constexpr unsigned int dim = Problem::dim;
        const auto f = [](const Point<dim> &, const double &)
        {
            Tensor<1, dim> result;
            for (unsigned int d = 0; d < dim; ++d)
                result[d] = 0.0;
            return result;
        };

        Problem problem(parameters.mesh_file_name,
                        parameters.degree_velocity,
                        parameters.degree_pressure,
                        parameters.nu,
                        f,
                        parameters.T,
                        parameters.theta,
                        parameters.delta_t);
        problem.set_nonlinear_solver_parameters(parameters.nonlinear_max_iterations,
                                                parameters.nonlinear_tolerance);
        problem.set_nonlinear_solver_strategy(parameters.nonlinear_method,
                                              parameters.picard_relaxation);
        problem.set_linear_solver_parameters(parameters.gmres_restart_length,
                                             parameters.pressure_regularization,
                                             parameters.linear_max_iterations,
                                             parameters.linear_relative_tolerance,
                                             parameters.linear_absolute_tolerance);
        problem.set_preconditioner(parameters.preconditioner);
        problem.set_simple_pressure_relaxation(
          parameters.simple_pressure_relaxation);
        problem.set_stabilization_options(parameters.stabilization);

        Case benchmark_case(parameters);
        benchmark_case.apply_to(problem);

        problem.run();
    }
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string parameter_file =
      (argc > 1 ? std::string(argv[1]) : "../flow_past_cylinder_2d.prm");

    const unsigned int problem_dimension = read_problem_dimension(parameter_file);
    if (problem_dimension == 2)
    {
        run_flow_past_cylinder_case<NavierStokes2D,
                                    FlowPastCylinder2DParser,
                                    FlowPastCylinder2DCase>(parameter_file);
    }
    else if (problem_dimension == 3)
    {
        run_flow_past_cylinder_case<NavierStokes3D,
                                    FlowPastCylinder3DParser,
                                    FlowPastCylinder3DCase>(parameter_file);
    }
    else
        throw std::runtime_error("Unsupported problem dimension.");

    return 0;
}
