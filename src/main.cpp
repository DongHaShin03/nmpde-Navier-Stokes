#include "FlowPastCylinder2D.hpp"

static constexpr unsigned int dim = NavierStokes2D::dim;

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string parameter_file =
      (argc > 1 ? std::string(argv[1]) : "../flow_past_cylinder_2d.prm");

    const FlowPastCylinder2DParameters parameters =
      FlowPastCylinder2DParameters::read(parameter_file);

    const auto f = [](const Point<dim> &, const double &)
    {
        Tensor<1, dim> result;
        result[0] = 0.0;
        result[1] = 0.0;
        return result;
    };

    NavierStokes2D problem(parameters.mesh_file_name,
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
    problem.set_stabilization_options(parameters.stabilization);

    FlowPastCylinder2DCase benchmark_case(parameters);
    benchmark_case.apply_to(problem);

    problem.run();

    return 0;
}
