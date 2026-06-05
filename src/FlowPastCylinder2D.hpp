#ifndef FLOW_PAST_CYLINDER_2D_HPP
#define FLOW_PAST_CYLINDER_2D_HPP

#include <deal.II/base/parameter_handler.h>

#include "NavierStokes2D.hpp"

struct FlowPastCylinder2DConfig
{
    static constexpr unsigned int dimension = 2;

    std::string mesh_file_name = "../mesh/navierstokes_L0.msh";
    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;

    double nu = 0.0005;
    double T = 7.0;
    double theta = 0.6;
    double delta_t = 0.1;

    std::string output_directory = "benchmark_results/default_run";
    std::string run_id = "default_run";
    std::string benchmark_id = "unknown";
    std::string mesh_name = "unknown";

    NonlinearMethod nonlinear_method = NonlinearMethod::Oseen;
    unsigned int nonlinear_max_iterations = 2;
    double nonlinear_tolerance = 1e-6;
    double picard_relaxation = 1.0;

    unsigned int gmres_restart_length = 300;
    double pressure_regularization = 0.0;
    unsigned int linear_max_iterations = 10000;
    double linear_relative_tolerance = 1e-6;
    double linear_absolute_tolerance = 1e-10;
    PreconditionerKind preconditioner = PreconditionerKind::Simple;
    double simple_pressure_relaxation = 0.7;
    PreconditionerIterationOptions preconditioner_iterations;

    StabilizationOptions stabilization = {true, true, 0.01, true};

    double inlet_velocity = 1.5;
    double inlet_channel_height = 0.41;
    double inlet_ramp_time = 8.0;
    double outlet_pressure = 0.0;

    double force_coefficient_reference_velocity = 1.0;
    double force_coefficient_reference_length = 0.1;

    types::boundary_id inlet_boundary_id = 1;
    types::boundary_id outlet_boundary_id = 2;
    types::boundary_id walls_boundary_id = 3;
    types::boundary_id cylinder_boundary_id = 5;

};

class FlowPastCylinder2DParser
{
    public:
        static void declare_parameters(ParameterHandler &prm);
        static FlowPastCylinder2DConfig read(const std::string &parameter_file);

    private:
        static FlowPastCylinder2DConfig parse_parameters(ParameterHandler &prm);
};

class FlowPastCylinder2DInlet : public Function<2>
{
    public:
        explicit FlowPastCylinder2DInlet(const double inlet_velocity = 1.5,
                                         const double channel_height = 0.41,
                                         const double ramp_time = 0.0);

        void vector_value(const Point<2> &, Vector<double> &values) const override;
        double speed() const;

    private:
        const double inlet_velocity;
        const double channel_height;
        const double ramp_time;
};

class FlowPastCylinder2DOutletPressure : public Function<2>
{
    public:
        explicit FlowPastCylinder2DOutletPressure(const double outlet_pressure = 0.0);

        double value(const Point<2> &, const unsigned int component = 0) const override;

    private:
        const double outlet_pressure;
};

class FlowPastCylinder2DCase
{
    public:
        static constexpr unsigned int dim = NavierStokes2D::dim;

        explicit FlowPastCylinder2DCase(const FlowPastCylinder2DConfig &parameters);

        void apply_to(NavierStokes2D &problem);

    private:
        const double force_coefficient_reference_velocity;
        const double force_coefficient_reference_length;
        const types::boundary_id inlet_boundary_id;
        const types::boundary_id outlet_boundary_id;
        const types::boundary_id walls_boundary_id;
        const types::boundary_id cylinder_boundary_id;

        FlowPastCylinder2DInlet inlet;
        FlowPastCylinder2DOutletPressure outlet;
        Functions::ZeroFunction<dim> zero_velocity;
};

#endif
