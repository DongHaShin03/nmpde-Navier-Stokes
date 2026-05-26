#ifndef FLOW_PAST_CYLINDER_3D_HPP
#define FLOW_PAST_CYLINDER_3D_HPP

#include <deal.II/base/parameter_handler.h>

#include "NavierStokes3D.hpp"

struct FlowPastCylinder3DConfig
{
    std::string mesh_file_name = "../mesh/navierstokes3D_L0.msh";
    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;

    double nu = 0.001;
    double T = 7.0;
    double theta = 0.6;
    double delta_t = 0.05;

    NonlinearMethod nonlinear_method = NonlinearMethod::None;
    unsigned int nonlinear_max_iterations = 2;
    double nonlinear_tolerance = 1e-6;
    double picard_relaxation = 1.0;

    unsigned int gmres_restart_length = 300;
    double pressure_regularization = 0.0;
    unsigned int linear_max_iterations = 10000;
    double linear_relative_tolerance = 1e-6;
    double linear_absolute_tolerance = 1e-10;
    PreconditionerKind preconditioner = PreconditionerKind::PCD;

    StabilizationOptions stabilization = {true, true, 0.01, true, false};

    double inlet_velocity = 2.25;
    double inlet_channel_height = 0.41;
    double inlet_channel_width = 0.41;
    double inlet_ramp_time = 8.0;
    double outlet_pressure = 0.0;

    double force_coefficient_reference_velocity = 1.0;
    double force_coefficient_reference_length = 0.1;
    double force_coefficient_reference_span = 0.41;

    types::boundary_id inlet_boundary_id = 1;
    types::boundary_id outlet_boundary_id = 2;
    types::boundary_id walls_boundary_id = 3;
    types::boundary_id cylinder_boundary_id = 5;
};

class FlowPastCylinder3DParser
{
    public:
        static void declare_parameters(ParameterHandler &prm);
        static FlowPastCylinder3DConfig read(const std::string &parameter_file);

    private:
        static FlowPastCylinder3DConfig parse_parameters(ParameterHandler &prm);
};

class FlowPastCylinder3DInlet : public Function<3>
{
    public:
        explicit FlowPastCylinder3DInlet(const double inlet_velocity = 2.25,
                                         const double channel_height = 0.41,
                                         const double channel_width = 0.41,
                                         const double ramp_time = 0.0);

        void vector_value(const Point<3> &, Vector<double> &values) const override;
        double speed() const;

    private:
        const double inlet_velocity;
        const double channel_height;
        const double channel_width;
        const double ramp_time;
};

class FlowPastCylinder3DOutletPressure : public Function<3>
{
    public:
        explicit FlowPastCylinder3DOutletPressure(const double outlet_pressure = 0.0);

        double value(const Point<3> &, const unsigned int component = 0) const override;

    private:
        const double outlet_pressure;
};

class FlowPastCylinder3DCase
{
    public:
        static constexpr unsigned int dim = NavierStokes3D::dim;

        explicit FlowPastCylinder3DCase(const FlowPastCylinder3DConfig &parameters);

        void apply_to(NavierStokes3D &problem);

    private:
        const double force_coefficient_reference_velocity;
        const double force_coefficient_reference_area;
        const types::boundary_id inlet_boundary_id;
        const types::boundary_id outlet_boundary_id;
        const types::boundary_id walls_boundary_id;
        const types::boundary_id cylinder_boundary_id;

        FlowPastCylinder3DInlet inlet;
        FlowPastCylinder3DOutletPressure outlet;
        Functions::ZeroFunction<dim> zero_velocity;
};

#endif
