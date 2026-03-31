#ifndef FLOW_PAST_CYLINDER_2D_HPP
#define FLOW_PAST_CYLINDER_2D_HPP

#include <deal.II/base/parameter_handler.h>

#include "NavierStokes2D.hpp"

struct FlowPastCylinder2DParameters
{
    std::string mesh_file_name = "../mesh/ns-mesh2D-level0.msh";
    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;

    double nu = 0.05;
    double T = 1.0;
    double theta = 1.0;
    double delta_t = 0.01;

    double inlet_velocity = 1.0;
    double outlet_pressure = -1.0;

    double force_coefficient_reference_velocity = 0.1;
    double force_coefficient_reference_length = 25.0;

    types::boundary_id inlet_boundary_id = 1;
    types::boundary_id outlet_boundary_id = 2;
    types::boundary_id walls_boundary_id = 3;
    types::boundary_id cylinder_boundary_id = 5;

    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);
    static FlowPastCylinder2DParameters read(const std::string &parameter_file);
};

class FlowPastCylinder2DInlet : public Function<2>
{
    public:
        explicit FlowPastCylinder2DInlet(const double inlet_velocity = 1.0);

        void vector_value(const Point<2> &, Vector<double> &values) const override;
        double speed() const;

    private:
        const double inlet_velocity;
};

class FlowPastCylinder2DOutletPressure : public Function<2>
{
    public:
        explicit FlowPastCylinder2DOutletPressure(const double outlet_pressure = -1.0);

        double value(const Point<2> &, const unsigned int component = 0) const override;

    private:
        const double outlet_pressure;
};

class FlowPastCylinder2DCase
{
    public:
        static constexpr unsigned int dim = NavierStokes2D::dim;

        explicit FlowPastCylinder2DCase(const FlowPastCylinder2DParameters &parameters);

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
