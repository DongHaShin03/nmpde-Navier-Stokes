#include "FlowPastCylinder2D.hpp"

void FlowPastCylinder2DParameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Mesh and discretization");
    prm.declare_entry("Mesh file",
                      "../mesh/ns-mesh2D-level0.msh",
                      Patterns::Anything());
    prm.declare_entry("Velocity degree", "2", Patterns::Integer(1));
    prm.declare_entry("Pressure degree", "1", Patterns::Integer(1));
    prm.declare_entry("Final time", "1.0", Patterns::Double(0.0));
    prm.declare_entry("Theta", "1.0", Patterns::Double(0.0, 1.0));
    prm.declare_entry("Time step", "0.01", Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    prm.declare_entry("Viscosity", "0.05", Patterns::Double(0.0));
    prm.declare_entry("Inlet velocity", "1.0", Patterns::Double(0.0));
    prm.declare_entry("Outlet pressure", "-1.0", Patterns::Double());
    prm.leave_subsection();

    prm.enter_subsection("Force coefficients");
    prm.declare_entry("Reference velocity", "0.1", Patterns::Double(0.0));
    prm.declare_entry("Reference length", "25.0", Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Boundary ids");
    prm.declare_entry("Inlet", "1", Patterns::Integer(0));
    prm.declare_entry("Outlet", "2", Patterns::Integer(0));
    prm.declare_entry("Walls", "3", Patterns::Integer(0));
    prm.declare_entry("Cylinder", "5", Patterns::Integer(0));
    prm.leave_subsection();
}

void FlowPastCylinder2DParameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Mesh and discretization");
    mesh_file_name = prm.get("Mesh file");
    degree_velocity = static_cast<unsigned int>(prm.get_integer("Velocity degree"));
    degree_pressure = static_cast<unsigned int>(prm.get_integer("Pressure degree"));
    T = prm.get_double("Final time");
    theta = prm.get_double("Theta");
    delta_t = prm.get_double("Time step");
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    nu = prm.get_double("Viscosity");
    inlet_velocity = prm.get_double("Inlet velocity");
    outlet_pressure = prm.get_double("Outlet pressure");
    prm.leave_subsection();

    prm.enter_subsection("Force coefficients");
    force_coefficient_reference_velocity = prm.get_double("Reference velocity");
    force_coefficient_reference_length = prm.get_double("Reference length");
    prm.leave_subsection();

    prm.enter_subsection("Boundary ids");
    inlet_boundary_id = static_cast<types::boundary_id>(prm.get_integer("Inlet"));
    outlet_boundary_id = static_cast<types::boundary_id>(prm.get_integer("Outlet"));
    walls_boundary_id = static_cast<types::boundary_id>(prm.get_integer("Walls"));
    cylinder_boundary_id = static_cast<types::boundary_id>(prm.get_integer("Cylinder"));
    prm.leave_subsection();
}

FlowPastCylinder2DParameters FlowPastCylinder2DParameters::read(
  const std::string &parameter_file)
{
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(parameter_file);

    FlowPastCylinder2DParameters parameters;
    parameters.parse_parameters(prm);
    return parameters;
}

FlowPastCylinder2DInlet::FlowPastCylinder2DInlet(const double inlet_velocity_)
  : Function<2>(3)
  , inlet_velocity(inlet_velocity_)
{}

void FlowPastCylinder2DInlet::vector_value(const Point<2> &,
                                           Vector<double> &values) const
{
    values[0] = inlet_velocity;
    values[1] = 0.0;
    values[2] = 0.0;
}

double FlowPastCylinder2DInlet::speed() const
{
    return inlet_velocity;
}

FlowPastCylinder2DOutletPressure::FlowPastCylinder2DOutletPressure(
  const double outlet_pressure_)
  : outlet_pressure(outlet_pressure_)
{}

double FlowPastCylinder2DOutletPressure::value(const Point<2> &,
                                               const unsigned int) const
{
    return outlet_pressure;
}

FlowPastCylinder2DCase::FlowPastCylinder2DCase(
  const FlowPastCylinder2DParameters &parameters)
  : force_coefficient_reference_velocity(
      parameters.force_coefficient_reference_velocity)
  , force_coefficient_reference_length(parameters.force_coefficient_reference_length)
  , inlet_boundary_id(parameters.inlet_boundary_id)
  , outlet_boundary_id(parameters.outlet_boundary_id)
  , walls_boundary_id(parameters.walls_boundary_id)
  , cylinder_boundary_id(parameters.cylinder_boundary_id)
  , inlet(parameters.inlet_velocity)
  , outlet(parameters.outlet_pressure)
  , zero_velocity(dim + 1)
{}

void FlowPastCylinder2DCase::apply_to(NavierStokes2D &problem)
{
    problem.dirichlet[inlet_boundary_id] = &inlet;
    problem.dirichlet[walls_boundary_id] = &zero_velocity;
    problem.dirichlet[cylinder_boundary_id] = &zero_velocity;
    problem.neumann[outlet_boundary_id] = &outlet;
    problem.initial_condition =
      std::make_unique<FlowPastCylinder2DInlet>(inlet.speed());
    problem.set_force_coefficient_parameters(force_coefficient_reference_velocity,
                                             force_coefficient_reference_length,
                                             cylinder_boundary_id);
}
