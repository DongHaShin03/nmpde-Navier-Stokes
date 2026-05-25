#include "FlowPastCylinder2D.hpp"

#include <algorithm>
#include <cmath>

void FlowPastCylinder2DParameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Mesh and discretization");
    prm.declare_entry("Mesh file",
                      "../mesh/ns-mesh2D-level1.msh",
                      Patterns::Anything());
    prm.declare_entry("Velocity degree", "2", Patterns::Integer(1));
    prm.declare_entry("Pressure degree", "1", Patterns::Integer(1));
    prm.declare_entry("Final time", "0.05", Patterns::Double(0.0));
    prm.declare_entry("Theta", "1.0", Patterns::Double(0.0, 1.0));
    prm.declare_entry("Time step", "0.0025", Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    prm.declare_entry("Nonlinear method",
                      "none",
                      Patterns::Selection(
                        "none|picard|picard_relaxed|newton|newton_damped"));
    prm.declare_entry("Nonlinear iterations", "8", Patterns::Integer(1));
    prm.declare_entry("Nonlinear tolerance", "1e-6", Patterns::Double(0.0));
    prm.declare_entry("Picard relaxation", "1.0", Patterns::Double(0.0, 1.0));
    prm.declare_entry("GMRES restart length", "800", Patterns::Integer(1));
    prm.declare_entry("Pressure regularization", "0.0", Patterns::Double(0.0));
    prm.declare_entry("Linear max iterations", "100000", Patterns::Integer(1));
    prm.declare_entry("Linear relative tolerance", "5e-2", Patterns::Double(0.0));
    prm.declare_entry("Linear absolute tolerance", "2e-2", Patterns::Double(0.0));
    prm.declare_entry("Preconditioner",
                      "yosida",
                      Patterns::Selection(
                        "identity|simple|block_diagonal|block_triangular|yosida|pcd"));
    prm.leave_subsection();

    prm.enter_subsection("Stabilization");
    prm.declare_entry("Temam", "true", Patterns::Bool());
    prm.declare_entry("Grad-div", "false", Patterns::Bool());
    prm.declare_entry("Grad-div coefficient", "0.0", Patterns::Double(0.0));
    prm.declare_entry("SUPG", "false", Patterns::Bool());
    prm.declare_entry("PSPG", "false", Patterns::Bool());
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    prm.declare_entry("Viscosity", "0.5", Patterns::Double(0.0));
    prm.declare_entry("Inlet velocity", "0.05", Patterns::Double(0.0));
    prm.declare_entry("Inlet channel height", "4.1", Patterns::Double(0.0));
    prm.declare_entry("Inlet ramp time", "0.0", Patterns::Double(0.0));
    prm.declare_entry("Outlet pressure", "0.0", Patterns::Double());
    prm.leave_subsection();

    prm.enter_subsection("Force coefficients");
    prm.declare_entry("Reference velocity", "0.05", Patterns::Double(0.0));
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

    prm.enter_subsection("Solver");
    nonlinear_method = parse_nonlinear_method(prm.get("Nonlinear method"));
    nonlinear_max_iterations =
      static_cast<unsigned int>(prm.get_integer("Nonlinear iterations"));
    nonlinear_tolerance = prm.get_double("Nonlinear tolerance");
    picard_relaxation = prm.get_double("Picard relaxation");
    gmres_restart_length =
      static_cast<unsigned int>(prm.get_integer("GMRES restart length"));
    pressure_regularization = prm.get_double("Pressure regularization");
    linear_max_iterations =
      static_cast<unsigned int>(prm.get_integer("Linear max iterations"));
    linear_relative_tolerance = prm.get_double("Linear relative tolerance");
    linear_absolute_tolerance = prm.get_double("Linear absolute tolerance");
    preconditioner = parse_preconditioner_kind(prm.get("Preconditioner"));
    prm.leave_subsection();

    prm.enter_subsection("Stabilization");
    stabilization.temam = prm.get_bool("Temam");
    stabilization.grad_div = prm.get_bool("Grad-div");
    stabilization.gamma_grad_div = prm.get_double("Grad-div coefficient");
    stabilization.supg = prm.get_bool("SUPG");
    stabilization.pspg = prm.get_bool("PSPG");
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    nu = prm.get_double("Viscosity");
    inlet_velocity = prm.get_double("Inlet velocity");
    inlet_channel_height = prm.get_double("Inlet channel height");
    inlet_ramp_time = prm.get_double("Inlet ramp time");
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

FlowPastCylinder2DInlet::FlowPastCylinder2DInlet(const double inlet_velocity_,
                                                 const double channel_height_,
                                                 const double ramp_time_)
  : Function<2>(3)
  , inlet_velocity(inlet_velocity_)
  , channel_height(channel_height_)
  , ramp_time(ramp_time_)
{}

// Inlet velocity profile
void FlowPastCylinder2DInlet::vector_value(const Point<2> &point,
                                           Vector<double> &values) const
{
    double ramp_factor = 1.0;
    if (ramp_time > 0.0)
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        ramp_factor = std::sin(pi * this->get_time() / ramp_time);
    }

    const double y = std::clamp(point[1], 0.0, channel_height);
    const double profile =
      4.0 * inlet_velocity * y * (channel_height - y) /
      (channel_height * channel_height);

    values[0] = ramp_factor * profile;
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
  , inlet(parameters.inlet_velocity,
          parameters.inlet_channel_height,
          parameters.inlet_ramp_time)
  , outlet(parameters.outlet_pressure)
  , zero_velocity(dim + 1)
{}

void FlowPastCylinder2DCase::apply_to(NavierStokes2D &problem)
{
    problem.dirichlet[inlet_boundary_id] = &inlet;
    problem.dirichlet[walls_boundary_id] = &zero_velocity;
    problem.dirichlet[cylinder_boundary_id] = &zero_velocity;
    problem.neumann[outlet_boundary_id] = &outlet;
    problem.set_force_coefficient_parameters(force_coefficient_reference_velocity,
                                             force_coefficient_reference_length,
                                             cylinder_boundary_id);
}
