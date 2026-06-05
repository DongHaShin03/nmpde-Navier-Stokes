#include "FlowPastCylinder3D.hpp"

#include <algorithm>
#include <cmath>

void FlowPastCylinder3DParser::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Mesh and discretization");
    prm.declare_entry("Dimension", "3", Patterns::Integer(2, 3));
    prm.declare_entry("Mesh file",
                      "../mesh/navierstokes3D_L0.msh",
                      Patterns::Anything());
    prm.declare_entry("Velocity degree", "2", Patterns::Integer(1));
    prm.declare_entry("Pressure degree", "1", Patterns::Integer(1));
    prm.declare_entry("Final time", "7", Patterns::Double(0.0));
    prm.declare_entry("Theta", "0.6", Patterns::Double(0.0, 1.0));
    prm.declare_entry("Time step", "0.05", Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    prm.declare_entry("Nonlinear method",
                      "oseen",
                      Patterns::Selection(
                        "oseen|none|picard|picard_relaxed"));
    prm.declare_entry("Nonlinear iterations", "2", Patterns::Integer(1));
    prm.declare_entry("Nonlinear tolerance", "1e-6", Patterns::Double(0.0));
    prm.declare_entry("Picard relaxation", "1.0", Patterns::Double(0.0, 1.0));
    prm.declare_entry("GMRES restart length", "300", Patterns::Integer(1));
    prm.declare_entry("Pressure regularization", "0.0", Patterns::Double(0.0));
    prm.declare_entry("Linear max iterations", "10000", Patterns::Integer(1));
    prm.declare_entry("Linear relative tolerance", "1e-6", Patterns::Double(0.0));
    prm.declare_entry("Linear absolute tolerance", "1e-10", Patterns::Double(0.0));
    prm.declare_entry("Preconditioner",
                      "simple",
                      Patterns::Selection(
                        "simple|block_triangular|yosida|pcd"));
    prm.declare_entry("SIMPLE pressure relaxation",
                      "0.7",
                      Patterns::Double(0.0, 1.0));
    prm.declare_entry("Block triangular velocity max iterations",
                      "100",
                      Patterns::Integer(1));
    prm.declare_entry("Block triangular Schur max iterations",
                      "250",
                      Patterns::Integer(1));
    prm.declare_entry("Block triangular velocity relative tolerance",
                      "1e-2",
                      Patterns::Double(0.0));
    prm.declare_entry("Block triangular Schur relative tolerance",
                      "1e-3",
                      Patterns::Double(0.0));
    prm.declare_entry("SIMPLE velocity max iterations",
                      "5",
                      Patterns::Integer(1));
    prm.declare_entry("SIMPLE Schur max iterations",
                      "20",
                      Patterns::Integer(1));
    prm.declare_entry("SIMPLE velocity relative tolerance",
                      "1e-2",
                      Patterns::Double(0.0));
    prm.declare_entry("SIMPLE Schur relative tolerance",
                      "1e-3",
                      Patterns::Double(0.0));
    prm.declare_entry("PCD velocity max iterations",
                      "10",
                      Patterns::Integer(1));
    prm.declare_entry("PCD pressure max iterations",
                      "20",
                      Patterns::Integer(1));
    prm.declare_entry("PCD velocity relative tolerance",
                      "1e-2",
                      Patterns::Double(0.0));
    prm.declare_entry("PCD pressure relative tolerance",
                      "1e-3",
                      Patterns::Double(0.0));
    prm.declare_entry("Yosida velocity max iterations",
                      "100000",
                      Patterns::Integer(1));
    prm.declare_entry("Yosida Schur max iterations",
                      "100000",
                      Patterns::Integer(1));
    prm.declare_entry("Yosida correction max iterations",
                      "100000",
                      Patterns::Integer(1));
    prm.declare_entry("Yosida relative tolerance",
                      "1e-2",
                      Patterns::Double(0.0));
    prm.declare_entry("Preconditioner absolute tolerance",
                      "1e-12",
                      Patterns::Double(0.0));
    prm.declare_entry("Yosida absolute tolerance",
                      "1e-14",
                      Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Stabilization");
    prm.declare_entry("Temam", "true", Patterns::Bool());
    prm.declare_entry("Grad-div", "true", Patterns::Bool());
    prm.declare_entry("Grad-div coefficient", "0.01", Patterns::Double(0.0));
    prm.declare_entry("SUPG", "true", Patterns::Bool());
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    prm.declare_entry("Viscosity", "0.001", Patterns::Double(0.0));
    prm.declare_entry("Inlet velocity", "2.25", Patterns::Double(0.0));
    prm.declare_entry("Inlet channel height", "0.41", Patterns::Double(0.0));
    prm.declare_entry("Inlet channel width", "0.41", Patterns::Double(0.0));
    prm.declare_entry("Inlet ramp time", "8.0", Patterns::Double(0.0));
    prm.declare_entry("Outlet pressure", "0.0", Patterns::Double());
    prm.leave_subsection();

    prm.enter_subsection("Force coefficients");
    prm.declare_entry("Reference velocity", "1.0", Patterns::Double(0.0));
    prm.declare_entry("Reference length", "0.1", Patterns::Double(0.0));
    prm.declare_entry("Reference span", "0.41", Patterns::Double(0.0));
    prm.leave_subsection();

    prm.enter_subsection("Boundary ids");
    prm.declare_entry("Inlet", "1", Patterns::Integer(0));
    prm.declare_entry("Outlet", "2", Patterns::Integer(0));
    prm.declare_entry("Walls", "3", Patterns::Integer(0));
    prm.declare_entry("Cylinder", "5", Patterns::Integer(0));
    prm.leave_subsection();

    prm.enter_subsection("Output");
    prm.declare_entry("Output directory","benchmark_results/default_run",Patterns::Anything());
    prm.declare_entry("Run id", "default_run", Patterns::Anything());
    prm.declare_entry("Benchmark id", "unknown", Patterns::Anything());
    prm.declare_entry("Mesh name", "unknown", Patterns::Anything());
    prm.leave_subsection();
}

FlowPastCylinder3DConfig FlowPastCylinder3DParser::parse_parameters(
  ParameterHandler &prm)
{
    FlowPastCylinder3DConfig config;

    prm.enter_subsection("Mesh and discretization");
    config.mesh_file_name = prm.get("Mesh file");
    config.degree_velocity =
      static_cast<unsigned int>(prm.get_integer("Velocity degree"));
    config.degree_pressure =
      static_cast<unsigned int>(prm.get_integer("Pressure degree"));
    config.T = prm.get_double("Final time");
    config.theta = prm.get_double("Theta");
    config.delta_t = prm.get_double("Time step");
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    config.nonlinear_method = parse_nonlinear_method(prm.get("Nonlinear method"));
    config.nonlinear_max_iterations =
      static_cast<unsigned int>(prm.get_integer("Nonlinear iterations"));
    config.nonlinear_tolerance = prm.get_double("Nonlinear tolerance");
    config.picard_relaxation = prm.get_double("Picard relaxation");
    config.gmres_restart_length =
      static_cast<unsigned int>(prm.get_integer("GMRES restart length"));
    config.pressure_regularization = prm.get_double("Pressure regularization");
    config.linear_max_iterations =
      static_cast<unsigned int>(prm.get_integer("Linear max iterations"));
    config.linear_relative_tolerance =
      prm.get_double("Linear relative tolerance");
    config.linear_absolute_tolerance =
      prm.get_double("Linear absolute tolerance");
    config.preconditioner = parse_preconditioner_kind(prm.get("Preconditioner"));
    config.simple_pressure_relaxation =
      prm.get_double("SIMPLE pressure relaxation");
    config.preconditioner_iterations.block_triangular_velocity_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("Block triangular velocity max iterations"));
    config.preconditioner_iterations.block_triangular_schur_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("Block triangular Schur max iterations"));
    config.preconditioner_iterations.block_triangular_velocity_relative_tolerance =
      prm.get_double("Block triangular velocity relative tolerance");
    config.preconditioner_iterations.block_triangular_schur_relative_tolerance =
      prm.get_double("Block triangular Schur relative tolerance");
    config.preconditioner_iterations.simple_velocity_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("SIMPLE velocity max iterations"));
    config.preconditioner_iterations.simple_schur_max_iterations =
      static_cast<unsigned int>(prm.get_integer("SIMPLE Schur max iterations"));
    config.preconditioner_iterations.simple_velocity_relative_tolerance =
      prm.get_double("SIMPLE velocity relative tolerance");
    config.preconditioner_iterations.simple_schur_relative_tolerance =
      prm.get_double("SIMPLE Schur relative tolerance");
    config.preconditioner_iterations.pcd_velocity_max_iterations =
      static_cast<unsigned int>(prm.get_integer("PCD velocity max iterations"));
    config.preconditioner_iterations.pcd_pressure_max_iterations =
      static_cast<unsigned int>(prm.get_integer("PCD pressure max iterations"));
    config.preconditioner_iterations.pcd_velocity_relative_tolerance =
      prm.get_double("PCD velocity relative tolerance");
    config.preconditioner_iterations.pcd_pressure_relative_tolerance =
      prm.get_double("PCD pressure relative tolerance");
    config.preconditioner_iterations.yosida_velocity_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("Yosida velocity max iterations"));
    config.preconditioner_iterations.yosida_schur_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("Yosida Schur max iterations"));
    config.preconditioner_iterations.yosida_correction_max_iterations =
      static_cast<unsigned int>(
        prm.get_integer("Yosida correction max iterations"));
    config.preconditioner_iterations.yosida_relative_tolerance =
      prm.get_double("Yosida relative tolerance");
    config.preconditioner_iterations.preconditioner_absolute_tolerance =
      prm.get_double("Preconditioner absolute tolerance");
    config.preconditioner_iterations.yosida_absolute_tolerance =
      prm.get_double("Yosida absolute tolerance");
    prm.leave_subsection();

    prm.enter_subsection("Stabilization");
    config.stabilization.temam = prm.get_bool("Temam");
    config.stabilization.grad_div = prm.get_bool("Grad-div");
    config.stabilization.gamma_grad_div =
      prm.get_double("Grad-div coefficient");
    config.stabilization.supg = prm.get_bool("SUPG");
    prm.leave_subsection();

    prm.enter_subsection("Physics");
    config.nu = prm.get_double("Viscosity");
    config.inlet_velocity = prm.get_double("Inlet velocity");
    config.inlet_channel_height = prm.get_double("Inlet channel height");
    config.inlet_channel_width = prm.get_double("Inlet channel width");
    config.inlet_ramp_time = prm.get_double("Inlet ramp time");
    config.outlet_pressure = prm.get_double("Outlet pressure");
    prm.leave_subsection();

    prm.enter_subsection("Force coefficients");
    config.force_coefficient_reference_velocity =
      prm.get_double("Reference velocity");
    config.force_coefficient_reference_length =
      prm.get_double("Reference length");
    config.force_coefficient_reference_span =
      prm.get_double("Reference span");
    prm.leave_subsection();

    prm.enter_subsection("Boundary ids");
    config.inlet_boundary_id =
      static_cast<types::boundary_id>(prm.get_integer("Inlet"));
    config.outlet_boundary_id =
      static_cast<types::boundary_id>(prm.get_integer("Outlet"));
    config.walls_boundary_id =
      static_cast<types::boundary_id>(prm.get_integer("Walls"));
    config.cylinder_boundary_id =
      static_cast<types::boundary_id>(prm.get_integer("Cylinder"));
    prm.leave_subsection();

    prm.enter_subsection("Output");
    config.output_directory = prm.get("Output directory");
    config.run_id = prm.get("Run id");
    config.benchmark_id = prm.get("Benchmark id");
    config.mesh_name = prm.get("Mesh name");
    prm.leave_subsection();

    return config;
}

FlowPastCylinder3DConfig FlowPastCylinder3DParser::read(
  const std::string &parameter_file)
{
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input(parameter_file);

    return parse_parameters(prm);
}

FlowPastCylinder3DInlet::FlowPastCylinder3DInlet(
  const double inlet_velocity_,
  const double channel_height_,
  const double channel_width_,
  const double ramp_time_)
  : Function<3>(4)
  , inlet_velocity(inlet_velocity_)
  , channel_height(channel_height_)
  , channel_width(channel_width_)
  , ramp_time(ramp_time_)
{}

void FlowPastCylinder3DInlet::vector_value(const Point<3> &point,Vector<double> &values) const
{
    double ramp_factor = 1.0;
    if (ramp_time > 0.0)
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        ramp_factor = std::sin(pi * this->get_time() / ramp_time);
    }

    const double y = std::clamp(point[1], 0.0, channel_height);
    const double z = std::clamp(point[2], 0.0, channel_width);

    // 3D Schaefer-Turek inlet profile on a rectangular cross-section:
    // U_x(y,z) = 16 U_m y(H-y) z(W-z) / (H^2 W^2), U_y = U_z = 0.
    // For H=W and U_m=2.25 the cross-section mean velocity is 1.
    const double profile =
      16.0 * inlet_velocity *
      y * (channel_height - y) *
      z * (channel_width - z) /
      (channel_height * channel_height *
       channel_width * channel_width);

    values[0] = ramp_factor * profile;
    values[1] = 0.0;
    values[2] = 0.0;
    values[3] = 0.0;
}

double FlowPastCylinder3DInlet::speed() const
{
    return inlet_velocity;
}

FlowPastCylinder3DOutletPressure::FlowPastCylinder3DOutletPressure(
  const double outlet_pressure_)
  : outlet_pressure(outlet_pressure_)
{}

double FlowPastCylinder3DOutletPressure::value(const Point<3> &,
                                               const unsigned int) const
{
    return outlet_pressure;
}

FlowPastCylinder3DCase::FlowPastCylinder3DCase(
  const FlowPastCylinder3DConfig &parameters)
  : force_coefficient_reference_velocity(parameters.force_coefficient_reference_velocity)
  , force_coefficient_reference_area(parameters.force_coefficient_reference_length * parameters.force_coefficient_reference_span)
  , force_coefficient_reference_length(parameters.force_coefficient_reference_length)
  , inlet_boundary_id(parameters.inlet_boundary_id)
  , outlet_boundary_id(parameters.outlet_boundary_id)
  , walls_boundary_id(parameters.walls_boundary_id)
  , cylinder_boundary_id(parameters.cylinder_boundary_id)
  , inlet(parameters.inlet_velocity,
          parameters.inlet_channel_height,
          parameters.inlet_channel_width,
          parameters.inlet_ramp_time)
  , outlet(parameters.outlet_pressure)
  , zero_velocity(dim + 1)
{}

void FlowPastCylinder3DCase::apply_to(NavierStokes3D &problem)
{
    problem.dirichlet[inlet_boundary_id] = &inlet;
    problem.dirichlet[walls_boundary_id] = &zero_velocity;
    problem.dirichlet[cylinder_boundary_id] = &zero_velocity;
    problem.neumann[outlet_boundary_id] = &outlet;
    problem.set_force_coefficient_parameters(force_coefficient_reference_velocity,
                                             force_coefficient_reference_area,
                                             force_coefficient_reference_length,
                                             cylinder_boundary_id);
}
