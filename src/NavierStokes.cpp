#include "NavierStokes.hpp"
#include "preconditioners/PreconditionerFactory.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

//CONSTRUCTOR
template <int dim>
NavierStokes<dim>::NavierStokes(const std::string  &mesh_file_name_,
                                const unsigned int &degree_velocity_,
                                const unsigned int &degree_pressure_,
                                const double       &nu_,
                                const std::function<Tensor<1, dim>(const Point<dim> &, const double &)> &f_,
                                const double       &T_,
                                const double       &theta_,
                                const double       &delta_t_)
  : mesh_file_name(mesh_file_name_)
  , degree_velocity(degree_velocity_)
  , degree_pressure(degree_pressure_)
  , nu(nu_)
  , f(f_)
  , T(T_)
  , delta_t(delta_t_)
  , theta(theta_)
  , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  , mesh(MPI_COMM_WORLD)
  , dof_handler(mesh)
  , pcout(std::cout, mpi_rank == 0)
{}

template <int dim>
void NavierStokes<dim>::set_nonlinear_solver_parameters(
  const unsigned int max_iterations,
  const double       tolerance)
{
    nonlinear_max_iterations = std::max(1u, max_iterations);
    nonlinear_tolerance = std::max(1e-12, tolerance);
}

template <int dim>
void NavierStokes<dim>::set_nonlinear_solver_strategy(
  const NonlinearMethod method,
  const double          relaxation)
{
    nonlinear_method = method;
    picard_relaxation = std::min(1.0, std::max(0.0, relaxation));
}

template <int dim>
void NavierStokes<dim>::set_linear_solver_parameters(
  const unsigned int gmres_restart_length_,
  const double       pressure_regularization_,
  const unsigned int linear_max_iterations_,
  const double       linear_relative_tolerance_,
  const double       linear_absolute_tolerance_)
{
    gmres_restart_length = std::max(1u, gmres_restart_length_);
    pressure_regularization = std::max(0.0, pressure_regularization_);
    linear_max_iterations = std::max(1u, linear_max_iterations_);
    linear_relative_tolerance = std::max(0.0, linear_relative_tolerance_);
    linear_absolute_tolerance = std::max(0.0, linear_absolute_tolerance_);
}

template <int dim>
void NavierStokes<dim>::set_preconditioner(
  const PreconditionerKind preconditioner_kind_)
{
    preconditioner_kind = preconditioner_kind_;
}

template <int dim>
void NavierStokes<dim>::set_stabilization_options(
  const StabilizationOptions &options)
{
    stabilization_options = options;
}

template <int dim>
double NavierStokes<dim>::compute_supg_tau(const double beta_norm,
                                           const double h_K) const
{

    // tau_K = ( (2/dt)^2 + (2|beta|/h_K)^2 + (4nu/h_K^2)^2 )^{-1/2}.
    const double transient_scale = 2.0 / delta_t;
    const double convective_scale = 2.0 * beta_norm / h_K;
    const double diffusive_scale = 4.0 * nu / (h_K * h_K);

    const double denominator =
      std::sqrt(transient_scale * transient_scale +
                convective_scale * convective_scale +
                diffusive_scale * diffusive_scale);

    return 1.0 / denominator;
}

template <int dim>
bool NavierStokes<dim>::needs_velocity_mass_matrix() const
{
    return preconditioner_kind == PreconditionerKind::Yosida ||
           preconditioner_kind == PreconditionerKind::PCD;
}

template <int dim>
bool NavierStokes<dim>::needs_pressure_mass_matrix() const
{
    return preconditioner_kind == PreconditionerKind::Simple ||
           preconditioner_kind == PreconditionerKind::BlockTriangular ||
           preconditioner_kind == PreconditionerKind::PCD;
}

template <int dim>
bool NavierStokes<dim>::needs_pcd_pressure_operators() const
{
    return preconditioner_kind == PreconditionerKind::PCD;
}

template <int dim>
void NavierStokes<dim>::setup()
{
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        // Read serially and then partitioned to all MPI processes
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream mesh_file(mesh_file_name);
        grid_in.read_msh(mesh_file);

        // Partition of the mesh depending on the MPI ranks
        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data =
          TriangulationDescription::Utilities::create_description_from_triangulation(
            mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim,
                                             fe_scalar_pressure, 1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
              << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
              << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
        quadrature_boundary =
          std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
        pcout << "  Quadrature points per face = "
              << quadrature_boundary->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.distribute_dofs(*fe);

        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs =
          DoFTools::extract_locally_relevant_dofs(dof_handler);

        const std::vector<types::global_dof_index> dofs_per_block =
          DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

        block_owned_dofs.resize(2);
        block_relevant_dofs.resize(2);
        block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
        block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
        block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
        block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

        pcout << "  Number of DoFs: " << std::endl;
        pcout << "    Velocity = " << n_u << std::endl;
        pcout << "    Pressure = " << n_p << std::endl;
        pcout << "    Total    = " << n_u + n_p << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;

        // In this case is a 3x3 matrix, which indicates which components can be coupled in the matrix (like a mask)
        // Should be p-p = 0
        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim)
                    coupling[c][d] = DoFTools::none;
                else
                    coupling[c][d] = DoFTools::always;
            }
        }

        // We create a block sparsity pattern
        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);

        // See the mesh, the FE and coupling table
        // For every cell, if they are in the same cell, and their components are in the coupling table,
        // it puts a possible entry in the patterm
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

        //PRESSURE MASS MATRIX COUPLING
        Table<2, DoFTools::Coupling> coupling_pm(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim)
                    coupling_pm[c][d] = DoFTools::always;
                else
                    coupling_pm[c][d] = DoFTools::none;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
          block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        coupling_pm,
                                        sparsity_pressure_mass);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        // Initialize the matrix with sparsity
        // This is usefull to construct matrices with the same sparsity constructed with the
        // coupling matrices
        static_matrix.reinit(sparsity);
        convection_matrix.reinit(sparsity);
        system_matrix.reinit(sparsity);
        if (needs_velocity_mass_matrix())
            velocity_mass.reinit(sparsity);
        if (needs_pressure_mass_matrix())
            pressure_mass.reinit(sparsity_pressure_mass);
        if (needs_pcd_pressure_operators())
        {
            pressure_laplacian_discrete.reinit(sparsity_pressure_mass);
            pressure_convection_diffusion.reinit(sparsity_pressure_mass);
        }

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
        old_solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
        //linearization_point.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

// Assemble the time-independent part of the monolithic matrix:
// [ (1/dt)M_u + theta*nu*A_u + gamma*G   -B^T ]
// [ B                                           0 ]
// and, only when requested, the auxiliary pressure/velocity operators used by
// the preconditioners.
template <int dim>
void NavierStokes<dim>::assemble_static()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling static matrices (mass + stiffness + pressure blocks)" << std::endl;

    const bool assemble_velocity_mass = needs_velocity_mass_matrix();
    const bool assemble_pressure_mass = needs_pressure_mass_matrix();

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    
    FullMatrix<double> cell_static(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_velocity_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    static_matrix = 0.0;
    if (assemble_velocity_mass)
        velocity_mass = 0.0;
    if (assemble_pressure_mass)
        pressure_mass = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_static = 0.0;
        if (assemble_velocity_mass)
            cell_velocity_mass = 0.0;
        if (assemble_pressure_mass)
            cell_pressure_mass = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> phi_vel_i = fe_values[velocity].value(i, q);
                const Tensor<2, dim> grad_phi_vel_i = fe_values[velocity].gradient(i, q);
                const double div_phi_vel_i = fe_values[velocity].divergence(i, q);
                const double psi_i = fe_values[pressure].value(i, q);

                // --- LHS contributions ---
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> phi_vel_j = fe_values[velocity].value(j, q);
                    const Tensor<2, dim> grad_phi_vel_j = fe_values[velocity].gradient(j, q);
                    const double div_phi_vel_j = fe_values[velocity].divergence(j, q);
                    const double psi_j = fe_values[pressure].value(j, q);

                    // Velocity mass in the monolithic F block:
                    // (1/dt) M_u, with (M_u)_{ij} = (phi_j, phi_i)
                    cell_static(i, j) +=
                      (1.0 / delta_t) * (phi_vel_j * phi_vel_i) *
                      fe_values.JxW(q);

                    // Pure velocity mass used by Yosida and PCD:
                    // (M_u)_{ij} = (phi_j, phi_i)
                    if (assemble_velocity_mass)
                        cell_velocity_mass(i, j) +=
                          (phi_vel_j * phi_vel_i) * fe_values.JxW(q);

                    // Velocity diffusion in the monolithic F block:
                    // theta * nu * A_u, with
                    // (A_u)_{ij} = (grad(phi_j), grad(phi_i))
                    cell_static(i, j) += theta * nu * scalar_product(grad_phi_vel_j, grad_phi_vel_i) *
                      fe_values.JxW(q);

                    // Pressure-gradient block -B^T:
                    // -(p_j, div(v_i))
                    cell_static(i, j) -= psi_j * div_phi_vel_i * fe_values.JxW(q);

                    // Divergence constraint block B:
                    // (q_i, div(u_j))
                    cell_static(i, j) += psi_i * div_phi_vel_j * fe_values.JxW(q);

                    // Grad-div stabilization on the velocity block:
                    // gamma * (div(u_j), div(v_i))
                    if (stabilization_options.grad_div &&
                        stabilization_options.gamma_grad_div > 0.0)
                        cell_static(i, j) +=
                          stabilization_options.gamma_grad_div *
                          div_phi_vel_j * div_phi_vel_i *
                          fe_values.JxW(q);

                    // Pressure mass:
                    // (M_p)_{ij} = (psi_j, psi_i)
                    // Used by pressure-mass Schur approximations and by PCD
                    if (assemble_pressure_mass)
                        cell_pressure_mass(i, j) +=
                          psi_i * psi_j * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);
        static_matrix.add(dof_indices, cell_static);
        if (assemble_velocity_mass)
            velocity_mass.add(dof_indices, cell_velocity_mass);
        if (assemble_pressure_mass)
            pressure_mass.add(dof_indices, cell_pressure_mass);

    }

    static_matrix.compress(VectorOperation::add);
    if (assemble_velocity_mass)
        velocity_mass.compress(VectorOperation::add);
    if (assemble_pressure_mass)
        pressure_mass.compress(VectorOperation::add);

    static_matrix_built = true;
}


// Assemble the time-dependent Oseen step:
// 1. velocity convection/stabilization C_u(beta) in convection_matrix
// 2. RHS M_u u^n/dt + theta f^{n+1} + (1-theta)f^n
//    minus the explicit theta parts of diffusion/convection/Temam
// 3. optional PCD pressure operator F_p(beta)
// 4. system_matrix = static_matrix + convection_matrix, then velocity
//    Dirichlet boundary conditions are imposed
template <int dim>
void NavierStokes<dim>::assemble_timestep(
  const TrilinosWrappers::MPI::BlockVector &beta_solution)
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling timestep" << std::endl;

    const bool assemble_pcd_pressure = needs_pcd_pressure_operators();

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_face = quadrature_boundary->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_values_boundary(*fe,
                                         *quadrature_boundary,
                                         update_values | update_normal_vectors |
                                         update_quadrature_points |
                                           update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_convection_diffusion(dofs_per_cell,
                                                          dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    convection_matrix = 0.0;
    if (assemble_pcd_pressure)
        pressure_convection_diffusion = 0.0;
    system_rhs = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    // old_solution is u^n; beta_solution is the convective field beta.
    // In the current Oseen mode beta_solution == old_solution.
    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
    std::vector<Tensor<2, dim>> previous_velocity_gradients(n_q);
    std::vector<double> previous_velocity_divergences(n_q);
    std::vector<Tensor<1, dim>> beta_velocity_values(n_q);
    std::vector<double> beta_velocity_divergences(n_q);
    std::vector<Tensor<1, dim>> beta_boundary_values(n_q_face);

    const double previous_time = time - delta_t;
    // Boundary Function objects store their time internally; keep them in sync
    // before evaluating time-dependent Dirichlet/Neumann data.
    for (const auto &boundary_function : dirichlet)
        if (boundary_function.second != nullptr)
            const_cast<Function<dim> *>(boundary_function.second)->set_time(time);
    for (const auto &boundary_function : neumann)
        if (boundary_function.second != nullptr)
            const_cast<Function<dim> *>(boundary_function.second)->set_time(time);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;

        if (assemble_pcd_pressure)
            cell_pressure_convection_diffusion = 0.0;
        cell_rhs = 0.0;

        fe_values[velocity].get_function_values(old_solution,
                                                previous_velocity_values);
        fe_values[velocity].get_function_gradients(old_solution,
                                                   previous_velocity_gradients);
        fe_values[velocity].get_function_divergences(
          old_solution, previous_velocity_divergences);
        fe_values[velocity].get_function_values(beta_solution,
                                                beta_velocity_values);
        fe_values[velocity].get_function_divergences(
          beta_solution, beta_velocity_divergences);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const Tensor<1, dim> u_old = previous_velocity_values[q];
            const Tensor<2, dim> grad_u_old = previous_velocity_gradients[q];
            const double div_u_old = previous_velocity_divergences[q];
            const Tensor<1, dim> beta = beta_velocity_values[q];
            const double div_beta = beta_velocity_divergences[q];

            const Tensor<1, dim> f_new_loc =
              f(fe_values.quadrature_point(q), time);
            const Tensor<1, dim> f_old_loc =
              f(fe_values.quadrature_point(q), previous_time);
            const double tau_K =
              (stabilization_options.supg ?
                 compute_supg_tau(beta.norm(), cell->diameter()) :
                 0.0);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> phi_vel_i = fe_values[velocity].value(i, q);
                const Tensor<2, dim> grad_phi_vel_i =
                  fe_values[velocity].gradient(i, q);
                const Tensor<1, dim> streamline_test = grad_phi_vel_i * beta;
                const double psi_i = fe_values[pressure].value(i, q);
                const Tensor<1, dim> grad_psi_i =
                  fe_values[pressure].gradient(i, q);

                //LHS contributions
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> phi_vel_j = fe_values[velocity].value(j, q);
                    const Tensor<2, dim> grad_phi_vel_j = fe_values[velocity].gradient(j, q);
                    const double psi_j = fe_values[pressure].value(j, q);
                    const Tensor<1, dim> grad_psi_j =
                      fe_values[pressure].gradient(j, q);

                    if (assemble_pcd_pressure)
                    {
                        // F_p(beta) = (1/dt) M_p + theta*nu*A_p
                        //             + theta*C_p(beta),
                        // where (C_p)_{ij} = (beta . grad(psi_j), psi_i)
                        cell_pressure_convection_diffusion(i, j) +=
                          ((1.0 / delta_t) * psi_j * psi_i +
                           theta * nu *
                             scalar_product(grad_psi_j, grad_psi_i) +
                           theta * (beta * grad_psi_j) * psi_i) *
                          fe_values.JxW(q);

                        if (stabilization_options.temam)
                        {
                            // Scalar Temam counterpart in pressure space:
                            // theta/2 * ((div beta) psi_j, psi_i)
                            cell_pressure_convection_diffusion(i, j) +=
                              theta * 0.5 * div_beta * psi_j * psi_i *
                              fe_values.JxW(q);
                        }
                    }
                    
                    // Oseen/Picard convection in the velocity block
                    // theta * ((beta . grad) phi_j, phi_i)
                    cell_matrix(i, j) += theta * scalar_product(grad_phi_vel_j * beta, phi_vel_i) *
                       fe_values.JxW(q);

                    // Temam skew-symmetry correction in velocity space
                    // theta/2 * ((div beta) phi_j, phi_i)
                    if (stabilization_options.temam)
                    {
                        cell_matrix(i, j) +=
                          theta * 0.5 * div_beta *
                          scalar_product(phi_vel_j, phi_vel_i) *
                          fe_values.JxW(q);
                    }

                    // SUPG stabilization, residual part tested along beta:
                    // tau_K * ( (1/dt)u_j + (beta.grad)u_j + grad(p_j),
                    //           (beta.grad)v_i )_K
                    if (stabilization_options.supg)
                    {
                        Tensor<1, dim> supg_velocity_residual;
                        for (unsigned int d = 0; d < dim; ++d)
                            supg_velocity_residual[d] =
                              (1.0 / delta_t) * phi_vel_j[d] +
                              (grad_phi_vel_j * beta)[d];

                        cell_matrix(i, j) +=
                          tau_K *
                          scalar_product(supg_velocity_residual,
                                         streamline_test) *
                          fe_values.JxW(q);

                        cell_matrix(i, j) +=
                          tau_K *
                          scalar_product(grad_psi_j, streamline_test) *
                          fe_values.JxW(q);
                    }
                }

                // Time derivative RHS:
                // (u^n/dt, v_i)
                cell_rhs(i) += 1.0 / delta_t * scalar_product(u_old, phi_vel_i)  * fe_values.JxW(q);
                Tensor<1, dim> theta_force;
                for (unsigned int d = 0; d < dim; ++d)
                    theta_force[d] =
                      theta * f_new_loc[d] + (1.0 - theta) * f_old_loc[d];
                
                
                // (theta f^{n+1} + (1-theta) f^n, v_i)
                cell_rhs(i) +=
                  scalar_product(theta_force, phi_vel_i) *
                  fe_values.JxW(q);

                // -(1-theta) * nu * (grad u^n, grad v_i)
                cell_rhs(i) -=
                  (1.0 - theta) * nu *
                  scalar_product(grad_u_old, grad_phi_vel_i) *
                  fe_values.JxW(q);

                // -(1-theta) * (((u^n.grad)u^n), v_i)
                cell_rhs(i) -=
                  (1.0 - theta) *
                  scalar_product(grad_u_old * u_old, phi_vel_i) *
                  fe_values.JxW(q);

                if (stabilization_options.temam)
                {
                    // -(1-theta)/2 * ((div u^n) u^n, v_i)
                    cell_rhs(i) -=
                      (1.0 - theta) * 0.5 * div_u_old *
                      scalar_product(u_old, phi_vel_i) *
                      fe_values.JxW(q);
                }

                if (stabilization_options.supg)
                {
                    Tensor<1, dim> supg_rhs;
                    for (unsigned int d = 0; d < dim; ++d)
                        supg_rhs[d] =
                          (1.0 / delta_t) * u_old[d] + f_new_loc[d];

                    // tau_K * ((u^n/dt + f^{n+1}), (beta.grad)v_i)_K.
                    cell_rhs(i) +=
                      tau_K * scalar_product(supg_rhs, streamline_test) *
                      fe_values.JxW(q);
                }
            }
        }

        // BCs
        if (cell->at_boundary())
        {
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
            {
                if (!cell->face(face)->at_boundary())
                    continue;

                const types::boundary_id face_id = cell->face(face)->boundary_id();
                const bool has_pcd_robin_boundary =
                  assemble_pcd_pressure && (dirichlet.count(face_id) > 0);
                const bool has_neumann_boundary =
                  (neumann.count(face_id) > 0);

                if (!has_pcd_robin_boundary && !has_neumann_boundary)
                    continue;

                fe_values_boundary.reinit(cell, face);

                if (has_pcd_robin_boundary)
                {
                    fe_values_boundary[velocity].get_function_values(
                      beta_solution, beta_boundary_values);

                    for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                        const double beta_dot_n =
                          beta_boundary_values[q] *
                          fe_values_boundary.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            const double psi_i =
                              fe_values_boundary[pressure].value(i, q);

                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                const double psi_j =
                                  fe_values_boundary[pressure].value(j, q);

                                cell_pressure_convection_diffusion(i, j) +=
                                  theta * beta_dot_n * psi_j * psi_i *
                                  fe_values_boundary.JxW(q);
                            }
                        }
                    }
                }

                if (!has_neumann_boundary)
                    continue;

                const Function<dim> *boundary_function = neumann[face_id];

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    
                    const double p_out =
                      boundary_function->value(fe_values_boundary.quadrature_point(q));

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        cell_rhs(i) -=
                          p_out *
                          scalar_product(fe_values_boundary.normal_vector(q), fe_values_boundary[velocity].value(i, q)) *
                          fe_values_boundary.JxW(q);
                    }
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        convection_matrix.add(dof_indices, cell_matrix);
        if (assemble_pcd_pressure)
            pressure_convection_diffusion.add(dof_indices,
                                              cell_pressure_convection_diffusion);
        system_rhs.add(dof_indices, cell_rhs);
    }

    convection_matrix.compress(VectorOperation::add);
    if (assemble_pcd_pressure)
        pressure_convection_diffusion.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    system_matrix.copy_from(static_matrix);
    system_matrix.add(1.0, convection_matrix);

    //apply dirichlet BCs - velocity components only
    std::map<types::global_dof_index, double> boundary_values;

    // Mask for dirichlet
    ComponentMask mask_velocity(dim + 1, true);
    mask_velocity.set(dim, false); //exclude pressure component

    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet,
                                             boundary_values,
                                             mask_velocity);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution_owned,
                                       system_rhs,
                                       true);
}


template <int dim>
void NavierStokes<dim>::solve()
{
    pcout << "===============================================" << std::endl;

    // Check if GMRES or FGMRES
    const double rhs_norm = system_rhs.l2_norm();
    const double linear_tolerance =
      std::max(linear_absolute_tolerance, linear_relative_tolerance * rhs_norm);
    SolverControl solver_control(linear_max_iterations, linear_tolerance);

    pcout << "Solving the linear system" << std::endl;
    pcout << "  RHS norm = " << rhs_norm
          << ", tol = " << linear_tolerance
          << ", max iters = " << linear_max_iterations
          << ", GMRES restart = " << gmres_restart_length << std::endl;

    //initial guess: sol from previous timestep
    solution_owned = solution;

    if (preconditioner_kind == PreconditionerKind::PCD)
    {
        pressure_laplacian_discrete = 0.0;
        TrilinosWrappers::MPI::Vector neg_velocity_mass_diagonal_inverse;
        neg_velocity_mass_diagonal_inverse.reinit(solution_owned.block(0));

        // Diagonal inverse of the velocity mass:
        // D_u^{-1} ~= diag(M_u)^{-1}
        for (const auto i :
             neg_velocity_mass_diagonal_inverse.locally_owned_elements())
        {
            const double d = velocity_mass.block(0, 0).diag_element(i);
            neg_velocity_mass_diagonal_inverse[i] =
              (std::abs(d) > 1e-30 ? -1.0 / d : 0.0);
        }
        neg_velocity_mass_diagonal_inverse.compress(VectorOperation::insert);

        // Discrete pressure Laplacian used by PCD
        // A_p^disc ~= B diag(M_u)^{-1} B^T
        system_matrix.block(1, 0)
          .mmult(pressure_laplacian_discrete.block(1, 1),
                 system_matrix.block(0, 1),
                 neg_velocity_mass_diagonal_inverse);
       
        pressure_laplacian_discrete.block(1, 1)
          .add(1e-8, pressure_mass.block(1, 1));
        pressure_laplacian_discrete.compress(VectorOperation::add);
    }

    RequiredMatrices required_matrices;
    required_matrices.velocity_stiffness = &system_matrix.block(0, 0);
    if (needs_velocity_mass_matrix())
        required_matrices.velocity_mass = &velocity_mass.block(0, 0);
    if (needs_pressure_mass_matrix())
        required_matrices.pressure_mass = &pressure_mass.block(1, 1);
    if (needs_pcd_pressure_operators())
    {
        required_matrices.pressure_laplacian_discrete =
          &pressure_laplacian_discrete.block(1, 1);
        required_matrices.pressure_convection_diffusion =
          &pressure_convection_diffusion.block(1, 1);
    }
    required_matrices.B = &system_matrix.block(1, 0);
    required_matrices.BT = &system_matrix.block(0, 1);
    required_matrices.solution_template = &solution_owned;

    pcout << "  Building preconditioner = "
          << to_string(preconditioner_kind) << std::endl;
    auto preconditioner = make_preconditioner(preconditioner_kind);
    preconditioner->initialize(required_matrices);
    pcout << "  Preconditioner ready" << std::endl;

    SolverFGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
      additional_data(gmres_restart_length);
    SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control,
                                                            additional_data);

    pcout << "  Starting outer FGMRES" << std::endl;
    solver.solve(system_matrix, solution_owned, system_rhs, *preconditioner);
    pcout << "  Outer FGMRES completed" << std::endl;

    pcout << "  " << solver_control.last_step() << " FGMRES iterations"
          << std::endl;
}

template <int dim>
void NavierStokes<dim>::output()
{
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> names(dim, "velocity");
    names.push_back("pressure");

    data_out.add_data_vector(dof_handler, solution, names, interpretation);

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    const fs::path folder_path = output_folder();
    std::string folder = folder_path.generic_string();
    if (!folder.empty() && folder.back() != '/')
        folder += '/';

    const std::string filename_prefix = "solution";

    data_out.write_vtu_with_pvtu_record(folder,
                                        filename_prefix,
                                        timestep_number,
                                        MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
        const std::string pvtu_filename =
          filename_prefix + "_" + std::to_string(timestep_number) + ".pvtu";
        times_and_names.push_back({time, pvtu_filename});

        std::ofstream pvd_file((folder_path / "solution.pvd").string());
        DataOutBase::write_pvd_record(pvd_file, times_and_names);
    }

    pcout << "Output written for step " << timestep_number << "..." << std::endl;
    pcout << "===============================================" << std::endl;
}

template <int dim>
void NavierStokes<dim>::run()
{
    if (mpi_rank == 0)
    {
        const fs::path folder = output_folder();
        if (!fs::exists(folder))
            fs::create_directories(folder);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    pcout << "===============================================" << std::endl;
    pcout << "   Running " << simulation_name() << std::endl;
    pcout << "   Scheme: semi-implicit, nonlinear method = "
          << to_string(nonlinear_method)
          << ", Picard relaxation = " << picard_relaxation << std::endl;
    pcout << "   T_final = " << T << ", dt = " << delta_t << std::endl;
    pcout << "   GMRES restart = " << gmres_restart_length << std::endl;
    pcout << "   Pressure block regularization = disabled"
          << " (parsed value " << pressure_regularization << " is ignored)"
          << std::endl;
    pcout << "   Preconditioner = " << to_string(preconditioner_kind)
          << std::endl;
    pcout << "   Stabilization: Temam = "
          << (stabilization_options.temam ? "on" : "off")
          << ", grad-div = "
          << (stabilization_options.grad_div ? "on" : "off")
          << " (gamma = " << stabilization_options.gamma_grad_div << ")"
          << ", SUPG = " << (stabilization_options.supg ? "on" : "off")
          << ", PSPG = " << (stabilization_options.pspg ? "on" : "off")
          << std::endl;
    pcout << "   Linear max iters = " << linear_max_iterations
          << ", rel tol = " << linear_relative_tolerance
          << ", abs tol = " << linear_absolute_tolerance << std::endl;
    pcout << "===============================================" << std::endl;

    if (nonlinear_method != NonlinearMethod::None)
        pcout << "   Warning: nonlinear strategies are parsed but the nonlinear "
                 "iteration loop is not implemented yet."
              << std::endl;
    if (stabilization_options.pspg)
        pcout << "   Warning: PSPG option is parsed but its assembly is "
                 "not implemented yet."
              << std::endl;

    setup();

    //apply initial condition
    Functions::ZeroFunction<dim> zero_initial(dim + 1);
    if (initial_condition)
        VectorTools::interpolate(dof_handler, *initial_condition, solution_owned);
    else
        VectorTools::interpolate(dof_handler, zero_initial, solution_owned);

    solution = solution_owned;
    old_solution = solution;

    time = 0.0;
    timestep_number = 0;
    times_and_names.clear();
    static_matrix_built = false;

    output(); //write t=0

    while (time < T - 0.5 * delta_t)
    {
        time += delta_t;
        ++timestep_number;

        pcout << "Timestep " << std::setw(3) << timestep_number
              << ", time = " << std::setw(6) << std::fixed
              << std::setprecision(4) << time << " :\n";
        
        //build static matrices only once
        if (!static_matrix_built)
          assemble_static();

        old_solution = solution;

        assemble_timestep(old_solution);

        solve();

        solution = solution_owned;

        //forces (drag / lift)
        if (mpi_rank == 0 && timestep_number == 1)
        {
            if (dim == 2)
                std::ofstream("coefficients.txt", std::ios::trunc);
            if (dim == 3)
                std::ofstream("coefficients_3d.txt", std::ios::trunc);
        }

        compute_forces();
        output();
    }
}

template class NavierStokes<2>;
template class NavierStokes<3>;
