#include "NavierStokes.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

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
  , theta(theta_)
  , delta_t(delta_t_)
  , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  , mesh(MPI_COMM_WORLD)
  , dof_handler(mesh)
  , pcout(std::cout, mpi_rank == 0)
{}

template <int dim>
void NavierStokes<dim>::setup()
{
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream mesh_file(mesh_file_name);
        grid_in.read_msh(mesh_file);

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

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim)
                    coupling[c][d] = DoFTools::always;
                else
                    coupling[c][d] = DoFTools::none;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
          block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        coupling,
                                        sparsity_pressure_mass);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(sparsity);
        pressure_mass.reinit(sparsity_pressure_mass);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
        old_solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

template <int dim>
void NavierStokes<dim>::assemble()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

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
                                           update_JxW_values);

    
                                           // check what does full matrix
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    system_rhs = 0.0;
    pressure_mass = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> old_velocity_values(n_q);
    std::vector<Tensor<2, dim>> old_velocity_grads(n_q);

    const double previous_time = time - delta_t;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_rhs = 0.0;
        cell_pressure_mass_matrix = 0.0;

        fe_values[velocity].get_function_values(old_solution, old_velocity_values);
        fe_values[velocity].get_function_gradients(old_solution, old_velocity_grads);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const Tensor<1, dim> f_new_loc = f(fe_values.quadrature_point(q), time);
            const Tensor<1, dim> f_old_loc =
              f(fe_values.quadrature_point(q), previous_time);
            const Tensor<1, dim> u_old = old_velocity_values[q];
            const Tensor<2, dim> u_old_grad = old_velocity_grads[q];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> phi_vel_i = fe_values[velocity].value(i, q);
                const Tensor<2, dim> grad_phi_vel_i =
                  fe_values[velocity].gradient(i, q);
                const double div_phi_vel_i = fe_values[velocity].divergence(i, q);
                const double psi_i = fe_values[pressure].value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const Tensor<1, dim> phi_vel_j = fe_values[velocity].value(j, q);
                    const Tensor<2, dim> grad_phi_vel_j =
                      fe_values[velocity].gradient(j, q);
                    const double div_phi_vel_j = fe_values[velocity].divergence(j, q);
                    const double psi_j = fe_values[pressure].value(j, q);

                    cell_matrix(i, j) +=
                      1.0 / delta_t * scalar_product(phi_vel_j, phi_vel_i) *
                      fe_values.JxW(q);
                    cell_matrix(i, j) +=
                      theta * nu * scalar_product(grad_phi_vel_j, grad_phi_vel_i) *
                      fe_values.JxW(q);
                    cell_matrix(i, j) +=
                      theta * (grad_phi_vel_j * u_old) * phi_vel_i *
                      fe_values.JxW(q);
                    cell_matrix(i, j) -= psi_j * div_phi_vel_i * fe_values.JxW(q);
                    cell_matrix(i, j) -= psi_i * div_phi_vel_j * fe_values.JxW(q);

                    cell_pressure_mass_matrix(i, j) +=
                      psi_i * psi_j / nu * fe_values.JxW(q);
                }

                cell_rhs(i) += theta * f_new_loc * phi_vel_i * fe_values.JxW(q);
                cell_rhs(i) +=
                  (1.0 - theta) * f_old_loc * phi_vel_i * fe_values.JxW(q);

                cell_rhs(i) +=
                  1.0 / delta_t * u_old * phi_vel_i * fe_values.JxW(q);
                cell_rhs(i) -=
                  (1.0 - theta) * nu * scalar_product(u_old_grad, grad_phi_vel_i) *
                  fe_values.JxW(q);
                cell_rhs(i) -=
                  (1.0 - theta) * (u_old_grad * u_old) * phi_vel_i *
                  fe_values.JxW(q);
            }
        }

        // Border condition
        if (cell->at_boundary())
        {
            for (unsigned int face = 0; face < cell->n_faces(); ++face)
            {
                if (!cell->face(face)->at_boundary())
                    continue;

                const types::boundary_id face_id = cell->face(face)->boundary_id();
                if (!neumann.count(face_id))
                    continue;

                fe_values_boundary.reinit(cell, face);
                const Function<dim> *boundary_function = neumann[face_id];

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    const double h_loc =
                      boundary_function->value(fe_values_boundary.quadrature_point(q));
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        cell_rhs(i) +=
                          h_loc *
                          scalar_product(fe_values_boundary.normal_vector(q),
                                         fe_values_boundary[velocity].value(i, q)) *
                          fe_values_boundary.JxW(q);
                    }
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
        pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);

    std::map<types::global_dof_index, double> boundary_values;

    // Mask for dirichlet
    ComponentMask mask_velocity(dim + 1, true);
    mask_velocity.set(dim, false);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             dirichlet,
                                             boundary_values,
                                             mask_velocity);

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution_owned,
                                       system_rhs,
                                       false);
}

template <int dim>
void NavierStokes<dim>::solve()
{
    pcout << "===============================================" << std::endl;

    // Check if GMRES or FGMRES
    const double linear_tolerance = std::max(1e-12, 1e-2 * system_rhs.l2_norm());
    SolverControl solver_control(100000, linear_tolerance);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    pcout << "Solving the linear system" << std::endl;

    solution_owned = 0.0;

    // Baseline for the preconditioner ----- #TODO
    PreconditionIdentity preconditioner;
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);

    pcout << "  " << solver_control.last_step() << " GMRES iterations"
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

    const std::string folder = output_folder();
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

        std::ofstream pvd_file(folder + "/solution.pvd");
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
            fs::create_directory(folder);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    pcout << "===============================================" << std::endl;
    pcout << "   Running " << simulation_name() << std::endl;
    pcout << "   T_final = " << T << ", dt = " << delta_t << std::endl;
    pcout << "===============================================" << std::endl;

    setup();

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

    output();

    while (time < T - 0.5 * delta_t)
    {
        time += delta_t;
        ++timestep_number;

        pcout << "Timestep " << std::setw(3) << timestep_number
              << ", time = " << std::setw(4) << std::fixed
              << std::setprecision(2) << time << " :\n";

        assemble();
        solve();

        solution = solution_owned;

        if (dim == 2 && mpi_rank == 0 && timestep_number == 1)
            std::ofstream("coefficients.txt", std::ios::trunc);

        compute_forces();
        output();
        old_solution = solution;
    }
}

template class NavierStokes<2>;
template class NavierStokes<3>;


