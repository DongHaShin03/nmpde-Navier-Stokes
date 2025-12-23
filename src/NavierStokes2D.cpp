#include "NavierStokes2D.hpp"

void
NavierStokes2D::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         dim,
                                         fe_scalar_pressure,
                                         1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    
    // blockComponent[0] = 0 -> u_x component
    // blockComponent[1] = 0 -> u_y component
    // blockComponent[2] = 1 -> pressure component
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);

    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
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
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
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

void
NavierStokes2D::assemble()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                           update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;
  pressure_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix               = 0.0;
      cell_rhs                  = 0.0;
      cell_pressure_mass_matrix = 0.0;

      // Extracting old velocity values for the formulation
      std::vector<Tensor<1, dim>> old_velocity_values(n_q);
      fe_values[velocity].get_function_values(old_solution, old_velocity_values);

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const Tensor<1, dim> u_old = old_velocity_values[q];
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, dim> phi_u_i = fe_values[velocity].value(i, q);
          const Tensor<2, dim> grad_phi_u_i = fe_values[velocity].gradient(i,q);
          const double div_phi_u_i = fe_values[velocity].divergence(i, q);
          const double phi_p_i = fe_values[pressure].value(i, q);
              
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const Tensor <1, dim> phi_u_j = fe_values[velocity].value(j, q);
              const Tensor <2, dim> grad_phi_u_j = fe_values[velocity].gradient(j, q);
              const double div_phi_u_j = fe_values[velocity].divergence(j, q);
              const double phi_p_j = fe_values[pressure].value(j, q);

              // Assembly Matrix (M / dt)
              cell_matrix(i, j) += (phi_u_i * phi_u_j / deltat) * fe_values.JxW(q);
              
              // Matrix A 
              cell_matrix(i, j) += nu * scalar_product(grad_phi_u_i, grad_phi_u_j) * fe_values.JxW(q);

              // Convective matrix C(u_n) (with Picard Linearization)
              cell_matrix(i, j) += (grad_phi_u_j * u_old) * phi_u_i * fe_values.JxW(q);

              // Pressure Matrix B^T (Pressure Gradient)
              cell_matrix(i, j) -= div_phi_u_i * phi_p_j * fe_values.JxW(q);

              // Matrix B (Velocity Divergence)
              cell_matrix(i, j) -= phi_p_i * div_phi_u_j * fe_values.JxW(q);

              // Mass matrix
              cell_pressure_mass_matrix(i, j) += phi_p_i * phi_p_j / nu * fe_values.JxW(q);
              
            }

            cell_rhs(i) += (u_old * phi_u_i / deltat) * fe_values.JxW(q);

        }
      }

      // Boundary integral for Neumann BCs.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == 2)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) +=
                            -p_out *
                            scalar_product(fe_face_values.normal_vector(q),
                                           fe_face_values[velocity].value(i,
                                                                          q)) *
                            fe_face_values.JxW(q);
                        }
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

  // Dirichlet boundary conditions.
  {      
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    ComponentMask mask_velocity(dim + 1, true);
    mask_velocity.set(dim, false);

    //In Gamma[1] -> Inlet Velocity (Left Wall)
    boundary_functions[1] = &inlet_velocity;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask_velocity);

    boundary_functions.clear();
    Functions::ZeroFunction<dim> zero_function(dim + 1);

    // In Gamma[3] -- > u = 0;
    boundary_functions[3] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask_velocity);
    boundary_functions.clear();                                         


    // In Gamma[5] --> Cylinder with boundaries conditions u = 0;
    boundary_functions[5] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask_velocity);

    MatrixTools::apply_boundary_values(boundary_values, 
                                       system_matrix, 
                                       solution_owned, 
                                       system_rhs, 
                                       false);
                                       
  }
}

void
NavierStokes2D::solve()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(100000, 1e-2 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0));

  pcout << "Solving the linear system" << std::endl;

  solution_owned = 0.0;

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void
NavierStokes2D::compute_forces(const unsigned int &time_step)
{

  const unsigned int n_q_face = quadrature_face->size();

  // Drag Force
  double force_x = 0.0;

  //Lift Force
  double force_y = 0.0;

  FEFaceValues<dim> fe_face_values(*fe,
                                    *quadrature_face,
                                    update_values | update_gradients | 
                                    update_normal_vectors | update_JxW_values);
  
  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<2, dim>> grad_u(n_q_face);
  std::vector<double> p(n_q_face);
  std::vector<Tensor<1,dim>> normal(n_q_face);

  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    if(!cell -> is_locally_owned())
      continue;
    if(cell ->at_boundary())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        if(cell -> face(f) ->at_boundary() && 
           cell ->face(f) -> boundary_id() == 5)
        {
          fe_face_values.reinit(cell, f);

          fe_face_values[velocity].get_function_gradients(solution, grad_u);
          fe_face_values[pressure].get_function_values(solution, p);

          normal = fe_face_values.get_normal_vectors();

          for(unsigned int q = 0; q < n_q_face; ++q)
          {
            //sigma = -p * I + nu*(grad_u + grad_u^T)
            Tensor<2, dim> stress;

            for(unsigned int i = 0; i < dim; ++i)
              for(unsigned int j = 0; j < dim; ++j)
                stress[i][j] = nu * (grad_u[q][i][j] + grad_u[q][j][i]);
             
            for(unsigned int i = 0; i < dim; ++i)
              stress[i][i] -= p[q];    
            
            Tensor<1, dim> traction = stress * normal[q];
            
            force_x += traction[0] * fe_face_values.JxW(q);
            force_y += traction[1] * fe_face_values.JxW(q);
          }
        }
      }
    }  
  }
  double total_force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
  double total_force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);

  // Coefficient
  const double U_mean = 0.1;
  const double L = 25;
  const double den = U_mean * U_mean * L;

  // Drag Coefficient
  double C_D = total_force_x / den;

  // Lift Coefficient
  double C_L = total_force_y / den;

  if(mpi_rank == 0) 
  {
    pcout << "   Step " << time_step << " Forces: Drag=" << total_force_x << ", Lift=" << total_force_y << std::endl;
    pcout << "   Coeffs: Cd=" << C_D << ", Cl=" << C_L << std::endl;

    std::ofstream file("coefficients.txt", std::ios::app);
    file << time << " " << C_D << " " << C_L << std::endl;
  }
}


void NavierStokes2D::output(const unsigned int &time_step)
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

  // Nome base senza estensione (es. "solution-10")
  const std::string filename_base = "solution-" + std::to_string(time_step);
  
  // 1. Generazione dei file .pvtu e .vtu
  // Questa funzione genererà file del tipo "solution-10_0.pvtu" su disco
  data_out.write_vtu_with_pvtu_record("./",
                                      filename_base,
                                      time_step,
                                      MPI_COMM_WORLD);

  // 2. Generazione del file .pvd (Indice temporale)
  if (mpi_rank == 0)
    {
      // CORREZIONE QUI:
      // Poiché deal.II ha salvato il file come "solution-X_0.pvtu" (visto nei tuoi screenshot),
      // dobbiamo aggiungere "_0.pvtu" al nome nel registro PVD.
      std::string pvtu_filename = filename_base + "_0.pvtu";

      times_and_names.push_back({time, pvtu_filename});

      std::ofstream pvd_file("solution.pvd");
      DataOutBase::write_pvd_record(pvd_file, times_and_names);
    }

  pcout << "Output written to " << filename_base << "..." << std::endl;
  pcout << "===============================================" << std::endl;
}

void NavierStokes2D::run(){
  pcout << "===============================================" << std::endl;
  pcout << "   Running Navier-Stokes Simulation" << std::endl;
  pcout << "   T_final = " << T << ", dt = " << deltat << std::endl;
  pcout << "===============================================" << std::endl;

  setup();

  solution = 0.0;
  old_solution = 0.0;
  time = 0.0;

  output(0);
  unsigned int time_step = 0;

  while (time < T)
  {
    time += deltat;
    time_step++;

    pcout << std::endl << "Time step " << time_step << " at t=" << time << std::endl;

    assemble();

    solve();

    compute_forces(time_step);

    if(time_step % 10 == 0)
      output(time_step);
    
    old_solution = solution;  
  }
}