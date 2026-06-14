template <int dim>
BenchmarkRunMetadata NavierStokes<dim>::build_benchmark_metadata() const
{
    BenchmarkRunMetadata metadata;
    metadata.dimension = dim;
    metadata.mesh_file = mesh_file_name;
    metadata.mesh_name = BenchmarkRecorder::effective_mesh_name(benchmark_output_options.mesh_name, mesh_file_name);
    metadata.output_directory = output_folder();
    metadata.mpi_ranks = mpi_size;
    metadata.dofs_velocity = n_velocity_dofs;
    metadata.dofs_pressure = n_pressure_dofs;
    metadata.dofs_total = n_total_dofs;
    metadata.num_cells = mesh.n_global_active_cells();
    metadata.dt = delta_t;
    metadata.nu = nu;
    metadata.preconditioner = to_string(preconditioner_kind);
    metadata.nonlinear_method = to_string(nonlinear_method);
    return metadata;
}

template <int dim>
void NavierStokes<dim>::reset_current_step_metrics()
{
    current_step_linear_solves = 0;
    current_step_gmres_iterations = 0;
    current_step_preconditioner_inner_solves = 0;
    current_step_preconditioner_inner_iterations = 0;
    current_step_preconditioner_inner_failures = 0;
    current_step_nonlinear_iterations = 0;
    current_step_gmres_final_residual = std::numeric_limits<double>::quiet_NaN();
    current_step_gmres_converged = true;
    current_step_assembly_time = 0.0;
    current_step_linear_solve_time = 0.0;
    current_step_preconditioner_setup_time = 0.0;
}

template <int dim>
std::string NavierStokes<dim>::benchmark_output_directory() const
{
    return benchmark_output_options.output_directory;
}

template <int dim>
BenchmarkFieldMetrics NavierStokes<dim>::compute_field_metrics() const
{
    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_JxW_values);
    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> velocity_values(quadrature->size());
    std::vector<double> pressure_values(quadrature->size());
    std::vector<double> divergence(quadrature->size());

    double local_divergence_squared = 0.0;
    double local_divergence_linf = 0.0;
    double local_velocity_squared = 0.0;
    double local_velocity_linf = 0.0;
    double local_pressure_squared = 0.0;
    double local_pressure_integral = 0.0;
    double local_volume = 0.0;
    double local_kinetic_energy = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values[velocity].get_function_values(solution, velocity_values);
        fe_values[velocity].get_function_divergences(solution, divergence);
        fe_values[pressure].get_function_values(solution, pressure_values);

        for (unsigned int q = 0; q < quadrature->size(); ++q)
        {
            const double weight = fe_values.JxW(q);
            const double velocity_norm = velocity_values[q].norm();
            const double velocity_norm_squared = velocity_norm * velocity_norm;
            const double pressure_value = pressure_values[q];

            local_divergence_squared += divergence[q] * divergence[q] * weight;
            local_divergence_linf = std::max(local_divergence_linf, std::abs(divergence[q]));
            local_velocity_squared += velocity_norm_squared * weight;
            local_velocity_linf = std::max(local_velocity_linf, velocity_norm);
            local_pressure_squared += pressure_value * pressure_value * weight;
            local_pressure_integral += pressure_value * weight;
            local_volume += weight;
            local_kinetic_energy += 0.5 * velocity_norm_squared * weight;
        }
    }

    const double global_divergence_squared = Utilities::MPI::sum(local_divergence_squared, MPI_COMM_WORLD);
    const double global_velocity_squared = Utilities::MPI::sum(local_velocity_squared, MPI_COMM_WORLD);
    const double global_pressure_squared = Utilities::MPI::sum(local_pressure_squared, MPI_COMM_WORLD);
    const double global_pressure_integral = Utilities::MPI::sum(local_pressure_integral, MPI_COMM_WORLD);
    const double global_volume = Utilities::MPI::sum(local_volume, MPI_COMM_WORLD);

    BenchmarkFieldMetrics metrics;
    metrics.div_l2 = std::sqrt(global_divergence_squared);
    metrics.div_linf = Utilities::MPI::max(local_divergence_linf, MPI_COMM_WORLD);
    metrics.velocity_l2 = std::sqrt(global_velocity_squared);
    metrics.velocity_linf = Utilities::MPI::max(local_velocity_linf, MPI_COMM_WORLD);
    metrics.pressure_l2 = std::sqrt(global_pressure_squared);
    metrics.pressure_mean = (global_volume > 0.0 ? global_pressure_integral / global_volume : std::numeric_limits<double>::quiet_NaN());
    metrics.kinetic_energy = Utilities::MPI::sum(local_kinetic_energy, MPI_COMM_WORLD);

    return metrics;
}

template <int dim>
double NavierStokes<dim>::compute_pressure_difference(const Point<dim> &front_point, const Point<dim> &back_point) const
{
    const auto pressure_at = [&](const Point<dim> &point)
    {
        FEFaceValues<dim> fe_face_values(*fe, *quadrature_boundary, update_values | update_quadrature_points);
        FEValuesExtractors::Scalar pressure(dim);
        std::vector<double> pressure_values(quadrature_boundary->size());

        double local_distance_squared = std::numeric_limits<double>::infinity();
        double local_value = 0.0;
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned() || !cell->at_boundary())
                continue;

            for (unsigned int face = 0; face < cell->n_faces(); ++face)
            {
                if (!cell->face(face)->at_boundary())
                    continue;

                fe_face_values.reinit(cell, face);
                fe_face_values[pressure].get_function_values(solution, pressure_values);

                for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                {
                    const double distance_squared = point.distance_square(fe_face_values.quadrature_point(q));
                    if (distance_squared < local_distance_squared)
                    {
                        local_distance_squared = distance_squared;
                        local_value = pressure_values[q];
                    }
                }
            }
        }

        const double global_distance_squared = Utilities::MPI::min(local_distance_squared, MPI_COMM_WORLD);

        if (!std::isfinite(global_distance_squared))
            return std::numeric_limits<double>::quiet_NaN();

        const double tolerance = 1e-12 * std::max(1.0, global_distance_squared);
        const bool owns_closest_sample = local_distance_squared <= global_distance_squared + tolerance;
        const double global_value = Utilities::MPI::sum(owns_closest_sample ? local_value : 0.0, MPI_COMM_WORLD);
        const unsigned int global_count = Utilities::MPI::sum(owns_closest_sample ? 1U : 0U, MPI_COMM_WORLD);

        return global_count > 0 ? global_value / static_cast<double>(global_count) : std::numeric_limits<double>::quiet_NaN();
    };

    const double front_pressure = pressure_at(front_point);
    const double back_pressure = pressure_at(back_point);

    if (!std::isfinite(front_pressure) || !std::isfinite(back_pressure))
        return std::numeric_limits<double>::quiet_NaN();

    return front_pressure - back_pressure;
}

template <int dim>
void NavierStokes<dim>::write_benchmark_metrics(const double drag_coefficient, const double lift_coefficient, const double side_coefficient, const double delta_pressure, const double reference_velocity, const double reference_length)
{
    BenchmarkStepRecord record;
    record.step = timestep_number;
    record.time = time;
    record.dt = delta_t;
    record.nu = nu;
    record.reynolds_number = nu > 0.0 ? reference_velocity * reference_length / nu : std::numeric_limits<double>::quiet_NaN();
    record.drag_coefficient = drag_coefficient;
    record.lift_coefficient = lift_coefficient;
    record.side_coefficient = side_coefficient;
    record.delta_pressure = delta_pressure;
    record.reference_velocity = reference_velocity;
    record.reference_length = reference_length;
    record.field_metrics = compute_field_metrics();
    record.solver_metrics.linear_solves = current_step_linear_solves;
    record.solver_metrics.gmres_iterations = current_step_gmres_iterations;
    record.solver_metrics.preconditioner_inner_solves = current_step_preconditioner_inner_solves;
    record.solver_metrics.preconditioner_inner_iterations = current_step_preconditioner_inner_iterations;
    record.solver_metrics.preconditioner_inner_failures = current_step_preconditioner_inner_failures;
    record.solver_metrics.nonlinear_iterations = current_step_nonlinear_iterations;
    record.solver_metrics.gmres_final_residual = current_step_gmres_final_residual;
    record.solver_metrics.gmres_converged = current_step_linear_solves > 0 && current_step_gmres_converged;
    record.solver_metrics.assembly_time = current_step_assembly_time;
    record.solver_metrics.preconditioner_setup_time = current_step_preconditioner_setup_time;
    record.solver_metrics.linear_solve_time = current_step_linear_solve_time;
    record.solver_metrics.total_step_time = step_timer.wall_time();
    benchmark_recorder.record_step(record);
}
