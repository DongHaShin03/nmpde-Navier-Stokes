#include "NavierStokes2D.hpp"

#include <stdexcept>

void NavierStokes2D::set_force_coefficient_parameters(
  const double reference_velocity,
  const double reference_length,
  const types::boundary_id cylinder_boundary_id_)
{
    force_coefficient_reference_velocity = reference_velocity;
    force_coefficient_reference_length = reference_length;
    cylinder_boundary_id = cylinder_boundary_id_;
}

void NavierStokes2D::compute_forces()
{
    if (force_coefficient_reference_velocity <= 0.0 ||
        force_coefficient_reference_length <= 0.0 ||
        cylinder_boundary_id == static_cast<types::boundary_id>(-1))
        throw std::runtime_error(
          "NavierStokes2D force-coefficient parameters were not initialized.");

    const unsigned int n_q_face = this->quadrature_boundary->size();

    double force_x = 0.0;
    double force_y = 0.0;

    FEFaceValues<dim> fe_face_values(*this->fe,
                                     *this->quadrature_boundary,
                                     update_values | update_gradients |
                                       update_normal_vectors | update_JxW_values);

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<2, dim>> grad_u(n_q_face);
    std::vector<double> p(n_q_face);
    std::vector<Tensor<1, dim>> normal(n_q_face);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        if (!cell->at_boundary())
            continue;

        for (unsigned int face = 0; face < cell->n_faces(); ++face)
        {
            if (!(cell->face(face)->at_boundary() &&
                  cell->face(face)->boundary_id() == cylinder_boundary_id))
                continue;

            fe_face_values.reinit(cell, face);

            fe_face_values[velocity].get_function_gradients(this->solution, grad_u);
            fe_face_values[pressure].get_function_values(this->solution, p);

            normal = fe_face_values.get_normal_vectors();

            for (unsigned int q = 0; q < n_q_face; ++q)
            {
                Tensor<2, dim> stress;

                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                        stress[i][j] = this->nu *
                                       (grad_u[q][i][j] + grad_u[q][j][i]);

                for (unsigned int i = 0; i < dim; ++i)
                    stress[i][i] -= p[q];

                const Tensor<1, dim> traction = stress * normal[q];

                force_x += traction[0] * fe_face_values.JxW(q);
                force_y += traction[1] * fe_face_values.JxW(q);
            }
        }
    }

    const double total_force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
    const double total_force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);

    const double denominator = force_coefficient_reference_velocity *
                               force_coefficient_reference_velocity *
                               force_coefficient_reference_length;

    const double C_D = total_force_x / denominator;
    const double C_L = total_force_y / denominator;

    if (this->mpi_rank == 0)
    {
        this->pcout << "   Step " << this->timestep_number << " Forces: Drag="
                    << total_force_x << ", Lift=" << total_force_y << std::endl;
        this->pcout << "   Coeffs: Cd=" << C_D << ", Cl=" << C_L << std::endl;

        std::ofstream file("coefficients.txt", std::ios::app);
        file << this->time << " " << C_D << " " << C_L << std::endl;
    }
}

std::string NavierStokes2D::simulation_name() const
{
    return "Navier-Stokes 2D Simulation";
}

std::string NavierStokes2D::output_folder() const
{
    return "results";
}

