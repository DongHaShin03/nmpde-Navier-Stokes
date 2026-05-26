#include "NavierStokes3D.hpp"

#include <fstream>
#include <stdexcept>

void NavierStokes3D::set_force_coefficient_parameters(
  const double reference_velocity,
  const double reference_area,
  const types::boundary_id cylinder_boundary_id_)
{
    force_coefficient_reference_velocity = reference_velocity;
    force_coefficient_reference_area = reference_area;
    cylinder_boundary_id = cylinder_boundary_id_;
}

void NavierStokes3D::compute_forces()
{
    if (force_coefficient_reference_velocity <= 0.0 ||
        force_coefficient_reference_area <= 0.0 ||
        cylinder_boundary_id == static_cast<types::boundary_id>(-1))
        throw std::runtime_error(
          "NavierStokes3D force-coefficient parameters were not initialized.");

    const unsigned int n_q_face = this->quadrature_boundary->size();

    Tensor<1, dim> force;
    for (unsigned int d = 0; d < dim; ++d)
        force[d] = 0.0;

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
                Tensor<2, dim> viscous_stress;

                // Viscous stress tensor:
                // tau(u) = nu * (grad u + grad u^T).
                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                        viscous_stress[i][j] =
                          this->nu * (grad_u[q][i][j] + grad_u[q][j][i]);

                // Boundary traction of the Cauchy stress:
                // sigma(u,p)n = tau(u)n - p n.
                const Tensor<1, dim> viscous_traction =
                  viscous_stress * normal[q];
                const Tensor<1, dim> pressure_traction = -p[q] * normal[q];
                const Tensor<1, dim> traction =
                  viscous_traction + pressure_traction;

                // The normal points out of the fluid domain, so the force on
                // the cylinder has the opposite sign:
                // F_cyl = - int_Gamma_c sigma(u,p)n dS.
                for (unsigned int d = 0; d < dim; ++d)
                    force[d] -= traction[d] * fe_face_values.JxW(q);
            }
        }
    }

    Tensor<1, dim> total_force;
    for (unsigned int d = 0; d < dim; ++d)
        total_force[d] = Utilities::MPI::sum(force[d], MPI_COMM_WORLD);

    // Drag/lift/side-force coefficient normalization in 3D:
    // C = F / (0.5 * U_ref^2 * A_ref), with density rho = 1.
    // A_ref is the projected reference area, usually D * span for the cylinder.
    const double denominator = 0.5 *
                               force_coefficient_reference_velocity *
                               force_coefficient_reference_velocity *
                               force_coefficient_reference_area;

    const double C_D = total_force[0] / denominator;
    const double C_L = total_force[1] / denominator;
    const double C_S = total_force[2] / denominator;

    if (this->mpi_rank == 0)
    {
        this->pcout << "   Step " << this->timestep_number
                    << " Forces: Drag=" << total_force[0]
                    << ", Lift=" << total_force[1]
                    << ", Side=" << total_force[2] << std::endl;
        this->pcout << "   Coeffs: Cd=" << C_D
                    << ", Cl=" << C_L
                    << ", Cs=" << C_S << std::endl;

        std::ofstream file("coefficients_3d.txt", std::ios::app);
        file << this->time << " " << C_D << " " << C_L << " " << C_S
             << std::endl;
    }
}

std::string NavierStokes3D::simulation_name() const
{
    return "Navier-Stokes 3D Simulation";
}

std::string NavierStokes3D::output_folder() const
{
    return "results_3d";
}

