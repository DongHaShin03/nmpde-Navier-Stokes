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
                // F_cyl = - int_Gamma_c sigma(u,p)n ds.
                force_x -= traction[0] * fe_face_values.JxW(q);
                force_y -= traction[1] * fe_face_values.JxW(q);
            }
        }
    }

    const double total_force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
    const double total_force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);

    // Drag/lift coefficient normalization in 2D:
    // C = F / (0.5 * U_ref^2 * L_ref), with density rho = 1.
    const double denominator = 0.5 *
                               force_coefficient_reference_velocity *
                               force_coefficient_reference_velocity *
                               force_coefficient_reference_length;

    const double C_D = total_force_x / denominator;
    const double C_L = total_force_y / denominator;

    // ----- QUI PRESSIONE DELTA P -----
    // Calcolare p_front - p_back nei punti benchmark davanti/dietro il cilindro.
    // Dopo l'implementazione, salvare Delta p insieme a Cd/Cl a ogni time step.

    // ----- QUI NORMA DIVERGENZA -----
    // Calcolare ||div(u_h)||_L2 come metrica di incomprimibilita'. Dopo il cambio,
    // usarla per confrontare grad-div disattivo e gamma_grad_div diversi.

    // ----- QUI STROUHAL / FREQUENZA LIFT -----
    // Per run instazionari, post-processare Cl(t) per stimare frequenza e St.
    // Dopo l'implementazione, produrre il valore medio su una finestra periodica.

    if (this->mpi_rank == 0)
    {
        this->pcout << "   Step " << this->timestep_number << " Forces: Drag="
                    << total_force_x << ", Lift=" << total_force_y << std::endl;
        this->pcout << "   Coeffs: Cd=" << C_D << ", Cl=" << C_L << std::endl;

        // ----- QUI OUTPUT CSV METRICHE -----
        // Sostituire/affiancare coefficients.txt con un CSV con header:
        // time,Cd,Cl,DeltaP,divL2,gmres_iters,nonlinear_iters,step_time.
        // Dopo il cambio, ogni benchmark deve essere confrontabile con script.
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

