#ifndef SIMPLE_HPP
#define SIMPLE_HPP

#include "NavierStokesPreconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

class Simple : public NavierStokesPreconditioner
{
    public:

        AssemblyFlags get_needed_matrices() const override
        {
            // SIMPLE needs M_p only as a tiny pressure-space shift for
            // S_SIMPLE = B diag(F)^{-1} B^T, which otherwise has the constant
            // pressure nullspace.
            return {false, true};
        }
        void initialize(const RequiredMatrices &data) override
        {
            if (data.velocity_stiffness == nullptr || data.pressure_mass == nullptr ||
                data.B == nullptr || data.BT == nullptr ||
                data.solution_template == nullptr)
                throw std::runtime_error(
                  "Simple preconditioner requires F, B, B^T, M_p and a solution template");

            F   = data.velocity_stiffness;
            M_p = data.pressure_mass;
            B   = data.B;
            B_T = data.BT;
            pressure_relaxation =
              std::min(1.0, std::max(0.0, data.simple_pressure_relaxation));
            velocity_max_iterations = data.preconditioner_iterations.simple_velocity_max_iterations;
            schur_max_iterations = data.preconditioner_iterations.simple_schur_max_iterations;
            velocity_relative_tolerance =data.preconditioner_iterations.simple_velocity_relative_tolerance;
            schur_relative_tolerance =data.preconditioner_iterations.simple_schur_relative_tolerance;
            absolute_tolerance =data.preconditioner_iterations.preconditioner_absolute_tolerance;

            diag_F_inv.reinit(data.solution_template->block(0));
            neg_diag_F_inv.reinit(data.solution_template->block(0));

            // D^{-1}, where D = diag(F). This is the SIMPLE approximation
            // F^{-1} ~= D^{-1}.
            for (const auto i : diag_F_inv.locally_owned_elements())
            {
                const double d = F->diag_element(i);
                diag_F_inv[i] =
                  (std::abs(d) > 1e-30 ? 1.0 / d : 0.0);
                neg_diag_F_inv[i] = -diag_F_inv[i];
            }
            diag_F_inv.compress(VectorOperation::insert);
            neg_diag_F_inv.compress(VectorOperation::insert);

            // The stored upper-right block is -B^T. Multiplying by
            // -diag(F)^{-1} therefore gives the positive SIMPLE Schur:
            // S_SIMPLE = B diag(F)^{-1} B^T.
            B->mmult(simple_schur, *B_T, neg_diag_F_inv);
            simple_schur.add(1e-8, *M_p);
            simple_schur.compress(VectorOperation::add);

            preconditioner_F.initialize(*F);
            preconditioner_S.initialize(simple_schur);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            TrilinosWrappers::MPI::Vector u_hat;
            u_hat.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector pressure_rhs;
            pressure_rhs.reinit(src.block(1));
            TrilinosWrappers::MPI::Vector pressure_correction;
            pressure_correction.reinit(src.block(1));
            TrilinosWrappers::MPI::Vector minus_bt_pressure;
            minus_bt_pressure.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector velocity_correction;
            velocity_correction.reinit(src.block(0));

            // Step 1: u_hat ~= F^{-1} r_u.
            time_section("simple_vmult/f_solve_limited", [&]()
            {
                solve(*F,
                      u_hat,
                      src.block(0),
                      preconditioner_F,
                      velocity_max_iterations,
                      velocity_relative_tolerance,
                      absolute_tolerance);
            });

            // Step 2: r_p = r_p - B u_hat.
            B->vmult(pressure_rhs, u_hat);
            pressure_rhs *= -1.0;
            pressure_rhs += src.block(1);

            // Step 3: z_p ~= S_SIMPLE^{-1} r_p,
            // where S_SIMPLE = B diag(F)^{-1} B^T.
            time_section("simple_vmult/schur_solve_limited", [&]()
            {
                solve(simple_schur,
                      pressure_correction,
                      pressure_rhs,
                      preconditioner_S,
                      schur_max_iterations,
                      schur_relative_tolerance,
                      absolute_tolerance);
            });
            
            // z_p = alpha * z_p, where alpha is the pressure relaxation factor      
            pressure_correction *= pressure_relaxation;

            // Step 4: z_u = u_hat + diag(F)^{-1} B^T p_corr.
            B_T->vmult(minus_bt_pressure, pressure_correction);
            for (const auto i : velocity_correction.locally_owned_elements())
                velocity_correction[i] =
                  neg_diag_F_inv[i] * minus_bt_pressure[i];
            velocity_correction.compress(VectorOperation::insert);

            dst.block(0) = u_hat;
            dst.block(0) += velocity_correction;
            dst.block(1) = pressure_correction;
        }

    private:
        bool solve(
          const TrilinosWrappers::SparseMatrix      &matrix,
          TrilinosWrappers::MPI::Vector             &solution,
          const TrilinosWrappers::MPI::Vector       &rhs,
          const TrilinosWrappers::PreconditionILU   &preconditioner,
          const unsigned int                         max_iterations,
          const double                               relative_tolerance,
          const double                               absolute_tolerance) const
        {
            const double rhs_norm = rhs.l2_norm();
            solution = 0.0;

            if (rhs_norm == 0.0)
                return true;

            SolverControl solver_control(
              max_iterations,
              std::max(absolute_tolerance, relative_tolerance * rhs_norm));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

            try
            {
                solver.solve(matrix, solution, rhs, preconditioner);
                return true;
            }
            catch (const SolverControl::NoConvergence &)
            {
                return false;
            }
        }

        const TrilinosWrappers::SparseMatrix *F = nullptr;
        const TrilinosWrappers::SparseMatrix *M_p = nullptr;
        const TrilinosWrappers::SparseMatrix *B = nullptr;
        const TrilinosWrappers::SparseMatrix *B_T = nullptr;
        double pressure_relaxation = 0.7;

        TrilinosWrappers::SparseMatrix simple_schur;
        TrilinosWrappers::MPI::Vector diag_F_inv;
        TrilinosWrappers::MPI::Vector neg_diag_F_inv;

        TrilinosWrappers::PreconditionILU preconditioner_F;
        TrilinosWrappers::PreconditionILU preconditioner_S;

        unsigned int velocity_max_iterations = 5;
        unsigned int schur_max_iterations = 20;
        double velocity_relative_tolerance = 1e-2;
        double schur_relative_tolerance = 1e-3;
        double absolute_tolerance = 1e-12;
};

#endif
