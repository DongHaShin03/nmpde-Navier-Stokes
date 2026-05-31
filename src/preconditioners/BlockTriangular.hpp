#ifndef BLOCK_TRIANGULAR_HPP
#define BLOCK_TRIANGULAR_HPP

#include "NavierStokesPreconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

class BlockTriangular : public NavierStokesPreconditioner
{
    public:

        AssemblyFlags get_needed_matrices() const override
        {
            // M_p is used only as a tiny pressure-space shift for the Schur
            // approximation S ~= B diag(F)^{-1} B^T.
            return {false, true};
        }
        void initialize(const RequiredMatrices &data) override
        {
            if (data.velocity_stiffness == nullptr || data.pressure_mass == nullptr ||
                data.B == nullptr || data.BT == nullptr ||
                data.solution_template == nullptr)
                throw std::runtime_error(
                  "BlockTriangular preconditioner requires F, B, B^T, M_p and a solution template.");

            F   = data.velocity_stiffness;
            M_p = data.pressure_mass;
            B   = data.B;
            B_T = data.BT;
            velocity_max_iterations =data.preconditioner_iterations.block_triangular_velocity_max_iterations;
            schur_max_iterations =data.preconditioner_iterations.block_triangular_schur_max_iterations;
            velocity_relative_tolerance =data.preconditioner_iterations.block_triangular_velocity_relative_tolerance;
            schur_relative_tolerance =data.preconditioner_iterations.block_triangular_schur_relative_tolerance;
            absolute_tolerance = data.preconditioner_iterations.preconditioner_absolute_tolerance;

            diag_F_inv.reinit(data.solution_template->block(0));
            neg_diag_F_inv.reinit(data.solution_template->block(0));

            // D^{-1}, where D = diag(F)
            for (const auto i : diag_F_inv.locally_owned_elements())
            {
                const double d = F->diag_element(i);
                diag_F_inv[i] =
                  (std::abs(d) > 1e-30 ? 1.0 / d : 0.0);
                neg_diag_F_inv[i] = -diag_F_inv[i];
            }
            diag_F_inv.compress(VectorOperation::insert);
            neg_diag_F_inv.compress(VectorOperation::insert);

            B->mmult(schur_approximation, *B_T, neg_diag_F_inv);
            schur_approximation.add(1e-8, *M_p);
            schur_approximation.compress(VectorOperation::add);

            preconditioner_F.initialize(*F);
            preconditioner_S.initialize(schur_approximation);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            TrilinosWrappers::MPI::Vector pressure_part;
            pressure_part.reinit(src.block(1));
            TrilinosWrappers::MPI::Vector minus_bt_pressure;
            minus_bt_pressure.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector velocity_rhs;
            velocity_rhs.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector velocity_part;
            velocity_part.reinit(src.block(0));

            // Step 1: z_p ~= S_tilde^{-1} r_p.
            time_section("block_triangular_vmult/schur_solve", [&]()
            {
                solve(schur_approximation,
                      pressure_part,
                      src.block(1),
                      preconditioner_S,
                      schur_max_iterations,
                      schur_relative_tolerance,
                      absolute_tolerance);
            });

            // Step 2: solve r_u = r_u - B_T z_p.
            B_T->vmult(minus_bt_pressure, pressure_part);
            velocity_rhs = src.block(0);
            velocity_rhs -= minus_bt_pressure;

            // Step 3: z_u = F^{-1} r_u.
            time_section("block_triangular_vmult/f_solve", [&]()
            {
                solve(*F,
                      velocity_part,
                      velocity_rhs,
                      preconditioner_F,
                      velocity_max_iterations,
                      velocity_relative_tolerance,
                      absolute_tolerance);
            });

            dst.block(0) = velocity_part;
            dst.block(1) = pressure_part;
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

        TrilinosWrappers::SparseMatrix schur_approximation;
        TrilinosWrappers::MPI::Vector diag_F_inv;
        TrilinosWrappers::MPI::Vector neg_diag_F_inv;

        TrilinosWrappers::PreconditionILU preconditioner_F;
        TrilinosWrappers::PreconditionILU preconditioner_S;
        unsigned int velocity_max_iterations = 100;
        unsigned int schur_max_iterations = 250;
        double velocity_relative_tolerance = 1e-2;
        double schur_relative_tolerance = 1e-3;
        double absolute_tolerance = 1e-12;
};

#endif
