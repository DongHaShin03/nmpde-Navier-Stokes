#ifndef PCD_PRECONDITIONER_HPP
#define PCD_PRECONDITIONER_HPP

#include "NavierStokesPreconditioner.hpp"

#include <algorithm>
#include <stdexcept>

class PCD : public NavierStokesPreconditioner
{
    public:
        AssemblyFlags get_needed_matrices() const override
        {
            // PCD needs M_u for A_p and M_p for the Schur complement.
            // S_PCD^{-1} = A_p^(-1) F_p M_p^(-1).
            return {true, true};
        }

        void initialize(const RequiredMatrices &data) override
        {
            // Safety checks
            if (data.velocity_stiffness == nullptr ||
                data.pressure_mass == nullptr ||
                (data.pressure_laplacian_discrete == nullptr &&
                 data.pressure_laplacian == nullptr) ||
                data.pressure_convection_diffusion == nullptr ||
                data.BT == nullptr)
                throw std::runtime_error(
                  "PCD preconditioner requires F, B^T, M_p, A_p or A_p^disc, and F_p.");

            
            F = data.velocity_stiffness;
            B_T = data.BT;
            M_p = data.pressure_mass;
            // Standard PCD uses the continuous pressure Laplacian. The
            // discrete Laplacian remains available as a backup/experiment,
            // but it is more sensitive to mesh quality and ILU robustness.
            A_p = (data.pressure_laplacian != nullptr ?
                     data.pressure_laplacian :
                     data.pressure_laplacian_discrete);
            A_p_fallback = (data.pressure_laplacian != nullptr ?
                              data.pressure_laplacian_discrete :
                              nullptr);
            F_p = data.pressure_convection_diffusion;
            
            // ILU approximations of the inverses of F, M_p and A_p
            preconditioner_F.initialize(*F);
            preconditioner_Mp.initialize(*M_p);
            preconditioner_Ap.initialize(*A_p);
            if (A_p_fallback != nullptr)
                preconditioner_Ap_fallback.initialize(*A_p_fallback);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const override
        {   
            // src = [r_u; r_p], 
            // dst = [z_u; z_p]

            // We are using TrilinosWrappers due to the matrix format


            // Pressure vector r_p, z1 = M_p^{-1} r_p
            TrilinosWrappers::MPI::Vector mp_inverse_rhs;
            mp_inverse_rhs.reinit(src.block(1));

            // z2 = F_p z1
            TrilinosWrappers::MPI::Vector fp_times_mp_inverse_rhs;
            fp_times_mp_inverse_rhs.reinit(src.block(1));

            // zp = A_p^{-1} F_p M_p^{-1} r_p = S_PCD^{-1} r_p
            TrilinosWrappers::MPI::Vector pressure_part;
            pressure_part.reinit(src.block(1));

            // bt_pressure = - B^T z_p
            TrilinosWrappers::MPI::Vector bt_pressure;
            bt_pressure.reinit(src.block(0));

            // rhs of the velocity part of the PCD preconditioner: r_u + B^T z_p
            TrilinosWrappers::MPI::Vector corrected_velocity_rhs;
            corrected_velocity_rhs.reinit(src.block(0));

            // Final result of the velocity part of the PCD preconditioner: z_u = F^{-1}(r_u + B^T z_p)
            TrilinosWrappers::MPI::Vector velocity_part;
            velocity_part.reinit(src.block(0));

            mp_inverse_rhs = 0.0;
            pressure_part = 0.0;
            velocity_part = 0.0;


            // Pressure part of the PCD Schur inverse:
            // p ~= S_PCD^{-1} r_p
            //   = A_p^{-1} F_p M_p^{-1} r_p

            // Step 1.1: z1 ~= M_p^{-1} r_p
            preconditioner_Mp.vmult(mp_inverse_rhs, src.block(1));

            // Step 1.2: z2 = F_p z1.
            F_p->vmult(fp_times_mp_inverse_rhs, mp_inverse_rhs);

            // Step 1.3: p ~= A_p^{-1} z2
            if (!solve(*A_p,
                       pressure_part,
                       fp_times_mp_inverse_rhs,
                       preconditioner_Ap,
                       250,
                       1e-3,
                       1e-12) &&
                A_p_fallback != nullptr)
                solve(*A_p_fallback,
                      pressure_part,
                      fp_times_mp_inverse_rhs,
                      preconditioner_Ap_fallback,
                      250,
                      1e-3,
                      1e-12);

            // Step 2: r_u = r_u + B^T p.
            B_T->vmult(bt_pressure, pressure_part);
            corrected_velocity_rhs = src.block(0);
            corrected_velocity_rhs -= bt_pressure;

            // Step 3: Velocity part: u ~= F^{-1} * ru.
            solve(*F,
                  velocity_part,
                  corrected_velocity_rhs,
                  preconditioner_F,
                  100,
                  1e-2,
                  1e-12);

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
        const TrilinosWrappers::SparseMatrix *B_T = nullptr;
        const TrilinosWrappers::SparseMatrix *M_p = nullptr;
        const TrilinosWrappers::SparseMatrix *A_p = nullptr;
        const TrilinosWrappers::SparseMatrix *A_p_fallback = nullptr;
        const TrilinosWrappers::SparseMatrix *F_p = nullptr;

        TrilinosWrappers::PreconditionILU preconditioner_F;
        TrilinosWrappers::PreconditionILU preconditioner_Mp;
        TrilinosWrappers::PreconditionILU preconditioner_Ap;
        TrilinosWrappers::PreconditionILU preconditioner_Ap_fallback;
};

#endif
