#ifndef YOSIDA_PRECONDITIONER_HPP
#define YOSIDA_PRECONDITIONER_HPP

#include "NavierStokesPreconditioner.hpp"

#include <algorithm>
#include <stdexcept>

class Yosida : public NavierStokesPreconditioner
{
    public:
        AssemblyFlags get_needed_matrices() const override
        {
            // Yosida only needs M_u 
            // S_Y ~= -B diag(M_u)^{-1} B^T.
            return {true, false};
        }

        void initialize(const RequiredMatrices &data) override
        {
            if (data.velocity_stiffness == nullptr || data.B == nullptr ||
                data.BT == nullptr || data.solution_template == nullptr)
                throw std::runtime_error(
                  "Yosida preconditioner requires F, B, B^T and a solution template");

            F = data.velocity_stiffness;
            B = data.B;
            B_T = data.BT;         
            M = data.velocity_mass;
            velocity_max_iterations = data.preconditioner_iterations.yosida_velocity_max_iterations;
            schur_max_iterations = data.preconditioner_iterations.yosida_schur_max_iterations;
            correction_max_iterations = data.preconditioner_iterations.yosida_correction_max_iterations;
            relative_tolerance =data.preconditioner_iterations.yosida_relative_tolerance;
            absolute_tolerance =data.preconditioner_iterations.yosida_absolute_tolerance;
                                                

            diag_D_inv.reinit(data.solution_template->block(0));
            neg_diag_D_inv.reinit(data.solution_template->block(0));

            for (const auto i : diag_D_inv.locally_owned_elements())
            {
                const double d = M->diag_element(i);
                diag_D_inv[i] = 1.0 / d;
                neg_diag_D_inv[i] = -1.0 / d;
            }
            diag_D_inv.compress(VectorOperation::insert);
            neg_diag_D_inv.compress(VectorOperation::insert);

            // negative_S_tilde = -B diag(M_u)^{-1} B^T.
            B->mmult(negative_S_tilde, *B_T, neg_diag_D_inv);

            preconditioner_F.initialize(*F);
            preconditioner_S.initialize(negative_S_tilde);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            TrilinosWrappers::MPI::Vector yu;
            yu.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector yp;
            yp.reinit(src.block(1));
            TrilinosWrappers::MPI::Vector tmp_p;
            tmp_p.reinit(src.block(1));
            TrilinosWrappers::MPI::Vector tmp_u;
            tmp_u.reinit(src.block(0));
            TrilinosWrappers::MPI::Vector correction_u;
            correction_u.reinit(src.block(0));

            SolverControl solver_F(velocity_max_iterations,
                                   std::max(absolute_tolerance,
                                            relative_tolerance *
                                              src.block(0).l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);
            
            // Step 1: y_u ~= F^{-1} r_u.
            time_section("yosida_vmult/f_solve", [&]()
            {
                try
                {
                    solver_gmres.solve(*F, yu, src.block(0), preconditioner_F);
                    record_inner_solve(solver_F.last_step(), true);
                }
                catch (const SolverControl::NoConvergence &)
                {
                    record_inner_solve(solver_F.last_step(), false);
                }
            });

            // Step 2: r_p = r_p - B y_u.
            B->vmult(tmp_p, yu);
            tmp_p *= -1.0;
            tmp_p += src.block(1);

            SolverControl solver_S(schur_max_iterations,
                                   std::max(absolute_tolerance,
                                            relative_tolerance * tmp_p.l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_S(solver_S);

            // Step 2: y_p ~= S_Y^(-1) r_p.
            time_section("yosida_vmult/schur_solve", [&]()
            {
                try
                {
                    solver_gmres_S.solve(negative_S_tilde, yp, tmp_p, preconditioner_S);
                    record_inner_solve(solver_S.last_step(), true);
                }
                catch (const SolverControl::NoConvergence &)
                {
                    record_inner_solve(solver_S.last_step(), false);
                }
            });

            dst.block(1) = yp;

            // Step 3 correction_u ~= F^{-1}(-B^T y_p).
            // Correction RHS in velocity space: stored B_T is -B^T.
            B_T->vmult(tmp_u, dst.block(1));

            SolverControl solver_F2(correction_max_iterations,
                                    std::max(absolute_tolerance,
                                             relative_tolerance *
                                               tmp_u.l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres2(solver_F2);
            
            time_section("yosida_vmult/f_correction_solve", [&]()
            {
                try
                {
                    solver_gmres2.solve(*F, correction_u, tmp_u, preconditioner_F);
                    record_inner_solve(solver_F2.last_step(), true);
                }
                catch (const SolverControl::NoConvergence &)
                {
                    record_inner_solve(solver_F2.last_step(), false);
                }
            });

            // Final block-triangular Yosida application:
            // y_u - correction_u, y_p.
            dst.block(0) = yu;
            dst.block(0) -= correction_u;
        }

    private:
        const TrilinosWrappers::SparseMatrix *F = nullptr;
        const TrilinosWrappers::SparseMatrix *B_T = nullptr;
        const TrilinosWrappers::SparseMatrix *B = nullptr;
        const TrilinosWrappers::SparseMatrix *M = nullptr;

        TrilinosWrappers::SparseMatrix negative_S_tilde;
        TrilinosWrappers::MPI::Vector diag_D_inv;
        TrilinosWrappers::MPI::Vector neg_diag_D_inv;

        TrilinosWrappers::PreconditionILU preconditioner_F;
        TrilinosWrappers::PreconditionILU preconditioner_S;

        unsigned int velocity_max_iterations = 100000;
        unsigned int schur_max_iterations = 100000;
        unsigned int correction_max_iterations = 100000;
        double relative_tolerance = 1e-2;
        double absolute_tolerance = 1e-14;
};

#endif
