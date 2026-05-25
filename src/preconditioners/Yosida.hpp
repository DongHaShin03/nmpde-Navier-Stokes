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
            const unsigned int maxiter = 100000;
            const double relative_tolerance = 1e-2;

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

            SolverControl solver_F(maxiter,
                                   std::max(1e-14,
                                            relative_tolerance *
                                              src.block(0).l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);
            
            // Step 1: y_u ~= F^{-1} r_u.
            solver_gmres.solve(*F, yu, src.block(0), preconditioner_F);

            // Step 2: r_p = r_p - B y_u.
            B->vmult(tmp_p, yu);
            tmp_p *= -1.0;
            tmp_p += src.block(1);

            SolverControl solver_S(maxiter,
                                   std::max(1e-14,
                                            relative_tolerance * tmp_p.l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_S(solver_S);

            // Step 2: y_p ~= S_Y^(-1) r_p.
            solver_gmres_S.solve(negative_S_tilde, yp, tmp_p, preconditioner_S);

            dst.block(1) = yp;

            // Step 3 correction_u ~= F^{-1}(-B^T y_p).
            // Correction RHS in velocity space: stored B_T is -B^T.
            B_T->vmult(tmp_u, dst.block(1));

            SolverControl solver_F2(maxiter,
                                    std::max(1e-14,
                                             relative_tolerance *
                                               tmp_u.l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres2(solver_F2);
            
            solver_gmres2.solve(*F, correction_u, tmp_u, preconditioner_F);

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
};

#endif
