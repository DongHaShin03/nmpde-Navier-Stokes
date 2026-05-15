#ifndef YOSIDA_PRECONDITIONER_HPP
#define YOSIDA_PRECONDITIONER_HPP

#include "NavierStokesPreconditioner.hpp"

#include <algorithm>
#include <stdexcept>

class Yosida : public NavierStokesPreconditioner
{
    public:
        AssemblyFlags get_needed_matrices() const override { return {true, false}; }

        void initialize(const RequiredMatrices &data) override
        {
            // ----- QUI YOSIDA: VALIDAZIONE NUMERICA -----
            // Controllare se usare velocity_mass vera oppure il blocco F come
            // fallback. Dopo l'implementazione di M_u, confrontare iterazioni
            // GMRES prima/dopo su Re 20 e Re 100.
            if (data.velocity_stiffness == nullptr || data.B == nullptr ||
                data.BT == nullptr || data.solution_template == nullptr)
                throw std::runtime_error(
                  "Yosida preconditioner requires F, B, B^T and a solution template.");

            F = data.velocity_stiffness;
            B = data.B;
            B_T = data.BT;
            M = (data.velocity_mass != nullptr ? data.velocity_mass :
                                                data.velocity_stiffness);

            diag_D_inv.reinit(data.solution_template->block(0));
            neg_diag_D_inv.reinit(data.solution_template->block(0));

            for (const auto i : diag_D_inv.locally_owned_elements())
            {
                const double d = M->diag_element(i);
                diag_D_inv[i] = 1.0 / d;
                neg_diag_D_inv[i] = -1.0 / d;
            }

            B->mmult(negative_S_tilde, *B_T, neg_diag_D_inv);

            preconditioner_F.initialize(*F);
            preconditioner_S.initialize(negative_S_tilde);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                   const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            // ----- QUI YOSIDA: TOLLERANZE INTERNE -----
            // Rendere maxiter e tolleranze dei solve interni configurabili.
            // Dopo il cambio, riportare anche le iterazioni interne se servono
            // al confronto tra precondizionatori.
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
            solver_gmres.solve(*F, yu, src.block(0), preconditioner_F);

            B->vmult(tmp_p, yu);
            tmp_p *= -1.0;
            tmp_p += src.block(1);

            SolverControl solver_S(maxiter,
                                   std::max(1e-14,
                                            relative_tolerance * tmp_p.l2_norm()));
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);
            solver_cg.solve(negative_S_tilde, yp, tmp_p, preconditioner_S);

            dst.block(1) = yp;

            B_T->vmult(tmp_u, dst.block(1));

            SolverControl solver_F2(maxiter,
                                    std::max(1e-14,
                                             relative_tolerance *
                                               tmp_u.l2_norm()));
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres2(solver_F2);
            solver_gmres2.solve(*F, correction_u, tmp_u, preconditioner_F);

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
