#ifndef PRECONDITIONERS_HPP
#define PRECONDITIONERS_HPP

#include <fstream>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>


using namespace dealii;

/*
block triangular precond        S = BD^-1Bt, D diagonal of F, alpha€(0,1]
[F  0] y1_u] = [src_u]        where A = (F Bt; -B 0) Ax = b con x=(u_n+1; p_n+1) b = (G 0)
[B -S][y1_p] = [src_p]

[I D^-1Bt] [dst_u] =  y1_u]
[0 alphaI] [dst_p] = [sol_p]
*/
class PreconditionerSIMPLE{

    public:
    // Initialize the preconditioner,
    void
    initialize(const TrilinosWrappers::SparseMatrix &F_,
               const TrilinosWrappers::SparseMatrix &B_,
               const TrilinosWrappers::SparseMatrix &B_t,
               const TrilinosWrappers::MPI::BlockVector &sol_owned 
            )
    {
      F = &F_; 
      B = &B_;
      B_T = &B_t;

      neg_diag_D_inv.reinit(sol_owned.block(0)); //(-D)^-1
      diag_D_inv.reinit(sol_owned.block(0));  //D^-1

      //construct diagDinv and neg:
      for(unsigned int i : diag_D_inv.locally_owned_elements()) //the index set owned by the current processor
      {
        double temp = F->diag_element(i);
        diag_D_inv[i] = 1.0 / temp;
        neg_diag_D_inv[i] = -1.0/temp;
      }

      //- Stilde = B*(- D^-1)*Bt approximated schur component
      B->mmult(neg_S_tilde, *B_T, neg_diag_D_inv); //mmult mult matrix matrix

      //preconditioners initialization
      preconditioner_F.initialize(*F);
      preconditioner_S.initialize(neg_S_tilde);
    }

    // Application of the preconditioner.
    //vmult = matrix vector mult
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst, 
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl solver_F(10000 /*maxiter*/, /*tol**/ 1e-2*src.block(0).l2_norm());

      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);

      //block lower triangular system

      //store temporary results:
      TrilinosWrappers::MPI::Vector y1_u = src.block(0); //u tilde = f
      TrilinosWrappers::MPI::Vector y1_p = src.block(1); //p tilde = g,

      TrilinosWrappers::MPI::Vector temp_1 = src.block(1); //

      solver_gmres.solve(*F, y1_u, src.block(0), preconditioner_F);

      B->vmult(temp_1, y1_u); //temp1=B y1_u
      temp_1 -= src.block(1);  //temp = B y1_u - g

      //solve -S_tilde * y1_p = temp_1
      SolverControl solver_S(10000, 1e-2*temp_1.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);

      solver_cg.solve(neg_S_tilde, y1_p, temp_1, preconditioner_S);

      //solve the correction system. upper triangular
      dst.block(1) = y1_p; //p_tilde
      dst.block(1) *= 1. / alpha; //introducting relaxation parameter

      //velocity correction
      //dst(0) = y1_u -inv(D)Bt dst(1)
      dst.block(0) = y1_u; //u^n+1 =u_tilde
      TrilinosWrappers::MPI::Vector tmp = src.block(0); //to have same dim as y1_u, initialization
      B_T->vmult(tmp, dst.block(1)); //tmp = Bt*p^n+1
      tmp.scale(diag_D_inv); //tmp= inv(D)*tmp 
      dst.block(0) -= tmp; //u^n+1= u_tilde -tmp

    }

  protected:
    const double alpha = 0.5;
    // block matrix
    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *B_T;
    const TrilinosWrappers::SparseMatrix *B;

    TrilinosWrappers::SparseMatrix neg_S_tilde;
    TrilinosWrappers::MPI::Vector diag_D_inv;
    TrilinosWrappers::MPI::Vector neg_diag_D_inv;

    //precondition block
    TrilinosWrappers::PreconditionILU preconditioner_F;
    TrilinosWrappers::PreconditionILU preconditioner_S;

};

//TODO Yosida
//TODO aSimple
//TODO PCD
//TODO S= pressure matrix o laplacian pressure
#endif