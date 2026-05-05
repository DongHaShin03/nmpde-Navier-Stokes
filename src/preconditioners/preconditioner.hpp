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
    initialize(const TrilinosWrappers::SparseMatrix &F_, //VELOCITY BLOCK
               const TrilinosWrappers::SparseMatrix &B_, //divergence block, stored as -B_incomp
               const TrilinosWrappers::SparseMatrix &B_t_, //gradient block
               const TrilinosWrappers::MPI::BlockVector &sol_owned 
            )
    {
      F = &F_; 
      B = &B_;
      B_T = &B_t_;

      neg_diag_D_inv.reinit(sol_owned.block(0)); //(-D)^-1
      diag_D_inv.reinit(sol_owned.block(0));  //D^-1
      

      //construct diagDinv and neg:
      for(unsigned int i : diag_D_inv.locally_owned_elements()) //the index set owned by the current processor
      {
        const double d = F->diag_element(i);
        diag_D_inv[i] = 1.0 / d;
        neg_diag_D_inv[i] = -1.0/d;
      }

      // Stilde = B_incomp*(D^-1)*Bt approximated schur component POS DEFINITE
      B->mmult(S_tilde, *B_T, neg_diag_D_inv); //mmult mult matrix matrix
      //S tilde = B_stored *diag(neg_diag_D_inv) * BT = (-B_incomp)*diag(-D^-1)*BT =  B_incomp *D^-1 *BT 

      //preconditioners initialization
      preconditioner_F.initialize(*F);
      preconditioner_S.initialize(S_tilde);
    }

    // Application of the preconditioner. dst = P^-1 src
    //vmult = matrix vector mult
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst, 
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      //SOLVE F utilde = f_src
      TrilinosWrappers::MPI::Vector y1_u;
      y1_u.reinit(src.block(0)); //initial guess = 0

      {
        SolverControl solver_F(10000 /*maxiter*/, /*tol**/ 1e-2*src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);
        solver_gmres.solve(*F, y1_u, src.block(0), preconditioner_F);
      }

      
      //SOLVE Stilde ptilde = -(g_src + B_stored * utilde)
      //full lower-triangular equation
      // (-B_incomp) *utilde + (-Stilde) * ptilde = g_src
      
      TrilinosWrappers::MPI::Vector rhs_p;
      rhs_p.reinit(src.block(1));
      B->vmult(rhs_p, y1_u); //rhs_p = B_stored * y1_u (è = utilde)
      rhs_p -= src.block(1); //rhs_p = B_stored * utilde -g_src

      TrilinosWrappers::MPI::Vector y1_p;
      y1_p.reinit(src.block(1)); //zero initial guess

      //solve S_tilde * y1_p = rhs_p
      {
        SolverControl solver_S(10000, 1e-2*rhs_p.l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);
        solver_cg.solve(S_tilde, y1_p, rhs_p, preconditioner_S);
      }

      
      //SOLVE the correction system. upper triangular
      dst.block(1) = y1_p; //p_tilde
      dst.block(1) *= 1. / alpha; //introducting relaxation parameter
                                  //p_out = p_tilde /alpha

      //velocity correction
      //dst(0) = y1_u -inv(D)Bt dst(1)
      //dst.block(0) = y1_u; //u^n+1 =u_tilde
      TrilinosWrappers::MPI::Vector correction; //to have same dim as y1_u, initialization
      correction.reinit(src.block(0));
      B_T->vmult(correction, dst.block(1)); //tmp = Bt*p^n+1 (=p_out)
      correction.scale(diag_D_inv); //correction = D^-1 BT p_out

      dst.block(0) = y1_u; 
      dst.block(0) -= correction;  //u^n+1= u_tilde -tmp

    }

  protected:
    const double alpha = 0.5; //standard is 1.0
    // block matrix
    const TrilinosWrappers::SparseMatrix *F = nullptr ;
    const TrilinosWrappers::SparseMatrix *B_T = nullptr;
    const TrilinosWrappers::SparseMatrix *B = nullptr;

    TrilinosWrappers::SparseMatrix S_tilde;
    TrilinosWrappers::MPI::Vector diag_D_inv;
    TrilinosWrappers::MPI::Vector neg_diag_D_inv;

    //precondition block
    TrilinosWrappers::PreconditionILU preconditioner_F;
    TrilinosWrappers::PreconditionILU preconditioner_S;

};

class PreconditionerAMG_SIMPLE{

    public:
    // Initialize the preconditioner,
    void
    initialize(const TrilinosWrappers::SparseMatrix &F_, //VELOCITY BLOCK
               const TrilinosWrappers::SparseMatrix &B_, //divergence block, stored as -B_incomp
               const TrilinosWrappers::SparseMatrix &B_t_, //gradient block
               const TrilinosWrappers::MPI::BlockVector &sol_owned 
            )
    {
      F = &F_; 
      B = &B_;
      B_T = &B_t_;

      neg_diag_D_inv.reinit(sol_owned.block(0)); //(-D)^-1
      diag_D_inv.reinit(sol_owned.block(0));  //D^-1
      

      //construct diagDinv and neg:
      for(unsigned int i : diag_D_inv.locally_owned_elements()) //the index set owned by the current processor
      {
        const double d = F->diag_element(i);
        diag_D_inv[i] = 1.0 / d;
        neg_diag_D_inv[i] = -1.0/d;
      }

      // Stilde = B_incomp*(D^-1)*Bt approximated schur component POS DEFINITE
      B->mmult(S_tilde, *B_T, neg_diag_D_inv); //mmult mult matrix matrix
      //S tilde = B_stored *diag(neg_diag_D_inv) * BT = (-B_incomp)*diag(-D^-1)*BT =  B_incomp *D^-1 *BT 

      //preconditioners initialization
      //AMG for F far superior to ILU for large convection-dominated systems.
        // ILU fill-in becomes useless as system size and Re grow; AMG does not.
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_F;
        amg_data_F.elliptic              = false; // F is NOT symmetric/elliptic (convection)
        amg_data_F.higher_order_elements = true;  // P2 velocity elements
        amg_data_F.n_cycles              = 1;
        amg_data_F.w_cycle               = false;
        amg_data_F.aggregation_threshold = 1e-4;
        preconditioner_F.initialize(*F, amg_data_F);
      
      preconditioner_S.initialize(S_tilde);
    }

    // Application of the preconditioner. dst = P^-1 src
    //vmult = matrix vector mult
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst, 
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      //SOLVE F utilde = f_src
      TrilinosWrappers::MPI::Vector y1_u;
      y1_u.reinit(src.block(0)); //initial guess = 0

      {//FIX solver control throws so use iterationNumb...
        SolverControl solver_F(10000 /*maxiter*/, /*tol**/ 1e-2*src.block(0).l2_norm());
        //IterationNumberControl solver_F(30, 1e-16);
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);
        solver_gmres.solve(*F, y1_u, src.block(0), preconditioner_F);
      }

      
      //SOLVE Stilde ptilde = -(g_src + B_stored * utilde)
      //full lower-triangular equation
      // (-B_incomp) *utilde + (-Stilde) * ptilde = g_src
      
      TrilinosWrappers::MPI::Vector rhs_p;
      rhs_p.reinit(src.block(1));
      B->vmult(rhs_p, y1_u); //rhs_p = B_stored * y1_u (è = utilde)
      rhs_p -= src.block(1); //rhs_p = B_stored * utilde -g_src

      TrilinosWrappers::MPI::Vector y1_p;
      y1_p.reinit(src.block(1)); //zero initial guess

      //solve S_tilde * y1_p = rhs_p
      {
        SolverControl solver_S(10000, 1e-2*rhs_p.l2_norm());
        //IterationNumberControl solver_S(30, 1e-16);
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);
        solver_cg.solve(S_tilde, y1_p, rhs_p, preconditioner_S);
      }

      
      //SOLVE the correction system. upper triangular
      dst.block(1) = y1_p; //p_tilde
      dst.block(1) *= 1. / alpha; //introducting relaxation parameter
                                  //p_out = p_tilde /alpha

      //velocity correction
      //dst(0) = y1_u -inv(D)Bt dst(1)
      //dst.block(0) = y1_u; //u^n+1 =u_tilde
      TrilinosWrappers::MPI::Vector correction; //to have same dim as y1_u, initialization
      correction.reinit(src.block(0));
      B_T->vmult(correction, dst.block(1)); //tmp = Bt*p^n+1 (=p_out)
      correction.scale(diag_D_inv); //correction = D^-1 BT p_out

      dst.block(0) = y1_u; 
      dst.block(0) -= correction;  //u^n+1= u_tilde -tmp

    }

  protected:
    const double alpha = 1.0; //standard is 1.0
    // block matrix
    const TrilinosWrappers::SparseMatrix *F = nullptr ;
    const TrilinosWrappers::SparseMatrix *B_T = nullptr;
    const TrilinosWrappers::SparseMatrix *B = nullptr;

    TrilinosWrappers::SparseMatrix S_tilde;
    TrilinosWrappers::MPI::Vector diag_D_inv;
    TrilinosWrappers::MPI::Vector neg_diag_D_inv;

    //precondition block
    TrilinosWrappers::PreconditionAMG preconditioner_F; //AMG: scales well for convection
    TrilinosWrappers::PreconditionILU preconditioner_S;

};

//TODO Yosida
class preconditionerYosida{
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
               const TrilinosWrappers::SparseMatrix &B_,
               const TrilinosWrappers::SparseMatrix &B_t,
               const TrilinosWrappers::SparseMatrix &M_,
               const TrilinosWrappers::MPI::BlockVector &sol_owned)
    {
      F = &F_;
      B = &B_;
      B_T = &B_t;
      M = &M_;
      
      diag_D_inv.reinit(sol_owned.block(0));
      neg_diag_D_inv.reinit(sol_owned.block(0));
      for (unsigned int i : diag_D_inv.locally_owned_elements())
      {
        //Note : we have assembled M as M/deltat
        diag_D_inv[i] = ( 1.0 / M->diag_element(i));  //  dt * (Mii)^-1
        neg_diag_D_inv[i] = ( -1.0 / M->diag_element(i));  //  dt * (Mii)^-1
      }

      // Create negative_S_tilde
      B->mmult(negative_S_tilde, *B_T, neg_diag_D_inv);
    
      // Initialize the preconditioners
      preconditioner_F.initialize(*F);
      preconditioner_S.initialize(negative_S_tilde);
    }
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const 
    {
      const unsigned int maxiter = 100000;
      const double tol = 1e-2;

      SolverControl solver_F(maxiter, tol * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);

      // Store in temporaries the results
      TrilinosWrappers::MPI::Vector yu = src.block(0);
      TrilinosWrappers::MPI::Vector yp = src.block(1);
      TrilinosWrappers::MPI::Vector tmp = src.block(1);
      TrilinosWrappers::MPI::Vector tmp2 = src.block(0);

      //Step 1
      //yu = F^-1 * src.0
      solver_gmres.solve(*F, yu, src.block(0), preconditioner_F);
      
      //Step 1.2) yp = negative_S_tilde^-1(src1-B*yu)
      B->vmult(tmp, yu); //tmp = B*yu
      tmp.add(-1.0, src.block(1)); // tmp = src.block(1) - tmp
      // neg_S*yp = (src(1) - Byu)==tmp(RHS)
      SolverControl solver_S(maxiter, tol * tmp.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);
      solver_cg.solve(negative_S_tilde, yp, tmp, preconditioner_S);

      //Step 2) 
      // dst1 = yp
      dst.block(1) = yp; 

      //dst0 = yu - F^-1*B_T*yp 
      B_T->vmult(tmp2, dst.block(1)); //tmp2 = B_T*yp (rhs)

      //Solve the linear system 
      res.reinit(src.block(0)); //to store the result of the  lin sys F res = tmp2
      dst.block(0) = yu; //init final velocity dest 
      SolverControl solver_F2(maxiter, tol * tmp2.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres2(solver_F2);
      solver_gmres2.solve(*F, res, tmp2, preconditioner_F); // res = F^-1 * tmp2 
      dst.block(0).sadd(-1,res); //update final velocity dest dstu = yu - res

    }
  protected:

    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *B_T;
    const TrilinosWrappers::SparseMatrix *B;
    const TrilinosWrappers::SparseMatrix *M;
    TrilinosWrappers::SparseMatrix negative_S_tilde;
    TrilinosWrappers::MPI::Vector diag_D_inv;
    TrilinosWrappers::MPI::Vector neg_diag_D_inv;
    TrilinosWrappers::PreconditionILU preconditioner_F;
    TrilinosWrappers::PreconditionILU preconditioner_S;

    mutable TrilinosWrappers::MPI::Vector res;

};
//TODO aSimple
//TODO PCD
//TODO S= pressure matrix o laplacian pressure
#endif