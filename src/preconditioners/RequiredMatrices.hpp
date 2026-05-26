#ifndef REQUIRED_MATRICES_HPP
#define REQUIRED_MATRICES_HPP

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

struct RequiredMatrices
{
    // F block matrix in [F -B^T; B 0] ( + Temam + Grad-div + SUPG if included) 
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr;

    // M_p = (psi_j, psi_i) 
    const TrilinosWrappers::SparseMatrix *pressure_mass      = nullptr;

    // M_u = (phi_j, phi_i)
    const TrilinosWrappers::SparseMatrix *velocity_mass      = nullptr;

    // B = (q_i, div(phi_j)) 
    const TrilinosWrappers::SparseMatrix *B                  = nullptr;

    // -B^T = -(p, div(phi_j))
    const TrilinosWrappers::SparseMatrix *BT                 = nullptr;

    // Vector layout template for auxiliary distributed vectors
    const TrilinosWrappers::MPI::BlockVector *solution_template = nullptr;

    // SIMPLE pressure correction relaxation:
    // z_p <- alpha z_p, with 0 <= alpha <= 1.
    double simple_pressure_relaxation = 0.7;

    
    // --- Pressure-space operators for PCD: ---

    // F_p = scalar pressure convection-diffusion operator built with beta.
    const TrilinosWrappers::SparseMatrix *pressure_convection_diffusion = nullptr;

    // Continuous A_p = (grad psi_j, grad psi_i), used as a PCD fallback.
    const TrilinosWrappers::SparseMatrix *pressure_laplacian = nullptr;

    // PCD pressure Laplacian A_p^disc ~= B diag(M_u)^(-1) B^T.
    const TrilinosWrappers::SparseMatrix *pressure_laplacian_discrete = nullptr;
};

#endif
