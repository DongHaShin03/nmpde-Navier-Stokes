#ifndef REQUIRED_MATRICES_HPP
#define REQUIRED_MATRICES_HPP

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

struct PreconditionerIterationOptions
{
    unsigned int block_triangular_velocity_max_iterations = 100;
    unsigned int block_triangular_schur_max_iterations = 250;
    double block_triangular_velocity_relative_tolerance = 1e-2;
    double block_triangular_schur_relative_tolerance = 1e-3;
    unsigned int simple_velocity_max_iterations = 5;
    unsigned int simple_schur_max_iterations = 20;
    double simple_velocity_relative_tolerance = 1e-2;
    double simple_schur_relative_tolerance = 1e-3;
    unsigned int pcd_velocity_max_iterations = 10;
    unsigned int pcd_pressure_max_iterations = 20;
    double pcd_velocity_relative_tolerance = 1e-2;
    double pcd_pressure_relative_tolerance = 1e-3;
    unsigned int yosida_velocity_max_iterations = 100000;
    unsigned int yosida_schur_max_iterations = 100000;
    unsigned int yosida_correction_max_iterations = 100000;
    double yosida_relative_tolerance = 1e-2;
    double preconditioner_absolute_tolerance = 1e-12;
    double yosida_absolute_tolerance = 1e-14;
};

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

    PreconditionerIterationOptions preconditioner_iterations;

    
    // --- Pressure-space operators for PCD: ---

    // F_p = scalar pressure convection-diffusion operator built with beta.
    const TrilinosWrappers::SparseMatrix *pressure_convection_diffusion = nullptr;

    // PCD pressure Laplacian A_p^disc ~= B diag(M_u)^(-1) B^T.
    const TrilinosWrappers::SparseMatrix *pressure_laplacian_discrete = nullptr;
};

#endif
