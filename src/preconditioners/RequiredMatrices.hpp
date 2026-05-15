#ifndef REQUIRED_MATRICES_HPP
#define REQUIRED_MATRICES_HPP

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

struct RequiredMatrices
{
    // ----- QUI MATRICI BASE PRECONDIZIONATORI -----
    // Aggiungere qui solo puntatori a matrici condivise tra precondizionatori.
    // Dopo aver aggiunto un campo, valorizzarlo in NavierStokes::solve() e
    // usarlo dal precondizionatore senza cambiare la firma initialize(...).
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr; 
    const TrilinosWrappers::SparseMatrix *pressure_mass      = nullptr; 
    const TrilinosWrappers::SparseMatrix *velocity_mass      = nullptr; 
    const TrilinosWrappers::SparseMatrix *B                  = nullptr; 
    const TrilinosWrappers::SparseMatrix *BT                 = nullptr;
    const TrilinosWrappers::MPI::BlockVector *solution_template = nullptr;

    // ----- QUI PCD -----
    // PCD richiede M_p, K_p e F_p coerenti con le boundary conditions di
    // pressione. Dopo l'implementazione, controllare che PCD sia selezionabile
    // da .prm e confrontarlo con SIMPLE/Yosida su Re 100 e Re 200.
    // PCD
    const TrilinosWrappers::SparseMatrix *pressure_convection_diffusion = nullptr; 
    const TrilinosWrappers::SparseMatrix *pressure_laplacian = nullptr; 
};

#endif
