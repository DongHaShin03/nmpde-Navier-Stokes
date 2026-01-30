#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "preconditioners/NavierStokesPreconditioner.hpp"

#include <fstream>
#include <iostream>

using namespace dealii;

class NavierStokes
{
    public: 
        static constexpr unsigned int dim = 2;
        using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p, const double &)>; 

        struct NeededMatrices
        {
            bool M_p = false;
            bool M_u = false; 
        };

        class PreconditionIdentity
        {
            public:
                void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const { dst = src; }
        };

        // Block-diagonal preconditioner.
        class PreconditionBlockDiagonal
        {
            public:
                // Initialize the preconditioner, given the velocity stiffness matrix and the pressure mass matrix.
                void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_, const TrilinosWrappers::SparseMatrix &pressure_mass_)
                {
                    velocity_stiffness = &velocity_stiffness_;
                    pressure_mass      = &pressure_mass_;

                    preconditioner_velocity.initialize(velocity_stiffness_);
                    preconditioner_pressure.initialize(pressure_mass_);
                }

                // Application of the preconditioner.
                void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const
                {
                    SolverControl solver_control_velocity(1000, 1e-2 * src.block(0).l2_norm());
                    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);
                    solver_cg_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);

                    SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
                    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
                    solver_cg_pressure.solve(*pressure_mass, dst.block(1), src.block(1), preconditioner_pressure);
                }

            protected:
                // Velocity stiffness matrix.
                const TrilinosWrappers::SparseMatrix *velocity_stiffness;

                // Preconditioner used for the velocity block.
                TrilinosWrappers::PreconditionILU preconditioner_velocity;

                // Pressure mass matrix.
                const TrilinosWrappers::SparseMatrix *pressure_mass;

                // Preconditioner used for the pressure block.
                TrilinosWrappers::PreconditionILU preconditioner_pressure;
        };

        NavierStokes
        (
            const std::string  &mesh_file_name_,
            const unsigned int &degree_velocity_,
            const unsigned int &degree_pressure_,
            const double &nu_,
            const VectorField &f_,
            const double &T_,
            const double &theta_,
            const double &delta_t_
        )
        : mesh_file_name(mesh_file_name_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , nu(nu_)
        , f(f_)
        , T(T_)
        , theta(theta_)
        , delta_t(delta_t_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , mesh(MPI_COMM_WORLD)
        , pcout(std::cout, mpi_rank == 0)
        {}

        void run();
        void set_preconditioner(std::unique_ptr<NavierStokesPreconditioner> prec) {preconditioner = std::move(prec);}

        std::unique_ptr<Function<dim>> initial_condition; 
        std::map<types::boundary_id, const Function<dim> *> dirichlet;
        std::map<types::boundary_id, const Function<dim> *> neumann;

    protected: 
        void setup(); 
        void assemble(); 
        void solve(); 
        void output(); 
        void compute_forces(); 

        const std::string mesh_file_name;
        const unsigned int degree_velocity;
        const unsigned int degree_pressure;
        const double nu; 
        const VectorField f;

        const double T;
        const double theta; 
        const double delta_t;
        unsigned int timestep_number = 0; 
        double time = 0.0;

        const unsigned int mpi_size;
        const unsigned int mpi_rank;
        parallel::fullydistributed::Triangulation<dim> mesh;

        std::unique_ptr<FiniteElement<dim>> fe;
        std::unique_ptr<Quadrature<dim>> quadrature;
        std::unique_ptr<Quadrature<dim-1>> quadrature_boundary;

        DoFHandler<dim> dof_handler;
        IndexSet locally_owned_dofs;
        std::vector<IndexSet> block_owned_dofs; // DoFs owned by current process in the velocity and pressure blocks.
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> block_relevant_dofs; // DoFs relevant to current process in the velocity and pressure blocks.

        TrilinosWrappers::BlockSparseMatrix system_matrix;
        TrilinosWrappers::MPI::BlockVector system_rhs;

        TrilinosWrappers::MPI::BlockVector solution_owned; // without ghost elements
        TrilinosWrappers::MPI::BlockVector solution;       // with ghost elements
        TrilinosWrappers::MPI::BlockVector old_solution;   // solution of previous time step

        // Preconditioner
        TrilinosWrappers::BlockSparseMatrix velocity_mass; // Mu
        TrilinosWrappers::BlockSparseMatrix pressure_mass; // Mp
        AssemblyFlags flags;

        std::unique_ptr<NavierStokesPreconditioner> preconditioner;

        ConditionalOStream pcout;
}; 

#endif