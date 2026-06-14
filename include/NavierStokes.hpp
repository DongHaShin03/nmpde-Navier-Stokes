#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

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

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "benchmarks/BenchmarkRecorder.hpp"
#include "NavierStokesOptions.hpp"

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace dealii;

template <int dim>
class NavierStokes
{
    public:
        static constexpr unsigned int dimension = dim;
        NavierStokes(
          const std::string  &mesh_file_name_,
          const unsigned int &degree_velocity_,
          const unsigned int &degree_pressure_,
          const double       &nu_,
          const std::function<Tensor<1, dim>(const Point<dim> &, const double &)> &f_,
          const double       &T_,
          const double       &theta_,
          const double       &delta_t_);

        virtual ~NavierStokes() = default;

        void run();
        void set_nonlinear_solver_parameters(const unsigned int max_iterations,
                                             const double       tolerance);
        void set_nonlinear_solver_strategy(const NonlinearMethod method,
                                           const double          relaxation);
        void set_linear_solver_parameters(const unsigned int gmres_restart_length_,
                                          const double       pressure_regularization_,
                                          const unsigned int linear_max_iterations_,
                                          const double       linear_relative_tolerance_,
                                          const double       linear_absolute_tolerance_);
        void set_preconditioner(const PreconditionerKind preconditioner_kind_);
        void set_simple_pressure_relaxation(const double relaxation);
        void set_preconditioner_iterations(const PreconditionerIterationOptions &options);
        void set_stabilization_options(const StabilizationOptions &options);
        void set_run_config_file(const std::string &parameter_file);
        void set_output_options(const std::string &output_directory, const std::string &run_id, const std::string &benchmark_id, const std::string &mesh_name);
        void set_write_solution_output(const bool enabled);
        void set_benchmark_statistics_start_time(const double start_time);
        void set_run_config_values(const std::map<std::string, std::string> &config_values);

        std::unique_ptr<Function<dim>> initial_condition;
        std::map<types::boundary_id, const Function<dim> *> dirichlet;
        std::map<types::boundary_id, const Function<dim> *> neumann;

    protected:
        virtual void compute_forces() = 0;
        virtual std::string simulation_name() const = 0;
        virtual std::string output_folder() const = 0;

        void setup();
        void assemble_static();
        void assemble_timestep(const TrilinosWrappers::MPI::BlockVector &beta_solution);
        void solve(TimerOutput &timer);
        void output();
        double compute_pressure_difference(const Point<dim> &front_point, const Point<dim> &back_point) const;
        void write_benchmark_metrics(const double drag_coefficient, const double lift_coefficient, const double side_coefficient, const double delta_pressure, const double reference_velocity, const double reference_length);
        std::string benchmark_output_directory() const;

        const std::string mesh_file_name;
        const unsigned int degree_velocity;
        const unsigned int degree_pressure;
        const double nu;
        const std::function<Tensor<1, dim>(const Point<dim> &, const double &)> f;

        const double T;
        const double delta_t;
        const double theta;
        unsigned int timestep_number = 0;
        double time = 0.0;

        // MPI / MESH
        const unsigned int mpi_size;
        const unsigned int mpi_rank;
        parallel::fullydistributed::Triangulation<dim> mesh;

        //FE
        std::unique_ptr<FiniteElement<dim>> fe;
        std::unique_ptr<Quadrature<dim>> quadrature;
        std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

        DoFHandler<dim> dof_handler;
        IndexSet locally_owned_dofs;
        std::vector<IndexSet> block_owned_dofs;
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> block_relevant_dofs;

        // Matrices for the monolithic Oseen/Navier-Stokes problem.
        //
        // system_matrix has the saddle-point block form
        //     [ F  -B^T ]
        //     [ B   0   ]
        // where F contains velocity mass, diffusion, convection and velocity
        // stabilizations. The other matrices below are auxiliary operators
        // used only by preconditioners that explicitly request them.
        TrilinosWrappers::BlockSparseMatrix static_matrix;
        TrilinosWrappers::BlockSparseMatrix convection_matrix;
        TrilinosWrappers::BlockSparseMatrix system_matrix;

        // M_u = (phi_j, phi_i)
        TrilinosWrappers::BlockSparseMatrix velocity_mass;

        // M_p = (psi_j, psi_i)
        TrilinosWrappers::BlockSparseMatrix pressure_mass;

        // A_p^disc ~= B diag(M_u)^{-1} B^T
        TrilinosWrappers::BlockSparseMatrix pressure_laplacian_discrete;

        // F_p = (1/dt)M_p + theta*nu*A_p + theta*C_p(beta)
        TrilinosWrappers::BlockSparseMatrix pressure_convection_diffusion;

        TrilinosWrappers::MPI::BlockVector system_rhs;

        TrilinosWrappers::MPI::BlockVector solution_owned;
        TrilinosWrappers::MPI::BlockVector solution;
        TrilinosWrappers::MPI::BlockVector old_solution;

        std::vector<std::pair<double, std::string>> times_and_names;

        types::global_dof_index n_velocity_dofs = 0;
        types::global_dof_index n_pressure_dofs = 0;
        types::global_dof_index n_total_dofs = 0;

        unsigned int gmres_restart_length = 200;
        double pressure_regularization = 1e-8;
        unsigned int linear_max_iterations = 100000;
        double linear_relative_tolerance = 1e-2;
        double linear_absolute_tolerance = 1e-12;

        //unused
        unsigned int nonlinear_max_iterations = 4;
        double nonlinear_tolerance = 1e-6;
        NonlinearMethod nonlinear_method = NonlinearMethod::Oseen;
        double picard_relaxation = 1.0;

        PreconditionerKind preconditioner_kind = PreconditionerKind::Simple;
        double simple_pressure_relaxation = 0.7;
        PreconditionerIterationOptions preconditioner_iterations;
        StabilizationOptions stabilization_options;

        ConditionalOStream pcout;
    private:
        BenchmarkRunMetadata build_benchmark_metadata() const;
        BenchmarkFieldMetrics compute_field_metrics() const;
        void reset_current_step_metrics();
        double compute_supg_tau(const double beta_norm,const double h_K) const;
        bool needs_velocity_mass_matrix() const;
        bool needs_pressure_mass_matrix() const;
        bool needs_pcd_pressure_operators() const;

        bool static_matrix_built = false;
        bool write_solution_output = true;

        BenchmarkOutputOptions benchmark_output_options;
        BenchmarkRecorder benchmark_recorder;
        Timer run_timer;
        Timer step_timer;

        unsigned int current_step_linear_solves = 0;
        unsigned int current_step_gmres_iterations = 0;
        unsigned int current_step_preconditioner_inner_solves = 0;
        unsigned int current_step_preconditioner_inner_iterations = 0;
        unsigned int current_step_preconditioner_inner_failures = 0;
        unsigned int current_step_nonlinear_iterations = 0;
        double current_step_gmres_final_residual = std::numeric_limits<double>::quiet_NaN();
        bool current_step_gmres_converged = true;
        double current_step_assembly_time = 0.0;
        double current_step_linear_solve_time = 0.0;
        double current_step_preconditioner_setup_time = 0.0;
};

extern template class NavierStokes<2>;
extern template class NavierStokes<3>;

#endif
