#ifndef NAVIER_STOKES_PRECONDITIONER_HPP
#define NAVIER_STOKES_PRECONDITIONER_HPP

#include "RequiredMatrices.hpp"
#include <deal.II/base/timer.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <string>

struct AssemblyFlags
{
    // Request assembly of M_u = (phi_j, phi_i).
    bool Mu = false;
    // Request assembly of M_p = (psi_j, psi_i).
    bool Mp = false;
};

struct PreconditionerSolveStatistics
{
    unsigned int solves = 0;
    unsigned int iterations = 0;
    unsigned int failures = 0;
};

class NavierStokesPreconditioner
{
    public:
        virtual ~NavierStokesPreconditioner() = default;

        virtual AssemblyFlags get_needed_matrices() const {return AssemblyFlags();}

        void set_timer(dealii::TimerOutput *timer_)
        {
            timer = timer_;
        }

        void reset_statistics() const
        {
            solve_statistics = {};
        }

        PreconditionerSolveStatistics statistics() const
        {
            return solve_statistics;
        }

        virtual void initialize(const RequiredMatrices &a) = 0;
        virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const = 0;

    protected:
        template <typename Callable>
        void time_section(const std::string &section, Callable &&callable) const
        {
            if (timer != nullptr)
            {
                dealii::TimerOutput::Scope timer_section(*timer, section);
                callable();
            }
            else
                callable();
        }

        void record_inner_solve(const unsigned int iterations, const bool converged) const
        {
            ++solve_statistics.solves;
            solve_statistics.iterations += iterations;
            if (!converged)
                ++solve_statistics.failures;
        }

        dealii::TimerOutput *timer = nullptr;
        mutable PreconditionerSolveStatistics solve_statistics;
};

#endif
