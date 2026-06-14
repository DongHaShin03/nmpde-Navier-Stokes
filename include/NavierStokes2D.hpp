#ifndef NAVIERSTOKES2D_HPP
#define NAVIERSTOKES2D_HPP

#include "NavierStokes.hpp"

class NavierStokes2D : public NavierStokes<2>
{
    public:
        static constexpr unsigned int dim = 2;

        using NavierStokes<2>::NavierStokes;

        void set_force_coefficient_parameters(const double reference_velocity,
                                              const double reference_length,
                                              const types::boundary_id cylinder_boundary_id_);

    protected:
        void compute_forces() override;
        std::string simulation_name() const override;
        std::string output_folder() const override;

    private:
        double force_coefficient_reference_velocity = 0.0;
        double force_coefficient_reference_length = 0.0;
        types::boundary_id cylinder_boundary_id = static_cast<types::boundary_id>(-1);
};

#endif

