#ifndef NAVIERSTOKES3D_HPP
#define NAVIERSTOKES3D_HPP

#include "NavierStokes.hpp"

class NavierStokes3D : public NavierStokes<3>
{
    public:
        static constexpr unsigned int dim = 3;

        using NavierStokes<3>::NavierStokes;

    protected:
        void compute_forces() override;
        std::string simulation_name() const override;
        std::string output_folder() const override;
};

#endif

