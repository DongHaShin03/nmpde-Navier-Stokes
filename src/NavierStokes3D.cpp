#include "NavierStokes3D.hpp"

void NavierStokes3D::compute_forces()
{
    // ----- QUI METRICHE 3D -----
    // Implementare drag/lift/side-force e, se serve, Delta p 3D usando la stessa
    // convenzione del caso 2D. Dopo l'implementazione, il caso Re 20 3D deve
    // produrre un CSV compatibile con quello 2D.
    // #TODO
}

std::string NavierStokes3D::simulation_name() const
{
    return "Navier-Stokes 3D Simulation";
}

std::string NavierStokes3D::output_folder() const
{
    return "results";
}

