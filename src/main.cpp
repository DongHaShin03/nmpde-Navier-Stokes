#include "NavierStokes2D.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/navierstokes-coarse.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  NavierStokes2D problem(mesh_file_name, degree_velocity, degree_pressure);

  problem.run();

  return 0;
} 