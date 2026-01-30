#include "NavierStokes.hpp"
#include "preconditioners/BlockTriangular.hpp"

using namespace std; 
static constexpr unsigned int dim = NavierStokes::dim; 

using Value = std::function<double(const Point<dim> &p)>; 
using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p)>; 

class Inlet : public Function<dim> 
{
    public: 
        Inlet() : Function<dim>(dim + 1) {}

        virtual void vector_value(const Point<dim> &, Vector<double> &values) const override
        {
            values[0] = u_0;

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }
    protected: 
        const double u_0 = 1.0; 
}; 

class Neumann : public Function<dim> 
{
    public: 
        Neumann(){}
        virtual double value(const Point<dim> &, const unsigned int = 0) const override
        {
            return -1.0; 
        }
}; 

class FunctionU0 : public Function<dim>
{
    public: 
        FunctionU0(){};

        virtual double value(const Point<dim> &, const unsigned int = 0) const override
        {
            return 0.0; 
        }
}; 

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string  mesh_file_name  = "../mesh/navierstokesFINAL.msh";
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;
    
    const double nu = 1.0; 
    const auto f = [](const Point<dim> &, const double &)
    {
        Tensor<1, dim> result; 
        result[0] = 0.0; 
        result[1] = 0.0; 
        result[2] = 0.0; 
        return result; 
    }; 
    
    const double T = 80.0; 
    const double theta = 1.0; 
    const double delta_t = 0.5; 

    NavierStokes problem(mesh_file_name, degree_velocity, degree_pressure, nu, f, T, theta, delta_t);

    auto preconditioner = std::make_unique<BlockTriangular>(); 
    problem.set_preconditioner(std::move(preconditioner)); 

    Inlet inlet_velocity; 
    Functions::ZeroFunction<dim> zero_function; 
    Neumann neumann_bc; 

    problem.dirichlet[1] = &inlet_velocity;  
    problem.dirichlet[3] = &zero_function;  
    problem.dirichlet[5] = &zero_function;  
    problem.neumann[2]   = &neumann_bc;  
    problem.initial_condition = std::make_unique<Inlet>(); 

    problem.run();

    return 0;
}