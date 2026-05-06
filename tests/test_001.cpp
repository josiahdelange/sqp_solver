#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class ConstrainedRosenbrock2D: public NonlinearProblem
{
public:
    using Vector = sqp::NonlinearProblem::Vector;
    using Matrix = sqp::NonlinearProblem::Matrix;

    ConstrainedRosenbrock2D()
    {
        num_var = 2;
        num_constr = 3;
    }

    void objective(const Vector& decision, double& obj) override
    {
        obj = std::pow(1 - decision[0], 2) +
            100*std::pow(decision[1] - std::pow(decision[0], 2), 2);
    }

    void constraint(const Vector& decision, Vector& constraints,
        Vector& lower_bnd, Vector& upper_bnd) override
    {
        constraints[0] = decision[0];
        constraints[1] = decision[1];
        constraints[2] = std::pow(decision[0], 2) + std::pow(decision[1], 2);

        lower_bnd[0] = -1.5;
        lower_bnd[1] = -1.5;
        lower_bnd[2] = 0.0;

        upper_bnd[0] = 1.5;
        upper_bnd[1] = 1.5;
        upper_bnd[2] = 2.0;
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    ConstrainedRosenbrock2D problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(3);

    // SQP solver initialization
    sqp::SQP solver;
    solver.settings().max_iter = 100;
    solver.settings().verbose = true;
    //solver.settings().line_search_max_iter = 10;
    //solver.settings().second_order_correction = true;
    //solver.settings().eta = 0.5;

    // Run SQP solver
    solver.solve(problem, x0, lambda0);

    // Print solution
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}
