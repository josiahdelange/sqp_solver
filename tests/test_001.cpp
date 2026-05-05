#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "sqp.hpp"

using namespace sqp;

class ConstrainedRosenbrock2D: public NonLinearProblem<double>
{
public:
    using Vector = sqp::NonLinearProblem<double>::Vector;
    using Matrix = sqp::NonLinearProblem<double>::Matrix;

    ConstrainedRosenbrock2D()
    {
        num_var = 2;
        num_constr = 2;
    }

    void objective(const Vector& decision, Scalar& obj) override
    {
        obj = 0;
        for(int i = 0; i < decision.rows() - 1; i++)
        {
            obj += pow(1 - decision[i], 2) + 100*pow(decision[i+1] - pow(decision[i], 2), 2);
        }
    }

    void constraint(
        const Vector& decision, Vector& constraints, Vector& lower_bnd, Vector& upper_bnd) override
    {
        const double infinity = std::numeric_limits<double>::infinity();
        constraints << decision(0) - decision(1), decision.squaredNorm();
        lower_bnd << -infinity, 1;
        upper_bnd << 0, 1;
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    ConstrainedRosenbrock2D problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd y0 = Eigen::VectorXd::Zero(2);

    // SQP solver initialization
    sqp::SQP<double> solver;
    solver.settings().max_iter = 100;
    //solver.settings().line_search_max_iter = 10;
    //solver.settings().second_order_correction = true;
    // solver.settings().eta = 0.5;

    // Run SQP solver
    solver.solve(problem, x0, y0);

    // Print solution
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}
