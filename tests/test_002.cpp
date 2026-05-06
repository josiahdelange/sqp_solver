#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class MishraBirdFunction: public NonLinearProblem<double>
{
public:
    using Vector = sqp::NonLinearProblem<double>::Vector;
    using Matrix = sqp::NonLinearProblem<double>::Matrix;

    MishraBirdFunction()
    {
        num_var = 2;
        num_constr = 1;
    }

    void objective(const Vector& decision, Scalar& obj) override
    {
        double x = decision[0];
        double y = decision[1];

        obj = std::sin(y)*std::exp(std::pow(1 - std::cos(x), 2)) +
            std::cos(x)*std::exp(std::pow(1 - std::sin(y), 2)) +
            std::pow(x - y, 2);
    }

    void constraint(const Vector& decision, Vector& constraints,
        Vector& lower_bnd, Vector& upper_bnd) override
    {
        Vector xy_vec = Vector::Zero(2);
        xy_vec[0] = decision[0] + 5;
        xy_vec[1] = decision[1] + 5;
        constraints[0] = 25 - xy_vec.squaredNorm();
        lower_bnd[0] = -1*std::numeric_limits<double>::infinity();
        upper_bnd[0] = std::numeric_limits<double>::infinity();
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    MishraBirdFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    x0[0] = -3.0;
    x0[1] = -2.0;
    Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(1);

    // SQP solver initialization
    sqp::SQP<double> solver;
    solver.settings().max_iter = 100;
    solver.settings().verbose = true;
    //solver.settings().line_search_max_iter = 10;
    //solver.settings().second_order_correction = true;
    // solver.settings().eta = 0.5;

    // Run SQP solver
    solver.solve(problem, x0, lambda0);

    // Print solution
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}
