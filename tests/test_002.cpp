#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class MishraBirdFunction: public NonlinearProblem
{
public:
    MishraBirdFunction()
    {
        num_var = 2;
        num_constr = 3;
    }

    void objective(const Eigen::VectorXd& decision, double& obj) override
    {
        double x = decision[0];
        double y = decision[1];

        obj = std::sin(y)*std::exp(std::pow(1 - std::cos(x), 2)) +
            std::cos(x)*std::exp(std::pow(1 - std::sin(y), 2)) +
            std::pow(x - y, 2);
    }

    void constraint(const Eigen::VectorXd& decision, Eigen::VectorXd& constraints,
        Eigen::VectorXd& lower_bnd, Eigen::VectorXd& upper_bnd) override
    {
        constraints[0] = decision[0];
        constraints[1] = decision[1];
        constraints[2] = std::pow(decision[0] + 5, 2) + std::pow(decision[1] + 5, 2);

        lower_bnd[0] = -10.0;
        lower_bnd[1] = -6.5;
        lower_bnd[2] = 0.0;

        upper_bnd[0] = 0.0;
        upper_bnd[1] = 0.0;
        upper_bnd[2] = 25.0;
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    MishraBirdFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(3);

    // SQP solver initialization
    sqp::SQP solver;
    solver.settings().max_iter = 100;
    solver.settings().verbose = true;
    solver.settings().tau = 0.5; // line search iteration decrease
    solver.settings().eta = 0.25; // line search parameter
    solver.settings().rho = 1e-2; // line search parameter
    solver.settings().eps_prim = 1e-9; // primal step termination threshold
    solver.settings().eps_dual = 1e-9; // dual step termination threshold
    solver.settings().eps_grad = 1e-12; // gradient finite difference step size

    // Run SQP solver
    solver.solve(problem, x0, lambda0);

    // Print solution (if found)
    bool test_case_passed = false;
    double sol_tol = 1e-3; // generous
    std::cout << "---------------------------------------------------" << std::endl;
    if(solver.info().status == SOLVED) {
        std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
        std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
        Eigen::VectorXd soln = solver.primal_solution();
        test_case_passed = (std::fabs(soln[0] - (-3.1302468)) - sol_tol)
            && (std::fabs(soln[1] - (-1.5821422)) - sol_tol);
    } else {
        std::cout << "no solution" << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;

    return (test_case_passed ? 0 : -1);
}
