#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class BoothFunction: public NonlinearProblem
{
public:
    BoothFunction()
    {
        num_var = 2;
        num_constr = 2;
    }

    void objective(const Eigen::VectorXd& decision, double& obj) override
    {
        obj = std::pow(decision[0]+ 2*decision[1] - 7, 2) +
            std::pow(2*decision[0]+ decision[1] - 5, 2);
    }

    void constraint(const Eigen::VectorXd& decision, Eigen::VectorXd& constraints,
        Eigen::VectorXd& lower_bnd, Eigen::VectorXd& upper_bnd) override
    {
        constraints[0] = decision[0];
        constraints[1] = decision[1];

        lower_bnd[0] = -10.0;
        lower_bnd[1] = -10.0;

        upper_bnd[0] = 10.0;
        upper_bnd[1] = 10.0;
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    BoothFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    x0[0] = -5.0;
    x0[1] = -5.0;
    Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(2);

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
        test_case_passed = (std::fabs(soln[0] - (1.0)) - sol_tol)
            && (std::fabs(soln[1] - (3.0)) - sol_tol);
    } else {
        std::cout << "no solution" << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;

    return (test_case_passed ? 0 : -1);
}
