#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class GoldsteinPriceFunction: public NonlinearProblem
{
public:
    GoldsteinPriceFunction()
    {
        num_var = 2;
        num_constr = 2;
    }

    void objective(const Eigen::VectorXd& decision, double& obj) override
    {
        obj = (1 + std::pow(decision[0] + decision[1] + 1, 2)*(19 - 14*decision[0] +
            3*std::pow(decision[0],2) - 14*decision[1] + 6*decision[0]*decision[1] +
            3*std::pow(decision[1],2)))*(30 + std::pow(2*decision[0] - 3*decision[1],2)*(
                18 - 32*decision[0] + 12*std::pow(decision[0],2) + 48*decision[1] -
                36*decision[0]*decision[1] + 27*std::pow(decision[1],2)));
    }

    void constraint(const Eigen::VectorXd& decision, Eigen::VectorXd& constraints,
        Eigen::VectorXd& lower_bnd, Eigen::VectorXd& upper_bnd) override
    {
        constraints[0] = decision[0];
        constraints[1] = decision[1];

        lower_bnd[0] = -2.0;
        lower_bnd[1] = -3.0;

        upper_bnd[0] = 2.0;
        upper_bnd[1] = 1.0;
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    GoldsteinPriceFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    //x0[0] = -0.5;
    //x0[1] = -0.5;
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
    std::cout << "---------------------------------------------------" << std::endl;
    if(solver.info().status == SOLVED) {
        std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
        std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
        
    } else {
        std::cout << "no solution" << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}
