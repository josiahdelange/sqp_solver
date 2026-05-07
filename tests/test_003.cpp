#include <iostream>
#include <Eigen/Dense>
#include "sqp.h"

using namespace sqp;

class TownsendModifiedFunction: public NonlinearProblem
{
public:
    TownsendModifiedFunction()
    {
        num_var = 2;
        num_constr = 3;
    }

    void objective(const Eigen::VectorXd& decision, double& obj) override
    {
        obj = -1*std::pow(std::cos((decision[0] - 0.1)*decision[1]), 2) -
            decision[0]*std::sin(3*decision[0] + decision[1]);
    }

    void constraint(const Eigen::VectorXd& decision, Eigen::VectorXd& constraints,
        Eigen::VectorXd& lower_bnd, Eigen::VectorXd& upper_bnd) override
    {
        constraints[0] = decision[0];
        constraints[1] = decision[1];
        constraints[2] = std::pow(decision[0], 2) + std::pow(decision[1], 2);

        lower_bnd[0] = -2.25;
        lower_bnd[1] = -2.5;
        lower_bnd[2] = 0.0;

        upper_bnd[0] = 2.25;
        upper_bnd[1] = 1.75;

        double t = std::atan2(decision[0], decision[1]);
        upper_bnd[2] = std::pow(2*std::cos(t) - (1.0/2.0)*std::cos(2*t) -
            (1.0/4.0)*std::cos(3*t) - (1.0/8.0)*std::cos(4*t), 2) +
            std::pow(2*std::sin(t), 2);
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    TownsendModifiedFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    x0[0] = 1.5;
    x0[1] = 1.0;
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
