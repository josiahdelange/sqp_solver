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
        num_constr = 1;
    }

    void objective(const Eigen::VectorXd& decision, double& obj) override
    {
        obj = -1*std::pow(std::cos((decision[0] - 0.1)*decision[1]), 2) -
            decision[0]*std::sin(3*decision[0] + decision[1]);
    }

    void constraint(const Eigen::VectorXd& decision, Eigen::VectorXd& constraints,
        Eigen::VectorXd& lower_bnd, Eigen::VectorXd& upper_bnd) override
    {
        Eigen::VectorXd xy_vec = Eigen::VectorXd::Zero(2);
        xy_vec[0] = decision[0];
        xy_vec[1] = decision[1];
        double t = std::atan2(decision[0], decision[1]);
        double xy_limit = std::pow(2*std::cos(t) - (1.0/2.0)*std::cos(2*t) -
            (1.0/4.0)*std::cos(3*t) - (1.0/8.0)*std::cos(4*t), 2) +
            std::pow(2*std::sin(t), 2);
        constraints[0] = xy_limit - decision.squaredNorm();
        lower_bnd[0] = -1*std::numeric_limits<double>::infinity();
        upper_bnd[0] = std::numeric_limits<double>::infinity();
    }
};

int main(int argc, char* argv[])
{
    // Nonlinear problem
    TownsendModifiedFunction problem;
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);
    x0[0] = 1.5;
    x0[1] = 1.0;
    Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(1);

    // SQP solver initialization
    sqp::SQP solver;
    solver.settings().max_iter = 100;
    solver.settings().verbose = true;
    //solver.settings().line_search_max_iter = 10;
    //solver.settings().second_order_correction = true;
    //solver.settings().eta = 0.5;
    solver.settings().eps_grad = 1e-1;

    // Run SQP solver
    solver.solve(problem, x0, lambda0);

    // Print solution
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "primal solution " << solver.primal_solution().transpose() << std::endl;
    std::cout << "dual solution " << solver.dual_solution().transpose() << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}
