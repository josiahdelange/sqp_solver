#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "sqp.hpp"

using namespace sqp;

template <typename SCALAR_TYPE, typename DERIVED_TYPE>
class NLPAutoDiff : public sqp::NonLinearProblem<SCALAR_TYPE>
{
public:
    using Scalar = SCALAR_TYPE;
    using Vector = sqp::NonLinearProblem<double>::Vector;
    using Matrix = sqp::NonLinearProblem<double>::Matrix;

    using ADScalar = Eigen::AutoDiffScalar<Vector>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    template <typename vec>
    void ADVectorSeed(vec& x)
    {
        for (int i = 0; i < x.rows(); i++) {
            x[i].derivatives() = Vector::Unit(this->num_var, i);
        }
    }

    void objective(const Vector& x, Scalar& obj) override
    {
        static_cast<DERIVED_TYPE*>(this)->objective(x, obj);
    }

    void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) override
    {
        static_cast<DERIVED_TYPE*>(this)->constraint(x, c, l, u);
    }

    void objective_linearized(const Vector& x, Vector& grad, Scalar& obj) override
    {
        ADVector ad_x = x;
        ADScalar ad_obj;
        ADVectorSeed(ad_x);
        /* Static polymorphism using CRTP */
        static_cast<DERIVED_TYPE*>(this)->objective(ad_x, ad_obj);
        obj = ad_obj.value();
        grad = ad_obj.derivatives();
    }

    void constraint_linearized(const Vector& x, Matrix& Jc, Vector& c,
        Vector& l, Vector& u) override
    {
        ADVector ad_c(this->num_constr);
        ADVector ad_x = x;

        ADVectorSeed(ad_x);
        static_cast<DERIVED_TYPE*>(this)->constraint(ad_x, ad_c, l, u);

        // Fill constraint Jacobian
        for (int i = 0; i < ad_c.rows(); i++)
        {
            c[i] = ad_c[i].value();
            Eigen::Ref<Vector> deriv = ad_c[i].derivatives();
            Jc.row(i) = deriv.transpose();
        }
    }
};

class ConstrainedRosenbrock2D: public NLPAutoDiff<double, ConstrainedRosenbrock2D>
{
public:
    ConstrainedRosenbrock2D()
    {
        num_var = 2;
        num_constr = 2;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& x, DerivedB& z)
    {
        // (a-x)^2 + b*(y-x^2)^2
        const double a = 1;
        const double b = 100;
        z = 0;
        for (int i = 0; i < x.rows() - 1; i++)
        {
            z += pow(a - x[i], 2) + b * pow(x[i + 1] - pow(x[i], 2), 2);
        }
    }

    template <typename A, typename B>
    void constraint(const A& x, B& c, Vector& l, Vector& u)
    {
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        // y >= x
        // x^2 + y^2 == 1
        c << x(0) - x(1), x.squaredNorm();
        u << 0, 1;
        l << -infinity, 1;
    }
};

class RosenbrockDisk: public NLPAutoDiff<double, RosenbrockDisk>
{
public:
    RosenbrockDisk()
    {
        num_var = 2;
        num_constr = 1;
    }

    template <typename DerivedA, typename DerivedB>
    void objective(const DerivedA& decision, DerivedB& cost)
    {
        cost = pow(1 - decision[0], 2) +
            100*pow(decision[1] - pow(decision[0], 2), 2);
    }

    template <typename A, typename B>
    void constraint(const A& decision, B& constraints,
        Vector& lower_bounds, Vector& upper_bounds)
    {
        constraints << decision.squaredNorm() - 2;
        lower_bounds << -1.5, -1.5;
        upper_bounds << 1.5, 1.5;
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
