#pragma once

#include <Eigen/Dense>
#include <limits>

#include "qp.h"

namespace sqp {

class SQP;

struct sqp_settings_t {
    double tau = 0.5;        /**< line search iteration decrease, 0 < tau < 1 */
    double eta = 0.25;       /**< line search parameter, 0 < eta < 1 */
    double rho = 0.5;        /**< line search parameter, 0 < rho < 1 */
    double eps_prim = 1e-4;  /**< primal step termination threshold, eps_prim > 0 */
    double eps_dual = 1e-4;  /**< dual step termination threshold, eps_dual > 0 */
    double eps_grad = 1e-12; /**< gradient finite difference step size, eps_grad > 0 */
    int max_iter = 100;
    int line_search_max_iter = 20;
    bool second_order_correction = false;
    bool verbose = false;
    std::function<void(const SQP&)> iteration_callback;

    bool validate() {
        bool valid;
        valid = 0.0 < tau && tau < 1.0 && 0.0 < eta && eta < 1.0 && 0.0 < rho &&
            rho < 1.0 && eps_prim < 0.0 && eps_dual < 0.0 && max_iter > 0 &&
            line_search_max_iter > 0;
        return valid;
    }
};

typedef enum { UNKNOWN, SOLVED, MAX_ITER_EXCEEDED, INVALID_SETTINGS } Status;
struct Info {
    int iter;
    int qp_solver_iter;
    Status status;
};

class NonlinearProblem {
public:
    int num_var;
    int num_constr;
    double _eps_grad;

    void set_eps_grad(const double& eps_grad) { _eps_grad = eps_grad; }

    virtual void objective(const Eigen::VectorXd& x, double& obj) = 0;
    virtual void constraint(const Eigen::VectorXd& x, Eigen::VectorXd& c,
        Eigen::VectorXd& l, Eigen::VectorXd& u) = 0;

    virtual void objective_linearized(const Eigen::VectorXd& x,
        double& obj, Eigen::VectorXd& grad)
    {
        objective(x, obj);
        Eigen::VectorXd err = Eigen::VectorXd::Zero(num_var);
        for(int ii = 0; ii < num_var; ii++) {
            err.setZero(num_var);
            err[ii] = _eps_grad;

            double obj1 = 0.0;
            double obj2 = 0.0;
            objective(x + err, obj1);
            objective(x - err, obj2);
            grad[ii] = (obj1 - obj2)/(2*_eps_grad);
        }
    }

    virtual void constraint_linearized(const Eigen::VectorXd& x, Eigen::VectorXd& c,
        Eigen::VectorXd& l, Eigen::VectorXd& u, Eigen::MatrixXd& Jc)
    {
        constraint(x, c, l, u);
        Eigen::VectorXd err = Eigen::VectorXd::Zero(num_var);
        for(int jj = 0; jj < num_constr; jj++) {
            for(int ii = 0; ii < num_var; ii++) {
                err.setZero(num_var);
                err[ii] = _eps_grad;

                Eigen::VectorXd c1 = Eigen::VectorXd::Zero(num_constr);
                Eigen::VectorXd c2 = Eigen::VectorXd::Zero(num_constr);
                constraint(x + err, c1, l, u);
                constraint(x - err, c2, l, u);
                Jc(jj,ii) = (c1[jj] - c2[jj])/(2*_eps_grad);
            }
        }
    }
};

/*
 * minimize     f(x)
 * subject to   l <= c(x) <= u
 */
class SQP {
public:
    // Constants
    static constexpr double DIV_BY_ZERO_REGUL = std::numeric_limits<double>::epsilon();

    // enforce 16 byte alignment
    // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SQP();
    ~SQP() = default;

    void solve(NonlinearProblem& prob, 
        const Eigen::VectorXd& x0, const Eigen::VectorXd& lambda0);
    void solve(NonlinearProblem& prob);

    inline const Eigen::VectorXd& primal_solution() const { return _x; }
    inline Eigen::VectorXd& primal_solution() { return _x; }

    inline const Eigen::VectorXd& dual_solution() const { return _lambda; }
    inline Eigen::VectorXd& dual_solution() { return _lambda; }

    inline const sqp_settings_t& settings() const { return _settings; }
    inline sqp_settings_t& settings() { return _settings; }

    inline const Info& info() const { return _info; }
    inline Info& info() { return _info; }

    // private:
    void run_solve(NonlinearProblem& prob);

    bool termination_criteria(const Eigen::VectorXd& x, NonlinearProblem& prob);
    void solve_qp(NonlinearProblem& prob, Eigen::VectorXd& p, Eigen::VectorXd& lambda);
    bool run_solve_qp(const Eigen::MatrixXd& P, const Eigen::VectorXd& q,
        const Eigen::MatrixXd& A, const Eigen::VectorXd& l,
        const Eigen::VectorXd& u, Eigen::VectorXd& prim, Eigen::VectorXd& dual);

    /** Second order correction by solving the same QP with corrected constraints. */
    void second_order_correction(NonlinearProblem& prob,
        Eigen::VectorXd& p, Eigen::VectorXd& lambda);

    /** Line search in direction p using l1 merit function. */
    double line_search(NonlinearProblem& prob, const Eigen::VectorXd& p);

    /** L1 norm of constraint violation */
    double constraint_norm(const Eigen::VectorXd& x, NonlinearProblem& prob);

    /** L1 norm of constraint violation, for given constraint evaluation */
    double constraint_norm(const Eigen::VectorXd &constr,
        const Eigen::VectorXd &l, const Eigen::VectorXd &u) const;

    /** L_inf norm of constraint violation */
    double max_constraint_violation(const Eigen::VectorXd& x, NonlinearProblem& prob);

    // Solver state variables
    Eigen::VectorXd _x;
    Eigen::VectorXd _lambda;
    Eigen::VectorXd _step_prev;
    Eigen::VectorXd _grad_L;
    Eigen::VectorXd _delta_grad_L;

    Eigen::MatrixXd _Hess;
    Eigen::VectorXd _grad_obj;
    double _obj;
    Eigen::MatrixXd _Jac_constr;
    Eigen::VectorXd _constr;
    Eigen::VectorXd _l, _u;

    // info
    double _dual_step_norm;
    double _primal_step_norm;

    sqp_settings_t _settings;
    Info _info;

    qp::QPSolver _qp_solver;
};

}  // namespace sqp
