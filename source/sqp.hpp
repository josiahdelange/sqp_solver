#pragma once

#include <Eigen/Dense>
#include <limits>

#include "qp.hpp"

namespace sqp {

template <typename T>
class SQP;

template <typename Scalar>
struct sqp_settings_t {
    Scalar tau = 0.5;       /**< line search iteration decrease, 0 < tau < 1 */
    Scalar eta = 0.25;      /**< line search parameter, 0 < eta < 1 */
    Scalar rho = 0.5;       /**< line search parameter, 0 < rho < 1 */
    Scalar eps_prim = 1e-4; /**< primal step termination threshold, eps_prim > 0 */
    Scalar eps_dual = 1e-4; /**< dual step termination threshold, eps_dual > 0 */
    int max_iter = 100;
    int line_search_max_iter = 20;
    bool second_order_correction = false;
    bool verbose = false;
    std::function<void(const SQP<Scalar>&)> iteration_callback;

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

    void print() {
        printf("SQP info:\n");
        printf("  iter: %d\n", iter);
        printf("  qp_solver_iter: %d\n", qp_solver_iter);
        printf("  status: ");
        switch (status) {
            case SOLVED:
                printf("SOLVED\n");
                break;
            case MAX_ITER_EXCEEDED:
                printf("MAX_ITER_EXCEEDED\n");
                break;
            case INVALID_SETTINGS:
                printf("INVALID_SETTINGS\n");
                break;
            default:
                printf("UNKNOWN\n");
                break;
        }
    }
};

template <typename SCALAR_TYPE = double>
struct NonLinearProblem {
    using Scalar = SCALAR_TYPE;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    int num_var;
    int num_constr;

    virtual void objective(const Vector& x, Scalar& obj) = 0;
    virtual void constraint(const Vector& x, Vector& c, Vector& l, Vector& u) = 0;

    virtual void objective_linearized(const Vector& x, Scalar& obj, Vector& grad)
    {
        objective(x, obj);
        Scalar eps_grad = 1e-12;
        Vector err = Vector::Zero(num_var);
        for(int ii = 0; ii < num_var; ii++) {
            err.setZero(num_var);
            err[ii] = eps_grad;

            Scalar obj1 = 0.0;
            Scalar obj2 = 0.0;
            objective(x + err, obj1);
            objective(x - err, obj2);
            grad[ii] = (obj1 - obj2)/(2*eps_grad);
        }
    }

    virtual void constraint_linearized(const Vector& x, Vector& c, Vector& l, Vector& u, Matrix& Jc)
    {
        constraint(x, c, l, u);
        Scalar eps_grad = 1e-12;
        Vector err = Vector::Zero(num_var);
        for(int jj = 0; jj < num_constr; jj++) {
            for(int ii = 0; ii < num_var; ii++) {
                err.setZero(num_var);
                err[ii] = eps_grad;

                Vector c1 = Vector::Zero(num_constr);
                Vector c2 = Vector::Zero(num_constr);
                constraint(x + err, c1, l, u);
                constraint(x - err, c2, l, u);
                Jc(jj,ii) = (c1[jj] - c2[jj])/(2*eps_grad);
            }
        }
    }
};

/*
 * minimize     f(x)
 * subject to   l <= c(x) <= u
 */
template <typename SCALAR_TYPE>
class SQP {
public:
    using Scalar = SCALAR_TYPE;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Problem = NonLinearProblem<Scalar>;
    using Settings = sqp_settings_t<Scalar>;

    // Constants
    static constexpr Scalar DIV_BY_ZERO_REGUL = std::numeric_limits<Scalar>::epsilon();

    // enforce 16 byte alignment
    // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SQP();
    ~SQP() = default;

    void solve(Problem& prob, const Vector& x0, const Vector& lambda0);
    void solve(Problem& prob);

    inline const Vector& primal_solution() const { return _x; }
    inline Vector& primal_solution() { return _x; }

    inline const Vector& dual_solution() const { return _lambda; }
    inline Vector& dual_solution() { return _lambda; }

    inline const Settings& settings() const { return _settings; }
    inline Settings& settings() { return _settings; }

    inline const Info& info() const { return _info; }
    inline Info& info() { return _info; }

    // private:
    void run_solve(Problem& prob);

    bool termination_criteria(const Vector& x, Problem& prob);
    void solve_qp(Problem& prob, Vector& p, Vector& lambda);
    bool run_solve_qp(const Matrix& P, const Vector& q, const Matrix& A, const Vector& l,
                      const Vector& u, Vector& prim, Vector& dual);

    /** Second order correction by solving the same QP with corrected constraints. */
    void second_order_correction(Problem& prob, Vector& p, Vector& lambda);

    /** Line search in direction p using l1 merit function. */
    Scalar line_search(Problem& prob, const Vector& p);

    /** L1 norm of constraint violation */
    Scalar constraint_norm(const Vector& x, Problem& prob);

    /** L1 norm of constraint violation, for given constraint evaluation */
    Scalar constraint_norm(const Vector &constr, const Vector &l, const Vector &u) const;

    /** L_inf norm of constraint violation */
    Scalar max_constraint_violation(const Vector& x, Problem& prob);

    // Solver state variables
    Vector _x;
    Vector _lambda;
    Vector _step_prev;
    Vector _grad_L;
    Vector _delta_grad_L;

    Matrix _Hess;
    Vector _grad_obj;
    Scalar _obj;
    Matrix _Jac_constr;
    Vector _constr;
    Vector _l, _u;

    // info
    Scalar _dual_step_norm;
    Scalar _primal_step_norm;

    Settings _settings;
    Info _info;

    qp::QPSolver<Scalar> _qp_solver;
};

extern template class SQP<double>;

}  // namespace sqp
