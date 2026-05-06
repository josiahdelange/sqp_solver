#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include "sqp.h"

#ifndef SOLVER_ASSERT
#define SOLVER_ASSERT(x) eigen_assert(x)
#endif

namespace sqp {

SQP::SQP() {
    // TODO(mi): Performance strongly depends on QP solver settings, which is bad.
    _qp_solver.settings().warm_start = true;
    _qp_solver.settings().check_termination = 10;
    _qp_solver.settings().eps_abs = 1e-4;
    _qp_solver.settings().eps_rel = 1e-4;
    _qp_solver.settings().max_iter = 100;
    _qp_solver.settings().adaptive_rho = true;
    _qp_solver.settings().adaptive_rho_interval = 50;
    _qp_solver.settings().alpha = 1.6;
    _qp_solver.settings().verbose = false;

    _info.iter = 0;
    _info.qp_solver_iter;
    _info.status = UNKNOWN;
}

void SQP::solve(NonlinearProblem& prob, const Vector& x0, const Vector& lambda0) {
    _x = x0;
    _lambda = lambda0;
    run_solve(prob);
}

void SQP::solve(NonlinearProblem& prob) {
    const int nx = prob.num_var;
    const int nc = prob.num_constr;

    _x.setZero(nx);
    _lambda.setZero(nc);
    run_solve(prob);
}

void SQP::run_solve(NonlinearProblem& prob) {
    Vector p;         // search direction
    Vector p_lambda;  // dual search direction
    double alpha;     // step size
    prob.set_eps_grad(_settings.eps_grad);

    const int nx = prob.num_var;
    const int nc = prob.num_constr;

    p.resize(nx);
    p_lambda.resize(nc);

    _step_prev.resize(nx);
    _grad_L.resize(nx);
    _delta_grad_L.resize(nx);

    _Hess.resize(nx, nx);
    _grad_obj.resize(nx);
    _Jac_constr.resize(nc, nx);
    _constr.resize(nc);
    _l.resize(nc);
    _u.resize(nc);

    _info.qp_solver_iter = 0;

    if(_settings.iteration_callback) {
        _settings.iteration_callback(*this);
    }

    int& iter = _info.iter;
    for(iter = 1; iter <= _settings.max_iter; iter++) {
        // Solve QP
        solve_qp(prob, p, p_lambda);
        p_lambda -= _lambda;

        alpha = line_search(prob, p);

        // take step
        _x = _x + alpha * p;
        _lambda = _lambda + alpha * p_lambda;

        // update step info
        _step_prev = alpha * p;
        _primal_step_norm = alpha * p.template lpNorm<Eigen::Infinity>();
        _dual_step_norm = alpha * p_lambda.template lpNorm<Eigen::Infinity>();

        if(_settings.iteration_callback) {
            _settings.iteration_callback(*this);
        }

        if(_settings.verbose) {
            printf("SQP info:\n");
            printf("  Solver iteration: %d\n", _info.iter);
            std::cout << "  Primal: " << _x.transpose() << "\n";
            std::cout << "  Dual: " << _lambda.transpose() << "\n";
            printf("  QP iterations: %d\n", _info.qp_solver_iter);
            printf("  Solver status: ");
            if(termination_criteria(_x, prob)) {
                _info.status = SOLVED;
                switch (_info.status) {
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
                break;
            }
            switch (_info.status) {
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
    }
    if(iter > _settings.max_iter) {
        _info.status = MAX_ITER_EXCEEDED;
    }
}

bool is_posdef_eigen(SQP::Matrix H) {
    Eigen::EigenSolver<SQP::Matrix> eigensolver(H);
    for(int i = 0; i < eigensolver.eigenvalues().rows(); i++) {
        double v = eigensolver.eigenvalues()(i).real();
        if(v <= 0) {
            return false;
        }
    }
    return true;
}

template <typename Matrix>
bool is_posdef(Matrix H) {
    Eigen::LLT<Matrix> llt(H);
    if(llt.info() == Eigen::NumericalIssue) {
        return false;
    }
    return true;
}

bool SQP::termination_criteria(const Vector& x, NonlinearProblem& prob) {
    if(_primal_step_norm <= _settings.eps_prim && _dual_step_norm <= _settings.eps_dual &&
        max_constraint_violation(x, prob) <= _settings.eps_prim) {
        return true;
    }
    return false;
}

template <typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x) {
    // return ((x.array() == x.array())).all();
    return x.array().isNaN().any();
}

void SQP::solve_qp(NonlinearProblem& prob, Vector& step, Vector& lambda) {
    /* QP from linearized NLP:
     * minimize     0.5 x'Px + q'x
     * subject to   l <= Ax + b <= u
     *
     * with:
     *   P      Hessian of Lagrangian
     *   q      objective gradient
     *   A,b    linearized constraint at current iterate
     *   l,u    constraint bounds
     *
     * transform to:
     * minimize     0.5 x'Px + q'x
     * subject to   l <= Ax <= u
     *
     * Where the constraint bounds l,u set to l=u for equality constraints or
     * set to +/-INFINITY if unbounded.
     */
    prob.objective_linearized(_x, _obj, _grad_obj);
    prob.constraint_linearized(_x, _constr, _l, _u, _Jac_constr);

    _delta_grad_L = -_grad_L;
    _grad_L = _grad_obj + _Jac_constr.transpose() * _lambda;

    // BFGS update
    if(_info.iter == 1) {
        _Hess.setIdentity();
    } else {
        _delta_grad_L += _grad_L;  // _delta_grad_L = _grad_L_prev - grad_L

        // Damped Broyden–Fletcher–Goldfarb–Shanno (BFGS) update, implementing "Procedure
        // 18.2 - Damped BFGS updating for SQP" from Numerical Optimization by Nocedal.
        double sy, sr, sBs;
        Eigen::VectorXd Bs, r;

        Bs.noalias() = _Hess*_step_prev;
        sBs = _step_prev.dot(Bs);
        sy = _step_prev.dot(_delta_grad_L);

        if (sy < 0.2 * sBs) {
            // Damped update to enforce positive definite Hessian
            double theta;
            theta = 0.8*sBs/(sBs - sy);
            r.noalias() = theta*_delta_grad_L + (1 - theta)*Bs;
            sr = theta*sy + (1 - theta)*sBs;
        } else {
            // Unmodified BFGS
            r = _delta_grad_L;
            sr = sy;
        }

        if (sr < std::numeric_limits<double>::epsilon()) {
            return;
        }

        _Hess.noalias() += -Bs * Bs.transpose() / sBs;
        _Hess.noalias() += r * r.transpose() / sr;
    }

    if(!is_posdef(_Hess)) {
        std::cout << "Hessian not positive definite\n";
        double tau = 1e-3;
        Vector v = Vector(prob.num_var);
        while (!is_posdef(_Hess)) {
            v.setConstant(tau);
            _Hess += v.asDiagonal();
            tau *= 10;
        }
    }
    if(is_nan(_Hess)) {
        std::cout << "Hessian is NaN\n";
    }

    SOLVER_ASSERT(is_posdef(_Hess));
    SOLVER_ASSERT(!is_nan(_Hess));

    // Constraints
    // from   l <= Ax + b <= u
    // to   l-b <= Ax     <= u-b
    Vector l = _l - _constr;
    Vector u = _u - _constr;
    Matrix& A = _Jac_constr;
    Matrix& P = _Hess;
    Vector& q = _grad_obj;

    // solve the QP
    run_solve_qp(P, q, A, l, u, step, lambda);
    if(_settings.second_order_correction) {
        second_order_correction(prob, step, lambda);
    }

    // TODO:
    // B is not convex then use grad_L as step direction
    // i.e. fallback to steepest descent of Lagrangian
}

bool SQP::run_solve_qp(const Matrix& P, const Vector& q, const Matrix& A, const Vector& l,
                          const Vector& u, Vector& prim, Vector& dual) {
    qp::QuadraticProblem _qp;

    _qp.P = &P;
    _qp.q = &q;
    _qp.A = &A;
    _qp.l = &l;
    _qp.u = &u;

    _qp_solver.setup(_qp);
    _qp_solver.solve(_qp);

    _info.qp_solver_iter += _qp_solver.info().iter;

    if(_qp_solver.info().status == qp::NUMERICAL_ISSUES) {
        std::cout << "QPSolver NUMERICAL_ISSUES\n";
        return false;
    }
    //if(_qp_solver.info().status == qp::MAX_ITER_EXCEEDED) {
    //    std::cout << "QPSolver MAX_ITER_EXCEEDED\n";
    //    return false;
    //}

    prim = _qp_solver.primal_solution();
    dual = _qp_solver.dual_solution();

    SOLVER_ASSERT(!is_nan(prim));
    SOLVER_ASSERT(!is_nan(dual));

    return true;
}

void SQP::second_order_correction(NonlinearProblem& prob, Vector& p, Vector& lambda) {
    // double mu, constr_l1, phi_l1;
    // constr_l1 = constraint_norm(constr_, _l, _u);
    // mu = (_grad_obj.dot(p) + 0.5 * p.dot(_Hess * p)) / ((1 - _settings.rho) * constr_l1);
    // phi_l1 = _obj + mu * constr_l1;

    // double _objstep, constr_l1_step, phi_l1_step;
    // Vector _xstep = _x + p;
    // prob.objective(_xstep, _objstep);
    // constr_l1_step = constraint_norm(_xstep, prob);
    // phi_l1_step = _objstep + mu * constr_l1_step;

    // printf("phi_l1_step %f  phi_l1 %f  constr_l1_step %f  constr_l1 %f\n", phi_l1_step, phi_l1,
    //        constr_l1_step, constr_l1);
    // if(phi_l1_step >= phi_l1 && constr_l1_step >= constr_l1) {
    {
        Vector _xstep = _x + p;
        Vector constr_step(_constr.rows());
        prob.constraint(_xstep, constr_step, _l, _u);

        Matrix& A = _Jac_constr;
        Matrix& P = _Hess;
        Vector& q = _grad_obj;

        Vector d = constr_step - A * p;
        Vector l = _l - d;
        Vector u = _u - d;

        // TODO: only l and u change, possible to update QP solver more efficiently
        run_solve_qp(P, q, A, l, u, p, lambda);
    }
}

double SQP::line_search(NonlinearProblem& prob, const Vector& p) {
    // Note: using members _obj and _grad_obj, which are updated in solve_qp().
    double mu, phi_l1, Dp_phi_l1;
    const double tau = _settings.tau;  // line search step decrease, 0 < tau < settings.tau

    double constr_l1 = constraint_norm(_constr, _l, _u);

    // get mu from merit function model using hessian of Lagrangian instead
    mu = (_grad_obj.dot(p) + 0.5 * p.dot(_Hess * p)) / ((1 - _settings.rho) * constr_l1);

    phi_l1 = _obj + mu * constr_l1;
    Dp_phi_l1 = _grad_obj.dot(p) - mu * constr_l1;

    double alpha = 1.0;
    int i;
    for(i = 1; i < _settings.line_search_max_iter; i++) {
        double _objstep;
        Vector _xstep = _x + alpha * p;
        prob.objective(_xstep, _objstep);

        double phi_l1_step = _objstep + mu * constraint_norm(_xstep, prob);
        if(phi_l1_step <= phi_l1 + alpha * _settings.eta * Dp_phi_l1) {
            // accept step
            break;
        } else {
            alpha = tau * alpha;
        }
    }
    return alpha;
}

double SQP::constraint_norm(
    const Vector &constr, const Vector &l, const Vector &u) const {
    double c_l1 = DIV_BY_ZERO_REGUL;

    // l <= c(x) <= u
    c_l1 += (l - constr).cwiseMax(0.0).sum();
    c_l1 += (constr - u).cwiseMax(0.0).sum();

    return c_l1;
}

double SQP::constraint_norm(const Vector& x, NonlinearProblem& prob) {
    // Note: uses members _constr, _l and _u as temporary
    prob.constraint(x, _constr, _l, _u);

    return constraint_norm(_constr, _l, _u);
}

double SQP::max_constraint_violation(const Vector& x, NonlinearProblem& prob) {
    // Note: uses members _constr, _l and _u as temporary
    double c_max = 0;
    prob.constraint(x, _constr, _l, _u);

    // l <= c(x) <= u
    if(prob.num_constr > 0) {
        c_max = fmax(c_max, (_l - _constr).maxCoeff());
        c_max = fmax(c_max, (_constr - _u).maxCoeff());
    }

    return c_max;
}

}  // namespace sqp
