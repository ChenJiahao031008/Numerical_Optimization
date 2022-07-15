#pragma once

#include <ostream>
#include <fstream>
#include "../../common/logger.hpp"
#include "../base/cost_fuction.hpp"

namespace modules::optimization{

class GradientDescent
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    double tau, init_tau; // todo: tau shouble be a vector
    double c;
    double epsilon;
    int max_iters;
    bool verbose;
    CostFunction* cost_function;

public:
    GradientDescent() : init_tau(tau), c(0.5), epsilon(1e-6), max_iters(100000), verbose(true){};

    GradientDescent(double tau_, double c_, double epsilon_, int max_iters_, bool verbose_)
        : init_tau(tau_), c(c_), epsilon(epsilon_),
          max_iters(max_iters_), verbose(verbose_) {};

    ~GradientDescent() = default;

    inline void SetCostFunction(CostFunction *cost) { cost_function = cost; };
    inline void SetTau(double tau_) { init_tau = tau_; };
    inline void SetC(double c_) { c = c_; };
    inline void SetEpsilon(double epsilon_) { epsilon = epsilon_; };
    inline void SetMaxIters(int max_iters_) { max_iters = max_iters_; };
    inline void SetVerbose(bool verbose_) { verbose = verbose_; };

    Eigen::VectorXd Solve();
    Eigen::VectorXd Solve(std::ofstream &out);

private:
    bool ArmijoCondition(Eigen::VectorXd &x);
};




}
