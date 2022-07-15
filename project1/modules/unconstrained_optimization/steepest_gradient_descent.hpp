#pragma once

#include "../../common/logger.hpp"
#include "../base/cost_fuction.hpp"

namespace modules::optimization{

class GradientDescent
{
private:
    double tau; // todo: tau shouble be a vector
    double c;
    double epsilon;
    int max_iters;
    CostFunction* cost_function;
public:
    GradientDescent() : tau(1.0), c(0.5), epsilon(1e-6), max_iters(5000) {};

    GradientDescent(double tau_, double c_, double epsilon_)
        : tau(tau_), c(c_), epsilon(epsilon_) {};

    ~GradientDescent() = default;

    inline void SetCostFunction(CostFunction *cost) { cost_function = cost; };

    inline void SetTau(double tau_) { tau = tau_; };

    inline void SetC(double c_) { c = c_; };

    inline void SetEpsilon(double epsilon_) { epsilon = epsilon_; };

    Eigen::VectorXd Solve();

private:
    bool ArmijoCondition(Eigen::VectorXd &x);
};




}
