#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../../common/logger.hpp"

namespace modules::optimization{

class CostFunction
{
protected:
    const int N;

    Eigen::VectorXd x;

public:
    CostFunction(Eigen::VectorXd &param, int size) : N(size), x(param){};

    virtual ~CostFunction() {};

    virtual Eigen::VectorXd ComputeFunction(const Eigen::VectorXd &x) = 0;

    virtual Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd &x) = 0;

    Eigen::VectorXd GetInitParam() { return x; };

    int GetParamSize() { return N; };
};

} // namespace modules::optimization
