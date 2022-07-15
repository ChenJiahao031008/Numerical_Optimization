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

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    CostFunction(Eigen::VectorXd &param) : N(param.rows()), x(param){};

    virtual ~CostFunction() {};

    virtual double ComputeFunction(const Eigen::VectorXd &x) = 0;

    virtual Eigen::VectorXd ComputeJacobian(const Eigen::VectorXd &x) = 0;

    Eigen::VectorXd GetInitParam() { return x; };

    int GetParamSize() { return N; };
};

} // namespace modules::optimization
