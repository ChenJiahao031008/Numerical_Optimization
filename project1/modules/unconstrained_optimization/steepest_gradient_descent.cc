#include "steepest_gradient_descent.hpp"

namespace modules::optimization
{

Eigen::VectorXd GradientDescent::Solve(){
    Eigen::VectorXd x = cost_function->GetInitParam();
    AINFO << "Iter Count 0: " << x.transpose();

    double delta = cost_function->ComputeJacobian(x).norm();
    int iter_count = 0;
    while (delta >= epsilon && iter_count < max_iters)
    {
        while (ArmijoCondition(x)){ tau = tau / 2.0; }
        Eigen::MatrixXd cur_d = cost_function->ComputeJacobian(x);
        x = x - tau * cur_d;
        delta = cur_d.norm();
        iter_count++;
        AINFO << "Iter [" << iter_count << "/" << max_iters << "]: " << delta;
        AINFO << "Current Param: " << x.transpose();
    }
    return x;
}

bool GradientDescent::ArmijoCondition(Eigen::VectorXd &x)
{
    Eigen::MatrixXd d = cost_function->ComputeJacobian(x);
    Eigen::VectorXd x_k0 = cost_function->ComputeFunction(x);
    Eigen::VectorXd x_k1 = cost_function->ComputeFunction(x - d * tau);
    Eigen::VectorXd left = x_k1 - x_k0;
    Eigen::VectorXd right = - c * tau * d.transpose() * d;
    AINFO << "ComputeJacobian : " << d.transpose();
    AINFO << "ComputeFunction x_k1 : " << x_k1;
    AINFO << "left = x_k1 - x_k0 : " << left;
    AINFO << "current tau : " << tau;
    if ( left[0] >= right[0] )
        return true;
    else
        return false;
}

}
