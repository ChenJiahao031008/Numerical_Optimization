#include "steepest_gradient_descent.hpp"

namespace modules::optimization
{

Eigen::VectorXd GradientDescent::Solve(){
    Eigen::VectorXd x = cost_function->GetInitParam();
    if (verbose)
        AINFO << "Iter Count 0: Init vector is [" << x.transpose() << "]";

    double delta = cost_function->ComputeJacobian(x).norm();
    int iter_count = 0;
    while (delta >= epsilon && iter_count < max_iters)
    {
        tau = init_tau;
        while (ArmijoCondition(x)){ tau = tau * 0.5; }
        Eigen::MatrixXd cur_d = cost_function->ComputeJacobian(x);
        x = x - tau * cur_d;
        delta = cur_d.norm();
        iter_count++;
        if (verbose){
            AINFO << "Iter-->[" << iter_count << "/" << max_iters << "] delta: " << delta;
            AINFO << "Current Param: " << x.transpose();
        }

    }
    return x;
}

bool GradientDescent::ArmijoCondition(Eigen::VectorXd &x)
{
    Eigen::MatrixXd d = cost_function->ComputeJacobian(x);
    Eigen::VectorXd x_k0 = cost_function->ComputeFunction(x);
    Eigen::VectorXd x_k1 = cost_function->ComputeFunction(x - d * tau);
    Eigen::VectorXd left = x_k1 - x_k0;
    Eigen::VectorXd right = -c * d.transpose() * d * tau;
    if ( left[0] >= right[0] )
        return true;
    else
        return false;
}

}
