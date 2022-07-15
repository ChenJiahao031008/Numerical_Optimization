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
        Eigen::VectorXd cur_d = cost_function->ComputeJacobian(x);
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

Eigen::VectorXd GradientDescent::Solve(std::ofstream &output)
{
    Eigen::VectorXd x = cost_function->GetInitParam();

    double delta = cost_function->ComputeJacobian(x).norm();
    int iter_count = 0;
    output << std::fixed << std::setprecision(6);
    while (delta >= epsilon && iter_count < max_iters)
    {
        tau = init_tau;
        while (ArmijoCondition(x))
        {
            tau = tau * 0.5;
        }
        Eigen::VectorXd cur_d = cost_function->ComputeJacobian(x);
        double cur_f = cost_function->ComputeFunction(x);
        x = x - tau * cur_d;
        delta = cur_d.norm();
        iter_count++;
        output << iter_count << " " << x.transpose() << " " << cur_f << std::endl;
    }

    return x;
}

bool GradientDescent::ArmijoCondition(Eigen::VectorXd &x)
{
    Eigen::VectorXd d = cost_function->ComputeJacobian(x);
    double x_k0 = cost_function->ComputeFunction(x);
    double x_k1 = cost_function->ComputeFunction(x - d * tau);
    double left = x_k1 - x_k0;
    double right = -(c * d.transpose() * d * tau)[0];
    if ( left >= right )
        return true;
    else
        return false;
}

}
