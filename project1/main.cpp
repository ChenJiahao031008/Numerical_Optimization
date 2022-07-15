#include "problem.hpp"

int main(int argc, char** argv)
{
    common::Logger logger(argc, argv);
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;

    Rosenbrock2dExample rosenbrock(x);
    GradientDescent gd;
    gd.SetCostFunction(&rosenbrock);
    gd.SetEpsilon(1e-3);
    gd.SetC(0.5);
    gd.SetTau(1.0);
    auto res = gd.Solve();
    AINFO << "result = [" << res.transpose() << "]";
    AINFO << "f = " << rosenbrock.ComputeFunction(res);

    // Example ex(x);
    // GradientDescent gd;
    // gd.SetCostFunction(&ex);
    // auto res = gd.Solve();
    // AINFO << "result = [" << res.transpose() << "]";
    // AINFO << "f = " << ex.ComputeFunction(res);

    return 0;
}
