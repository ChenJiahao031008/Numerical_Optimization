#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../common/logger.hpp"
#include "../modules/base/cost_fuction.hpp"
#include "../modules/unconstrained_optimization/steepest_gradient_descent.hpp"
#include "../problem.hpp"

TEST(TestUFreeOpt, TestSGDRosenbrock2d)
{
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    Rosenbrock2dExample rosenbrock(x);
    GradientDescent gd;
    gd.SetCostFunction(&rosenbrock);
    gd.SetEpsilon(1e-6);
    gd.SetC(0.5);
    gd.SetTau(1.0);
    gd.SetMaxIters(10000);
    gd.SetVerbose(false); // 过程不打印
    auto res = gd.Solve();
    Eigen::Vector2d true_result = {1.0, 1.0};
    EXPECT_NEAR ((res -true_result).norm(), 0, 1e-3);
}

TEST(TestUFreeOpt, TestSGDPolynomial)
{
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    Example polynomial(x);
    GradientDescent gd;
    gd.SetCostFunction(&polynomial);
    gd.SetEpsilon(1e-6);
    gd.SetC(0.5);
    gd.SetTau(1.0);
    gd.SetMaxIters(100);
    gd.SetVerbose(false); // 过程不打印
    auto res = gd.Solve();
    Eigen::Vector2d true_result = {1.0, 1.0};
    EXPECT_NEAR((res - true_result).norm(), 0, 1e-3);
}
