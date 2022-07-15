#include <Eigen/Core>
#include <Eigen/Dense>
#include "common/logger.hpp"
#include "modules/base/cost_fuction.hpp"
#include "modules/unconstrained_optimization/steepest_gradient_descent.hpp"

using namespace modules::optimization;

/**
 * @brief :
 *      待优化求解的Rosenbrock函数, 继承自CostFunction基类
 * @param :
 *      double* param: 初值; int size: 优化变量的维度
 */
class Rosenbrock : public CostFunction
{
public:

Rosenbrock(Eigen::VectorXd &param, int size) : CostFunction(param, size){};

~Rosenbrock() = default;

Eigen::VectorXd ComputeFunction(const Eigen::VectorXd &x) override
{
    Eigen::VectorXd result(1);
    result.setZero();
    result[0] = 0;
    for (size_t i = 0; i < N - 1; ++i)
    {
        double part_1 = x[i + 1] - x[i] * x[i];
        double part_2 = 1 - x[i];
        result[0] += 100.0 * part_1 * part_1 + part_2 * part_2;
    }
    return result;
}

Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd &x) override
{
    Eigen::MatrixXd jacobian(x.rows(),1);
    for (size_t i = 0; i < N; ++i)
    {
        if (i == 0){
            jacobian(i, 0) = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]);
        }else if (i == N - 1){
            jacobian(i, 0) = 200 * (x[i] - x[i - 1] * x[i - 1]);
        }else{
            jacobian(i, 0) = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]) + 200 * (x[i] - x[i - 1] * x[i - 1]);
        }
    }
    return jacobian;
}

};

class Example : public CostFunction
{

public:

Example(Eigen::VectorXd &param, int size) : CostFunction(param, size){};

~Example() = default;

Eigen::VectorXd ComputeFunction(const Eigen::VectorXd &x) override
{
    Eigen::VectorXd result(1);
    result.setZero();
    result[0] = x[0] * x[0] + 2 * x[1] * x[1] - 2 * x[0] * x[1] - 2 * x[1];
    return result;
}

Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd &x) override
{
    Eigen::MatrixXd jacobian(x.rows(), 1);
    jacobian(0, 0) = 2 * x[0] - 2 * x[1];
    jacobian(1, 0) = 4 * x[1] - 2 * x[0] - 2;
    return jacobian;
}

};


class Rosenbrock2 : public CostFunction
{
public:
    Rosenbrock2(Eigen::VectorXd &param, int size) : CostFunction(param, size){};

    ~Rosenbrock2() = default;

    Eigen::VectorXd ComputeFunction(const Eigen::VectorXd &x) override
    {
        Eigen::VectorXd result(1);
        result.setZero();
        result[0] = 0;
        for (size_t i = 0; i < N / 2; ++i)
        {
            double part_1 = x[2 * i] * x[2 * i] - x[2 * i + 1];
            double part_2 = x[2 * i] - 1;
            result[0] += 100.0 * part_1 * part_1 + part_2 * part_2;
        }
        return result;
    }

    Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd &x) override
    {
        Eigen::MatrixXd jacobian(x.rows(), 1);
        jacobian.setZero();
        for (size_t i = 0; i < N / 2; i++)
        {
            jacobian(2 * i, 0) = 400 * x[2 * i] * (x[2 * i] * x[2 * i] - x[2 * i + 1]) + 2 * (x[2 * i] - 1);
            jacobian(2 * i + 1, 0) = -200 * (x[2 * i] * x[2 * i] - x[2 * i + 1]);
        }
        return jacobian;
    }
};

int main(int argc, char** argv)
{
    common::Logger logger(argc, argv);
    Eigen::VectorXd x(2);
    x << 0.0, 0.0;
    Rosenbrock2 rosenbrock(x, x.rows());
    GradientDescent gd;
    gd.SetCostFunction(&rosenbrock);
    gd.SetEpsilon(1e-6);
    gd.SetC(0.01);
    gd.SetTau(1.0);

    auto res = gd.Solve();
    AINFO << res[0] << " " << res[1];
    AINFO << rosenbrock.ComputeFunction(res);

    // Example ex(x, x.rows());
    // GradientDescent gd;
    // gd.SetCostFunction(&ex);
    // auto res = gd.Solve();
    // AINFO << res[0] << " " << res[1];

    return 0;
}
