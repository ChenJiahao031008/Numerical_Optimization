#include <Eigen/Core>
#include <Eigen/Dense>
#include "common/logger.hpp"
#include "modules/base/cost_fuction.hpp"
#include "modules/unconstrained_optimization/steepest_gradient_descent.hpp"

using namespace modules::optimization;

/**
 * @brief :
 *      待优化求解的Rosenbrock函数, 继承自CostFunction基类, 来自wiki公式4
 * @param :
 *      double* param: 初值;
 */
class Rosenbrock : public CostFunction
{
public:
    Rosenbrock(Eigen::VectorXd &param) : CostFunction(param){};

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
        Eigen::MatrixXd jacobian(x.rows(), 1);
        for (size_t i = 0; i < N; ++i)
        {
            if (i == 0)
            {
                jacobian(i, 0) = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]);
            }
            else if (i == N - 1)
            {
                jacobian(i, 0) = 200 * (x[i] - x[i - 1] * x[i - 1]);
            }
            else
            {
                jacobian(i, 0) = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i]) + 200 * (x[i] - x[i - 1] * x[i - 1]);
            }
        }
        return jacobian;
    }
};

/**
 * @brief :
 *      待优化求解的Rosenbrock函数, 继承自CostFunction基类, 来自wiki公式3
 * @param :
 *      double* param: 初值;
 */
class Rosenbrock2 : public CostFunction
{
public:
    Rosenbrock2(Eigen::VectorXd &param) : CostFunction(param){};

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

/**
 * @brief :
 *      待优化求解的多项式函数, 继承自CostFunction基类
 * @param :
 *      double* param: 初值;
 */
class Example : public CostFunction
{

public:
    Example(Eigen::VectorXd &param) : CostFunction(param){};

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

/**
 * @brief :
 *      待优化求解的多项式函数, 继承自CostFunction基类，仅有两维便于可视化
 * @param :
 *      double* param: 初值;
 */
class Rosenbrock2dExample : public CostFunction
{
public:
    Rosenbrock2dExample(Eigen::VectorXd &param) : CostFunction(param){};

    ~Rosenbrock2dExample() = default;

    Eigen::VectorXd ComputeFunction(const Eigen::VectorXd &x) override
    {
        Eigen::VectorXd result(1);
        double part_1 = x[0] * x[0] - x[1];
        double part_2 = x[0] - 1;
        result[0] = 100.0 * part_1 * part_1 + part_2 * part_2;
        return result;
    }

    Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd &x) override
    {
        Eigen::MatrixXd jacobian(x.rows(), 1);
        jacobian(0, 0) = -2 * (1 - x[0]) - 400 * x[0] * ( x[1] - x[0] * x[0] );
        jacobian(1, 0) = 200 * (x[1] - x[0] * x[0]);
        return jacobian;
    }
};
