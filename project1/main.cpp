#include "problem.hpp"

int main(int argc, char** argv)
{
    common::Logger logger(argc, argv);

    Eigen::VectorXd x(2);
    x << 0.0, 0.0;

#if GCC_VERSION >= 90400
    namespace fs = std::filesystem;
#else
    namespace fs = boost::filesystem;
#endif
    const std::string recorder_path
        = fs::current_path().string()+ "/" + common::FLAGS_iter_recorder_file;
    fs::path recoder_file(recorder_path);
    AINFO << "Recorder File Path is : " << recoder_file.string();
    if (!fs::exists(recoder_file))
        fs::create_directories(recoder_file.parent_path());
    std::ofstream outfile(recorder_path, std::ios::out);

    Rosenbrock2dExample rosenbrock(x);
    GradientDescent gd;
    gd.SetCostFunction(&rosenbrock);
    gd.SetEpsilon(1e-3);
    gd.SetC(0.1);
    gd.SetTau(1.0);
    auto res = gd.Solve(outfile);
    AINFO << "result = [" << res.transpose() << "]";
    AINFO << "f = " << rosenbrock.ComputeFunction(res);

    return 0;
}
