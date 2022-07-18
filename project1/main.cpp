#include "problem.hpp"
// #define BACKWARD_HAS_DW 1
// #include "backward.hpp"
// namespace backward
// {
//     backward::SignalHandling sh;
// } // namespace backward

int main(int argc, char** argv)
{
    common::Logger logger(argc, argv);

    Eigen::Vector2d x(0, 0);

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

    GradientDescent gd;
    gd.SetEpsilon(1e-3);
    gd.SetC(0.1);
    gd.SetTau(1.0);
    gd.SetVerbose(true);

    Rosenbrock2dExample rosenbrock(x);
    auto res = gd.Solve(rosenbrock);
    AINFO << "result = [" << res.transpose() << "]";
    AINFO << "f = " << rosenbrock.ComputeFunction(res);

    return 0;
}
