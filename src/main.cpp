#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace py = pybind11;

Eigen::SparseMatrix<double> accord_ista_backtracking(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> X_init,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time,
    int logging_interval
    );

Eigen::SparseMatrix<double> accord_ista_constant(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> X_init,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time,
    int logging_interval
    );

Eigen::SparseMatrix<double> accord_fbs_backtracking(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> X_init,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Eigen::Ref<Eigen::VectorXi> hist_inner_itr_count,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time,
    int logging_interval
    );

Eigen::SparseMatrix<double> accord_fbs_constant(
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> S,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> X_init,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Omega_star,
    Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Eigen::Ref<Eigen::VectorXd> hist_hn,
    Eigen::Ref<Eigen::VectorXd> hist_successive_norm,
    Eigen::Ref<Eigen::VectorXd> hist_norm,
    Eigen::Ref<Eigen::VectorXd> hist_iter_time,
    int logging_interval
    );

PYBIND11_MODULE(_gaccord, m) {

    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gaccord

        .. autosummary::
           :toctree: _generate

           add
           subtract
           det
           inv
           mutate
           incr_vector
           incr_matrix
           incr_matrix_any
    )pbdoc";

    m.def(
        "accord_ista_backtracking",
        &accord_ista_backtracking,
        R"pbdoc(
            ACCORD-ISTA with backtracking

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "accord_ista_constant",
        &accord_ista_constant,
        R"pbdoc(
            ACCORD-ISTA with constant step size

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "accord_fbs_backtracking",
        &accord_fbs_backtracking,
        R"pbdoc(
            ACCORD-FBS with backtracking

            Column-major sparse matrix
        )pbdoc");

    m.def(
        "accord_fbs_constant",
        &accord_fbs_constant,
        R"pbdoc(
            ACCORD-FBS with constant step size

            Column-major sparse matrix
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
