#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <chrono>
#include <iomanip>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// Custom sign function
inline double sgn(double val) {
    return (val > 0) - (val < 0);
}

// Apply soft-thresholding with tau * t elementwise
void sthreshmat(MatrixXd & x, double tau, const MatrixXd & t){
    MatrixXd tmp1 = x.unaryExpr([](double val) { return sgn(val); });
    MatrixXd tmp2 = (x.cwiseAbs() - tau * t).cwiseMax(0.0);
    x = tmp1.cwiseProduct(tmp2);
}

// Utility: Get formatted current time string
string current_time() {
    auto now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Gradient computation with log barrier and l2 penalty
MatrixXd compute_ista_grad_h1(const MatrixXd & X, const MatrixXd & W, double lam2) {
    VectorXd diag = X.diagonal();
    VectorXd inv_diag = X.diagonal().cwiseInverse();
    MatrixXd grad = -inv_diag.asDiagonal().toDenseMatrix() + W + lam2 * X;
    return grad;
}

SparseMatrix<double> accord_ista_backtracking(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> X_init,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    Ref<VectorXi> hist_inner_itr_count,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_successive_norm,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time,
    int logging_interval
) {
    MatrixXd X = X_init;
    MatrixXd W = X * S;
    MatrixXd grad_h1 = compute_ista_grad_h1(X, W, lam2);
    double h1 = -X.diagonal().array().log().sum() + 0.5 * (X.transpose() * W).trace() + 0.5 * lam2 * X.squaredNorm();

    int outer_itr_count = 0;
    double tau = 0.5, c_ = 0.5;

    while (true) {
        auto t1 = high_resolution_clock::now();

        MatrixXd Xn, Wn, Step;
        double h1n;
        int inner_itr_count = 0;

        while (true) {
            Xn = X - tau * grad_h1;
            sthreshmat(Xn, tau, LambdaMat);

            if (Xn.diagonal().minCoeff() > 0) {
                Wn = Xn * S;
                h1n = -Xn.diagonal().array().log().sum() + 0.5 * (Xn.transpose() * Wn).trace() + 0.5 * lam2 * Xn.squaredNorm();
                Step = Xn - X;
                double Q = h1 + Step.cwiseProduct(grad_h1).sum() + 0.5 / tau * Step.squaredNorm();
                if (h1n <= Q) break;
            }
            tau *= c_;
            inner_itr_count++;
        }

        auto t2 = high_resolution_clock::now();
        double elapsed = duration_cast<duration<double>>(t2 - t1).count();

        grad_h1 = compute_ista_grad_h1(Xn, Wn, lam2);
        double hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
        double successive_norm = (Xn - X).norm();
        double omega_star_norm = (Xn - Omega_star).norm();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        hist_hn(outer_itr_count) = hn;
        hist_successive_norm(outer_itr_count) = successive_norm;
        hist_norm(outer_itr_count) = omega_star_norm;
        hist_iter_time(outer_itr_count) = elapsed;

        outer_itr_count++;
        if (logging_interval > 0 && outer_itr_count % logging_interval == 0)
            cout << "[ACCORD][" << current_time() << "] Iteration: " << outer_itr_count << " | hn: " << hn << "\n";

        if (successive_norm < epstol || outer_itr_count >= maxitr) break;

        X = Xn;
        W = Wn;
        h1 = h1n;
    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << outer_itr_count << " | hn: " << hist_hn(outer_itr_count-1) << "\n";
    return X.sparseView();
}

SparseMatrix<double> accord_ista_constant(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> X_init,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_successive_norm,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time,
    int logging_interval
) {
    MatrixXd X = X_init;
    MatrixXd W = X * S;
    MatrixXd grad_h1 = compute_ista_grad_h1(X, W, lam2);
    double h1;

    int itr_count = 0;
    while (true) {
        auto t1 = high_resolution_clock::now();

        MatrixXd Xn = X - tau * grad_h1;
        sthreshmat(Xn, tau, LambdaMat);
        MatrixXd Step = Xn - X;
        MatrixXd Wn = Xn * S;
        double h1n = -Xn.diagonal().array().log().sum() + 0.5 * (Xn.transpose() * Wn).trace() + 0.5 * lam2 * Xn.squaredNorm();

        auto t2 = high_resolution_clock::now();
        double elapsed = duration_cast<duration<double>>(t2 - t1).count();

        grad_h1 = compute_ista_grad_h1(Xn, Wn, lam2);
        double hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
        double successive_norm = (Xn - X).norm();
        double omega_star_norm = (Xn - Omega_star).norm();

        hist_hn(itr_count) = hn;
        hist_successive_norm(itr_count) = successive_norm;
        hist_norm(itr_count) = omega_star_norm;
        hist_iter_time(itr_count) = elapsed;

        itr_count++;
        if (logging_interval > 0 && itr_count % logging_interval == 0)
            cout << "[ACCORD][" << current_time() << "] Iteration: " << itr_count << " | hn: " << hn << "\n";

        if (successive_norm < epstol || itr_count >= maxitr) break;

        X = Xn;
        W = Wn;
        h1 = h1n;
    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << itr_count << " | hn: " << hist_hn(itr_count-1) << "\n";
    return X.sparseView();
}

SparseMatrix<double> accord_fbs_backtracking(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> X_init,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Ref<VectorXi> hist_inner_itr_count,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_successive_norm,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time,
    int logging_interval
    ) {
    int p = S.cols();
    
    MatrixXd X = X_init;
    MatrixXd W = X * S;
    MatrixXd grad_h1 = W + lam2*X;
    double h1 = 0.5 * (X.transpose() * W).trace() + 0.5 * lam2 * X.squaredNorm();

    int outer_itr_count, inner_itr_count;
    double c_ = 0.5;
    outer_itr_count = 0;

    while(true) {

      auto t1 = high_resolution_clock::now();
      MatrixXd Xn, Wn, Step;
      double h1n;
      ArrayXd y;

      tau = 1.0;
      inner_itr_count = 0;

      while(true) {
        Xn = X - tau * grad_h1;

        if (penalize_diag == true) {
          y = Xn.diagonal().array() - tau*LambdaMat.diagonal().array();
        } else {
          y = Xn.diagonal().array();
        }

        y = 0.5 * (y+(y.pow(2.0) + 4*tau*VectorXd::Ones(p).array()).sqrt());
        sthreshmat(Xn, tau, LambdaMat);
        Xn.diagonal() = y;

        if (Xn.diagonal().minCoeff() > 0) {
          Step = Xn - X;
          Wn = Xn * S;
          h1n = 0.5 * (Xn.transpose() * Wn).trace() + 0.5 * lam2 * Xn.squaredNorm();
          double Q = h1 + Step.cwiseProduct(grad_h1).sum() + 0.5 / tau * Step.squaredNorm();
          if (h1n <= Q) break;
        }
        tau *= c_;
        inner_itr_count++;
      }

      auto t2 = high_resolution_clock::now();
      double elapsed = duration_cast<duration<double>>(t2 - t1).count();

      grad_h1 = Wn + lam2 * Xn;
      double hn = h1n - Xn.diagonal().array().log().sum() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
      double successive_norm = (Xn - X).norm();
      double omega_star_norm = (Xn - Omega_star).norm();

      hist_inner_itr_count(outer_itr_count) = inner_itr_count;
      hist_hn(outer_itr_count) = hn;
      hist_successive_norm(outer_itr_count) = successive_norm;
      hist_norm(outer_itr_count) = omega_star_norm;
      hist_iter_time(outer_itr_count) = elapsed;

      outer_itr_count++;
      if (logging_interval > 0 && outer_itr_count % logging_interval == 0)
          cout << "[ACCORD][" << current_time() << "] Iteration: " << outer_itr_count << " | hn: " << hn << "\n";

      if (successive_norm < epstol || outer_itr_count >= maxitr) break;

      X = Xn;
      W = Wn;
      h1 = h1n;
  }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << outer_itr_count << " | hn: " << hist_hn(outer_itr_count-1) << "\n";
    return X.sparseView();
}

SparseMatrix<double> accord_fbs_constant(
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> S,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> X_init,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> Omega_star,
    Ref<MatrixXd, 0, Stride<Dynamic, Dynamic>> LambdaMat,
    double lam2,
    double epstol,
    int maxitr,
    double tau,
    bool penalize_diag,
    Ref<VectorXd> hist_hn,
    Ref<VectorXd> hist_successive_norm,
    Ref<VectorXd> hist_norm,
    Ref<VectorXd> hist_iter_time,
    int logging_interval
    ) {

    int p = S.cols();

    MatrixXd X = X_init;
    MatrixXd W = X * S;
    MatrixXd grad_h1 = W + lam2*X;
    ArrayXd y;

    int itr_count;
    
    itr_count = 0;
    while(true) {

      auto t1 = high_resolution_clock::now();
      MatrixXd Xn, Wn, Step;

      Xn = X - tau * grad_h1;

      if (penalize_diag == true) {
        y = Xn.diagonal().array() - tau*LambdaMat.diagonal().array();
      } else {
        y = Xn.diagonal().array();
      }

      y = 0.5 * (y+(y.pow(2.0) + 4*tau*VectorXd::Ones(p).array()).sqrt());
      sthreshmat(Xn, tau, LambdaMat);
      Xn.diagonal() = y;

      auto t2 = high_resolution_clock::now();
      double elapsed = duration_cast<duration<double>>(t2 - t1).count();

      Wn = Xn * S;
      grad_h1 = Wn + lam2 * Xn;
      double hn = - Xn.diagonal().array().log().sum() + 0.5*(Xn.transpose()*Wn).trace() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum() + 0.5*lam2*Xn.squaredNorm();
      double successive_norm = (Xn - X).norm();
      double omega_star_norm = (Xn - Omega_star).norm();

      hist_hn(itr_count) = hn;
      hist_successive_norm(itr_count) = successive_norm;
      hist_norm(itr_count) = omega_star_norm;
      hist_iter_time(itr_count) = elapsed;

      itr_count++;
      if (logging_interval > 0 && itr_count % logging_interval == 0)
          cout << "[ACCORD][" << current_time() << "] Iteration: " << itr_count << " | hn: " << hn << "\n";

      if (successive_norm < epstol || itr_count >= maxitr) break;

      X = Xn;
      W = Wn;
  }

  cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << itr_count << " | hn: " << hist_hn(itr_count-1) << "\n";
  return X.sparseView();
}
