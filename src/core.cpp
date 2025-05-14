#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <chrono>
#include <iomanip>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

double sgn(double val) {
    return (double(0) < val) - (val < double(0));
}

void sthreshmat(MatrixXd & x,
                double tau,
                const MatrixXd & t){

    MatrixXd tmp1(x.cols(), x.cols());
    MatrixXd tmp2(x.cols(), x.cols());

    tmp1 = x.array().unaryExpr(ptr_fun(sgn));
    tmp2 = (x.cwiseAbs() - tau*t).cwiseMax(0.0);

    x = tmp1.cwiseProduct(tmp2);

    return;
}

string current_time() {
    auto now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c);
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

//////////// ACCORD-ISTA with backtracking ////////////
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

    int p = S.cols();

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd subg(p, p);
    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd subgrad_h2(p, p);              // subgradient of h2
    MatrixXd tmp(p, p);

    double Q, hn, h1, h1n, successive_norm, omega_star_norm;
    double tau, tau_ = 1.0;
    double c_ = 0.5;
    double elapsed;

    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += 0.5 * (W + W.transpose());
    grad_h1 += lam2 * X;

    X = X_init.sparseView();
    W = X * S;
    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += W;
    grad_h1 += lam2 * X;
    h1 = - X.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(X.transpose())*W).trace();

    int outer_itr_count, inner_itr_count;        // iteration counts

    outer_itr_count = 0;
    while (true) {

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
 
        tau = tau_;

        inner_itr_count = 0;

        while (true) {
            tmp = MatrixXd(X) - tau*grad_h1;
            sthreshmat(tmp, tau, LambdaMat);
            Xn = tmp.sparseView();

            if (tmp.diagonal().minCoeff() > 0) {

                Step = Xn - X;
                Wn = Xn * S;

                h1n = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose())*Wn).trace() + 0.5*lam2*pow(Xn.norm(),2);
                Q = h1 + Step.cwiseProduct(grad_h1).sum() + (0.5/tau)*Step.squaredNorm();

                if (h1n <= Q)
                    break; 

            }

            tau *= c_;
            inner_itr_count += 1;
        }

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        elapsed = time_span.count();

        grad_h1 = -1*Xn.diagonal().asDiagonal().inverse();
        grad_h1 += Wn;
        grad_h1 += lam2 * Xn;

        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
        successive_norm = Step.norm();
        omega_star_norm = (Xn - Omega_star).norm();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        hist_hn(outer_itr_count) = hn;
        hist_successive_norm(outer_itr_count) = successive_norm;
        hist_norm(outer_itr_count) = omega_star_norm;
        hist_iter_time(outer_itr_count) = elapsed;

        outer_itr_count += 1;
        if (logging_interval > 0 && outer_itr_count % logging_interval == 0) {
            cout << "[ACCORD][" << current_time() << "] Iteration: " << outer_itr_count << " | hn: " << hn << "\n";
        }
        
        // change termination contition (omega_star_norm -> successive_norm)
        if (successive_norm < epstol || outer_itr_count >= maxitr) {
            if (outer_itr_count <= maxitr) {
                hist_inner_itr_count(outer_itr_count) = -1;
                hist_hn(outer_itr_count) = -1;
                hist_successive_norm(outer_itr_count) = -1;
                hist_norm(outer_itr_count) = -1;
                hist_iter_time(outer_itr_count) = -1;
            }
            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << outer_itr_count << " | hn: " << hn << "\n";

    return Xn;
}


//////////// ACCORD-ISTA with constant step size ////////////
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

    int p = S.cols();

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd subg(p, p);
    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd subgrad_h2(p, p);              // subgradient of h2
    MatrixXd tmp(p, p);

    double hn, h1, h1n, successive_norm, omega_star_norm;
    double elapsed;

    X.setIdentity();                         // initial guess: X = I
    W = X * S;

    grad_h1 = -1*X.diagonal().asDiagonal().inverse();
    grad_h1 += W;
    grad_h1 += lam2 * X;
    h1 = - X.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(X.transpose())*W).trace();

    int itr_count;                             // iteration counts

    itr_count = 0;
    while (true) {

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        tmp = MatrixXd(X) - tau*grad_h1;
        sthreshmat(tmp, tau, LambdaMat);
        Xn = tmp.sparseView();
        Step = Xn - X;
        Wn = Xn * S;
        h1n = - Xn.diagonal().array().log().sum() + 0.5 * (SparseMatrix<double>(Xn.transpose())*Wn).trace() + 0.5*lam2*pow(Xn.norm(),2);

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        elapsed = time_span.count();

        grad_h1 = -1*Xn.diagonal().asDiagonal().inverse();
        grad_h1 += Wn;
        grad_h1 += lam2 * Xn;

        hn = h1n + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
        successive_norm = Step.norm();
        omega_star_norm = (Xn - Omega_star).norm();

        hist_hn(itr_count) = hn;
        hist_successive_norm(itr_count) = successive_norm;
        hist_norm(itr_count) = omega_star_norm;
        hist_iter_time(itr_count) = elapsed;

        itr_count += 1;
        if (logging_interval > 0 &itr_count % logging_interval == 0) {
            cout << "[ACCORD][" << current_time() << "] Iteration: " << itr_count << " | hn: " << hn << "\n";
        }

        // change termination contition (omega_star_norm -> successive_norm)
        if (successive_norm < epstol || itr_count >= maxitr) {
            if (itr_count <= maxitr) {
                hist_hn(itr_count) = -1;
                hist_successive_norm(itr_count) = -1;
                hist_norm(itr_count) = -1;
                hist_iter_time(itr_count) = -1;
            }
            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << itr_count << " | hn: " << hn << "\n";

    return Xn;
}


//////////// ACCORD-FBS with backtracking ////////////
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

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    SparseMatrix<double> Step(p, p);        // Xn - X
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd tmp(p, p);
    ArrayXd y(p);                           // diagonal elements

    double Q, hn, h1, h1n, successive_norm, omega_star_norm;
    double c_ = 0.5;
    double tau_;
    double elapsed;

    X = X_init.sparseView();
    W = X * S;
    grad_h1 = W + lam2 * X;                 // gradient
    h1 = 0.5 * (SparseMatrix<double>(X.transpose())*W).trace() + 0.5*lam2*pow(X.norm(),2);

    int outer_itr_count, inner_itr_count;   // iteration counts

    outer_itr_count = 0;
    while (true) {

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        tau_ = 1.0;
        inner_itr_count = 0;
        while(true){

            tmp = MatrixXd(X) - tau_*grad_h1;
            
            if (penalize_diag == true) {
                y = tmp.diagonal().array() - tau_*LambdaMat.diagonal().array();
            } else {
                y = tmp.diagonal().array();
            }

            y = 0.5 * (y+(y.pow(2.0)+4*tau_*Eigen::VectorXd::Ones(p).array()).sqrt());
            sthreshmat(tmp, tau_, LambdaMat);
            tmp.diagonal() = y;
            Xn = tmp.sparseView();

            // backtracking line search bounded from below
            if (tmp.diagonal().minCoeff() > 0) {

                Step = Xn - X;
                Wn = Xn * S;

                h1n = 0.5*(SparseMatrix<double>(Xn.transpose())*Wn).trace() + 0.5*lam2*pow(Xn.norm(),2);
                Q = h1 + Step.cwiseProduct(grad_h1).sum() + (0.5/tau_)*Step.squaredNorm();

                if ((tau_ <= tau) || (h1n <= Q)){
                    break;
                }
            }

            tau_ *= c_;
            inner_itr_count += 1;
        }

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        elapsed = time_span.count();
       
        grad_h1 = Wn + lam2 * Xn;
        hn = h1n - Xn.diagonal().array().log().sum() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum();
        successive_norm = Step.norm();
        omega_star_norm = (Xn - Omega_star).norm();

        hist_inner_itr_count(outer_itr_count) = inner_itr_count;
        hist_hn(outer_itr_count) = hn;
        hist_successive_norm(outer_itr_count) = successive_norm;
        hist_norm(outer_itr_count) = omega_star_norm;
        hist_iter_time(outer_itr_count) = elapsed;
        
        outer_itr_count += 1;
        if (logging_interval > 0 && outer_itr_count % logging_interval == 0) {
            cout << "[ACCORD][" << current_time() << "] Iteration: " << outer_itr_count << " | hn: " << hn << "\n";
        }

        // change termination contition (omega_star_norm -> successive_norm)
        if (successive_norm < epstol || outer_itr_count >= maxitr) {
            if (outer_itr_count <= maxitr) {
                hist_inner_itr_count(outer_itr_count) = -1;
                hist_hn(outer_itr_count) = -1;
                hist_successive_norm(outer_itr_count) = -1;
                hist_norm(outer_itr_count) = -1;
                hist_iter_time(outer_itr_count) = -1;
            }
            break;
        } else {
            h1 = h1n;
            X = Xn;
            W = Wn;
        }

    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << outer_itr_count << " | hn: " << hn << "\n";

    return Xn;
}

//////////// ACCORD-FBS with constant step size ////////////
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

    SparseMatrix<double> X(p, p), Xn(p, p); // current and next estimate
    MatrixXd W(p, p), Wn(p, p);             // X*S

    MatrixXd grad_h1(p, p);                 // gradient of h1
    MatrixXd tmp(p, p);
    ArrayXd y(p);                           // diagonal elements

    double hn, successive_norm, omega_star_norm;
    double elapsed;

    X = X_init.sparseView();
    W = X * S;
    grad_h1 = W + lam2 * X;             // new gradient with l2-regularization

    int itr_count;                          // iteration counts

    itr_count = 0;
    while (true) {

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        tmp = MatrixXd(X) - tau*grad_h1;

        if (penalize_diag == true) {
            y = tmp.diagonal().array() - tau*LambdaMat.diagonal().array();
        } else {
            y = tmp.diagonal().array();
        }

        y = 0.5 * (y+(y.pow(2.0)+4*tau*Eigen::VectorXd::Ones(p).array()).sqrt());
        sthreshmat(tmp, tau, LambdaMat);
        tmp.diagonal() = y;
        Xn = tmp.sparseView();

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        elapsed = time_span.count();
       
        Wn = Xn * S;
        grad_h1 = Wn + lam2 * Xn;
        hn = - Xn.diagonal().array().log().sum() + 0.5*(SparseMatrix<double>(Xn.transpose())*Wn).trace() + Xn.cwiseAbs().cwiseProduct(LambdaMat).sum() + 0.5*lam2*pow(Xn.norm(),2);
        successive_norm = (Xn - X).norm();
        omega_star_norm = (Xn - Omega_star).norm();

        hist_hn(itr_count) = hn;
        hist_successive_norm(itr_count) = successive_norm;
        hist_norm(itr_count) = omega_star_norm;
        hist_iter_time(itr_count) = elapsed;
        
        itr_count += 1;
        if (logging_interval > 0 &itr_count % logging_interval == 0) {
            cout << "[ACCORD][" << current_time() << "] Iteration: " << itr_count << " | hn: " << hn << "\n";
        }

        // change termination contition (omega_star_norm -> successive_norm)
        if (successive_norm < epstol || itr_count >= maxitr) {
            if (itr_count <= maxitr) {
                hist_hn(itr_count) = -1;
                hist_successive_norm(itr_count) = -1;
                hist_norm(itr_count) = -1;
                hist_iter_time(itr_count) = -1;
            }
            break;
        } else {
            X = Xn;
            W = Wn;
        }
    }

    cout << "[ACCORD][" << current_time() << "] Total Iteration Count: " << itr_count << " | hn: " << hn << "\n";

    return Xn;
}
