#ifndef SVM_PREDICT_H
#define SVM_PREDICT_H


#include <iostream>
#include <vector>
#include <cmath>

// 计算点积
double dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// 线性核函数
double linear_kernel(const std::vector<double>& x1, const std::vector<double>& x2) {
    return dot_product(x1, x2);
}

// 高斯核函数（RBF核）
double gaussian_kernel(const std::vector<double>& x1, const std::vector<double>& x2, double gamma) {
    double sum = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return std::exp(-gamma * sum);
}

// SVM预测函数（线性核）
double svm_predict_linear(const std::vector<std::vector<double>>& support_vectors,
                          const std::vector<double>& coefficients,
                          double intercept,
                          const std::vector<double>& x) {
    double result = intercept;
    for (size_t i = 0; i < support_vectors.size(); ++i) {
        result += coefficients[i] * linear_kernel(support_vectors[i], x);
    }
    return result;
}

// SVM预测函数（高斯核）
double svm_predict_gaussian(const std::vector<std::vector<double>>& support_vectors,
                            const std::vector<double>& coefficients,
                            double intercept,
                            const std::vector<double>& x,
                            double gamma) {
    double result = intercept;
    for (size_t i = 0; i < support_vectors.size(); ++i) {
        result += coefficients[i] * gaussian_kernel(support_vectors[i], x, gamma);
    }
    return result;
}

#endif // SVM_PREDICT_H
