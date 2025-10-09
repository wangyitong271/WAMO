#ifndef GAN_H
#define GAN_H

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstddef>

// 激活函数实现
inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double tanh_activation(double x) {
    return tanh(x);
}

// 应用激活函数
std::vector<double> apply_activation(const std::vector<double>& input, const std::string& activation) {
    std::vector<double> output = input;

    if (activation == "relu") {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = relu(output[i]);
        }
    } else if (activation == "sigmoid") {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = sigmoid(output[i]);
        }
    } else if (activation == "tanh") {
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = tanh_activation(output[i]);
        }
    }
    // 如果是线性激活或其他，直接返回原值

    return output;
}

// 生成器前向传播函数
std::vector<double> generator_forward(
    const std::vector<double> &input,
    const std::vector<std::vector<double>> &weights,
    const std::vector<double> &bias,
    const std::string &activation = "relu")
{
    if (weights.size() == 0 || bias.size() == 0) {
        std::cout << "警告：生成器权重或偏置为空" << std::endl;
        std::vector<double> result(100);
        for (size_t i = 0; i < 100; ++i) {
            result[i] = 0.0;
        }
        return result;
    }

    if (input.size() != weights[0].size()) {
        std::cout << "错误：输入维度与权重不匹配" << std::endl;
        std::vector<double> result(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result[i] = 0.0;
        }
        return result;
    }

    std::vector<double> output(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        output[i] = 0.0;
    }

    // 矩阵乘法：output = weights * input + bias
    for (size_t i = 0; i < weights.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + bias[i];
    }

    // 应用激活函数
    output = apply_activation(output, activation);

    std::cout << "执行生成器层前向传播，输出维度：" << output.size() << std::endl;
    return output;
}

// 判别器前向传播函数
std::vector<double> discriminator_forward(const std::vector<double>& input,
                                 const std::vector<std::vector<double>>& weights,
                                 const std::vector<double>& bias,
                                 const std::string& activation = "sigmoid") {
    if (weights.size() == 0 || bias.size() == 0) {
        std::cout << "警告：判别器权重或偏置为空" << std::endl;
        std::vector<double> result(1);
        result[0] = 0.5;
        return result;
    }

    if (input.size() != weights[0].size()) {
        std::cout << "错误：输入维度与权重不匹配" << std::endl;
        std::vector<double> result(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            result[i] = 0.0;
        }
        return result;
    }

    std::vector<double> output(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        output[i] = 0.0;
    }

    // 矩阵乘法：output = weights * input + bias
    for (size_t i = 0; i < weights.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum + bias[i];
    }

    // 应用激活函数
    output = apply_activation(output, activation);

    std::cout << "执行判别器层前向传播，输出维度：" << output.size() << std::endl;
    return output;
}

#endif // GAN_H