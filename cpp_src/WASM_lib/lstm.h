#ifndef LSTM_H
#define LSTM_H

#include <vector>

#include <cmath>
#include <stdexcept>

// 激活函数实现
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double tanh_act(double x) {
    return std::tanh(x);
}

// 向量 Sigmoid
std::vector<double> vec_sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

// 向量 Tanh
std::vector<double> vec_tanh(const std::vector<double>& x) {
    std::vector<double> result(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = tanh_act(x[i]);
    }
    return result;
}

// 向量加法
std::vector<double> vec_add(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vec_add: Vectors must be of the same size.");
    }
    std::vector<double> result(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// 向量乘法（元素级）
std::vector<double> vec_mul(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vec_mul: Vectors must be of the same size.");
    }
    std::vector<double> result(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// 矩阵-向量乘法
std::vector<double> mat_vec_dot(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    size_t mat_rows = mat.size();
    size_t mat_cols = mat[0].size();
    if (vec.size() != mat_cols) {
        throw std::invalid_argument("mat_vec_dot: Matrix column size must match vector size.");
    }
    std::vector<double> result(mat_rows, 0.0);
    for (size_t i = 0; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<double> mat_transpose_dot_vec(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    size_t rows = mat.size();
    if (rows == 0) return {};
    size_t cols = mat[0].size();
    
    if (vec.size() != rows) {
        throw std::invalid_argument("mat_transpose_dot_vec: Vector size must match matrix rows.");
    }
    
    std::vector<double> result(cols, 0.0);
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            result[j] += mat[i][j] * vec[i];
        }
    }
    return result;
}

// 自定义 LSTM 前向传播函数
std::vector<std::vector<double>> custom_lstm_last_output(
    const std::vector<std::vector<std::vector<double>>>& X,
    const std::vector<std::vector<double>>& kernel,
    const std::vector<std::vector<double>>& recurrent_kernel,
    const std::vector<double>& bias,
    std::vector<std::vector<double>>& h,
    std::vector<std::vector<double>>& c
) {
    size_t batch_size = X.size();
    if (batch_size == 0) return {};
    size_t time_steps = X[0].size();
    size_t input_dim = X[0][0].size();
    size_t units = h[0].size();

    // 检查维度一致性
    if (kernel.size() != input_dim || kernel[0].size() != 4 * units) {
        throw std::invalid_argument("custom_lstm_last_output: Kernel dimensions do not match input_dim or units.");
    }
    if (recurrent_kernel.size() != units || recurrent_kernel[0].size() != 4 * units) {
        throw std::invalid_argument("custom_lstm_last_output: Recurrent kernel dimensions do not match units.");
    }
    if (bias.size() != 4 * units) {
        throw std::invalid_argument("custom_lstm_last_output: Bias size does not match 4 * units.");
    }

    // 初始化最后一个输出，形状 (batch_size, units)
    std::vector<std::vector<double>> last_outputs(batch_size, std::vector<double>(units, 0.0));

    for (size_t t = 0; t < time_steps; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            // 获取当前时间步的输入
            const std::vector<double>& x_t = X[b][t]; // (input_dim)

            // 计算 x_t * kernel
            std::vector<double> x_kernel = mat_transpose_dot_vec(kernel, x_t); // (4 * units)

            // 计算 h_prev * recurrent_kernel
            std::vector<double> h_prev = h[b]; // (units)
            std::vector<double> h_recurrent = mat_transpose_dot_vec(recurrent_kernel, h_prev); // (4 * units)

            // 计算 z = x_kernel + h_recurrent + bias
            std::vector<double> z = vec_add(x_kernel, h_recurrent);
            z = vec_add(z, bias); // (4 * units)

            // 分割 z 为四部分：输入门、遗忘门、细胞候选、输出门
            std::vector<double> z0(z.begin(), z.begin() + units); // 输入门
            std::vector<double> z1(z.begin() + units, z.begin() + 2 * units); // 遗忘门
            std::vector<double> z2(z.begin() + 2 * units, z.begin() + 3 * units); // 细胞候选
            std::vector<double> z3(z.begin() + 3 * units, z.end()); // 输出门

            // 应用激活函数
            std::vector<double> i = vec_sigmoid(z0); // 输入门
            std::vector<double> f = vec_sigmoid(z1); // 遗忘门
            std::vector<double> g = vec_tanh(z2);    // 细胞候选
            std::vector<double> o = vec_sigmoid(z3); // 输出门

            // 更新细胞状态 c = f * c_prev + i * g
            std::vector<double> c_prev = c[b];
            std::vector<double> f_c_prev = vec_mul(f, c_prev);
            std::vector<double> i_g = vec_mul(i, g);
            std::vector<double> c_new = vec_add(f_c_prev, i_g);

            // 更新隐藏状态 h = o * tanh(c)
            std::vector<double> tanh_c = vec_tanh(c_new);
            std::vector<double> h_new = vec_mul(o, tanh_c);

            // 更新状态
            c[b] = c_new;
            h[b] = h_new;

            // 仅存储最后一个时间步的输出
            if (t == time_steps - 1) {
                last_outputs[b] = h_new;
            }
        }
    }

    return last_outputs; // 形状: (batch_size, units)
}


std::vector<std::vector<double>> custom_dense(
    const std::vector<std::vector<double>>& input,
    const std::vector<std::vector<double>>& dense_kernel,
    const std::vector<double>& dense_bias
) {
    size_t batch_size = input.size();
    size_t input_dim = input[0].size(); 
    size_t output_dim = dense_bias.size(); 

    std::vector<std::vector<double>> output(batch_size, std::vector<double>(output_dim, 0.0));

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < output_dim; ++o) {
            double sum = 0.0;
            for (size_t i = 0; i < input_dim; ++i) {
                sum += input[b][i] * dense_kernel[i][o];
            }
            sum += dense_bias[o]; // 加上偏置
            output[b][o] = sum;    // 线性激活
        }
    }

    return output;
}

#endif