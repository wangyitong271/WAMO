#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;


vector<vector<double>> attention(vector<vector<double>> input) {
    int seq_len = input.size();
    int d_k = input[0].size();

    // Q, K, V
    std::vector<std::vector<double>> Q = input;
    std::vector<std::vector<double>> K = input;
    std::vector<std::vector<double>> V = input;

    // 计算Attention分数
    std::vector<std::vector<double>> scores(seq_len, std::vector<double>(seq_len, 0.0));
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < d_k; ++k) {
                scores[i][j] += Q[i][k] * K[j][k];
            }
            scores[i][j] /= std::sqrt(d_k);
        }
    }

    std::vector<std::vector<double>> attention_weights = scores;

    std::vector<std::vector<double>> output(seq_len, std::vector<double>(d_k, 0.0));
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            for (int k = 0; k < d_k; ++k) {
                output[i][k] += attention_weights[i][j] * V[j][k];
            }
        }
    }

    return output;

}



vector<double> global_average_pooling1d(vector<double> input) {
    
    vector<double> output(1, 0.0);
    for (size_t i = 0; i < input.size(); ++i) {
        output[0] += input[i];
    }
    output[0] /= input.size();



    return output;
}

std::vector<double> dense(
    const std::vector<double> &input,
    const std::vector<std::vector<double>> &weights,
    const std::vector<double> &bias,
    const std::string &activation = "relu")
{
    size_t output_units = bias.size();
    std::vector<double> output(output_units, 0.0);

    for (size_t i = 0; i < output_units; ++i)
    {
        for (size_t j = 0; j < input.size(); ++j)
        {
            output[i] += input[j] * weights[j][i];
        }
        output[i] += bias[i];
    }

    if (activation == "relu")
    {
        for (double &val : output)
        {
            val = std::max(0.0, val);
        }
    }

    return output;
}


#endif