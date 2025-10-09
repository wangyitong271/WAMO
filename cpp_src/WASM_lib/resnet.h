#ifndef RESNET_H
#define RESNET_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "cnn.h"

std::vector<std::vector<std::vector<double>>> residual_block(
    const std::vector<std::vector<std::vector<double>>> &input,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &weight1,
    const std::vector<double> &bias1,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &weight2,
    const std::vector<double> &bias2,
    int stride)
{

    auto conv1 = conv2d(input, weight1, bias1, stride, "same");
    auto relu1 = relu(conv1);

    auto conv2 = conv2d(relu1, weight2, bias2, 1, "same");
    
    // 残差连接
    std::vector<std::vector<std::vector<double>>> output = conv2;
    if(input.size() == output.size() && input[0].size() == output[0].size() 
       && input[0][0].size() == output[0][0].size()) {
        // 维度相同时直接相加
        for(size_t i = 0; i < output.size(); i++) {
            for(size_t j = 0; j < output[0].size(); j++) {
                for(size_t k = 0; k < output[0][0].size(); k++) {
                    output[i][j][k] += input[i][j][k];
                }
            }
        }
    }
    
    return relu(output);
}

std::vector<double> resnet_predict(
    const std::vector<std::vector<std::vector<double>>> &input,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &conv1_weight,
    const std::vector<double> &conv1_bias,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &res1_weight1,
    const std::vector<double> &res1_bias1,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &res1_weight2,
    const std::vector<double> &res1_bias2,
    const std::vector<std::vector<double>> &dense_weight,
    const std::vector<double> &dense_bias)
{
    // 初始卷积层
    auto x = conv2d(input, conv1_weight, conv1_bias, 1, "same");
    x = relu(x);
    x = max_pooling(x, 2, 2);
    
    // 残差块
    x = residual_block(x, res1_weight1, res1_bias1, res1_weight2, res1_bias2, 1);
    
    // 平均池化
    x = max_pooling(x, 2, 2);
    
    // 展平
    auto flattened = flatten(x);
    
    // 全连接层
    return dense(flattened, dense_weight, dense_bias, "softmax");
}

#endif
