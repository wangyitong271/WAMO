#ifndef POOLING_H
#define POOLING_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace DeepLearning {

/**
 * 池化算子基类
 * 用于降低特征图的空间尺寸，减少参数数量和计算量
 */
class PoolingOperator {
public:
    virtual ~PoolingOperator() = default;
    
    /**
     * 执行池化操作
     * @param input 输入特征图 [batch, height, width, channels]
     * @param kernel_size 池化核大小
     * @param stride 步长
     * @param padding 填充
     * @return 池化后的特征图
     */
    virtual std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            int kernel_size, int stride = 1, int padding = 0) = 0;
    
    /**
     * 反向传播
     * @param grad_output 输出梯度
     * @param input_shape 输入形状
     * @return 输入梯度
     */
    virtual std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output,
             const std::vector<int>& input_shape) = 0;
};

/**
 * 最大池化算子
 * 在每个池化窗口中选择最大值
 */
class MaxPooling : public PoolingOperator {
private:
    std::vector<std::vector<std::vector<std::vector<int>>>> max_indices;
    
public:
    MaxPooling() = default;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            int kernel_size, int stride = 1, int padding = 0) override {
        
        int batch_size = input.size();
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int channels = input[0][0][0].size();
        
        // 计算输出尺寸
        int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        
        // 初始化输出和索引
        std::vector<std::vector<std::vector<std::vector<float>>>> output(
            batch_size, std::vector<std::vector<std::vector<float>>>(
                output_height, std::vector<std::vector<float>>(
                    output_width, std::vector<float>(channels, 0.0f)))));
        
        max_indices = std::vector<std::vector<std::vector<std::vector<int>>>>(
            batch_size, std::vector<std::vector<std::vector<int>>>(
                output_height, std::vector<std::vector<int>>(
                    output_width, std::vector<int>(channels, 0))));
        
        // 执行最大池化
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int max_idx = 0;
                        
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int input_h = h * stride + kh - padding;
                                int input_w = w * stride + kw - padding;
                                
                                if (input_h >= 0 && input_h < input_height && 
                                    input_w >= 0 && input_w < input_width) {
                                    float val = input[b][input_h][input_w][c];
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = input_h * input_width + input_w;
                                    }
                                }
                            }
                        }
                        
                        output[b][h][w][c] = max_val;
                        max_indices[b][h][w][c] = max_idx;
                    }
                }
            }
        }
        
        return output;
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output,
             const std::vector<int>& input_shape) override {
        
        int batch_size = grad_output.size();
        int output_height = grad_output[0].size();
        int output_width = grad_output[0][0].size();
        int channels = grad_output[0][0][0].size();
        
        int input_height = input_shape[1];
        int input_width = input_shape[2];
        
        std::vector<std::vector<std::vector<std::vector<float>>>> grad_input(
            batch_size, std::vector<std::vector<std::vector<float>>>(
                input_height, std::vector<std::vector<float>>(
                    input_width, std::vector<float>(channels, 0.0f))));
        
        // 反向传播梯度
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        int max_idx = max_indices[b][h][w][c];
                        int max_h = max_idx / input_width;
                        int max_w = max_idx % input_width;
                        grad_input[b][max_h][max_w][c] += grad_output[b][h][w][c];
                    }
                }
            }
        }
        
        return grad_input;
    }
};

/**
 * 平均池化算子
 * 在每个池化窗口中计算平均值
 */
class AveragePooling : public PoolingOperator {
public:
    AveragePooling() = default;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            int kernel_size, int stride = 1, int padding = 0) override {
        
        int batch_size = input.size();
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int channels = input[0][0][0].size();
        
        // 计算输出尺寸
        int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        
        std::vector<std::vector<std::vector<std::vector<float>>>> output(
            batch_size, std::vector<std::vector<std::vector<float>>>(
                output_height, std::vector<std::vector<float>>(
                    output_width, std::vector<float>(channels, 0.0f))));
        
        // 执行平均池化
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        float sum = 0.0f;
                        int count = 0;
                        
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int input_h = h * stride + kh - padding;
                                int input_w = w * stride + kw - padding;
                                
                                if (input_h >= 0 && input_h < input_height && 
                                    input_w >= 0 && input_w < input_width) {
                                    sum += input[b][input_h][input_w][c];
                                    count++;
                                }
                            }
                        }
                        
                        output[b][h][w][c] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }
        
        return output;
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output,
             const std::vector<int>& input_shape) override {
        
        int batch_size = grad_output.size();
        int output_height = grad_output[0].size();
        int output_width = grad_output[0][0].size();
        int channels = grad_output[0][0][0].size();
        
        int input_height = input_shape[1];
        int input_width = input_shape[2];
        int kernel_size = 2; // 假设kernel_size为2
        
        std::vector<std::vector<std::vector<std::vector<float>>>> grad_input(
            batch_size, std::vector<std::vector<std::vector<float>>>(
                input_height, std::vector<std::vector<float>>(
                    input_width, std::vector<float>(channels, 0.0f))));
        
        // 反向传播梯度
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        float grad_val = grad_output[b][h][w][c] / (kernel_size * kernel_size);
                        
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int input_h = h + kh;
                                int input_w = w + kw;
                                
                                if (input_h < input_height && input_w < input_width) {
                                    grad_input[b][input_h][input_w][c] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return grad_input;
    }
};

/**
 * 全局平均池化
 * 对整个特征图进行平均池化
 */
class GlobalAveragePooling {
public:
    std::vector<std::vector<float>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) {
        
        int batch_size = input.size();
        int channels = input[0][0][0].size();
        
        std::vector<std::vector<float>> output(
            batch_size, std::vector<float>(channels, 0.0f));
        
        for (int b = 0; b < batch_size; ++b) {
            int height = input[b].size();
            int width = input[b][0].size();
            int total_pixels = height * width;
            
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        sum += input[b][h][w][c];
                    }
                }
                output[b][c] = sum / total_pixels;
            }
        }
        
        return output;
    }
};

} // namespace DeepLearning

#endif // POOLING_H
