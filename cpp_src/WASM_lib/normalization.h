#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace DeepLearning {

/**
 * 归一化算子基类
 * 用于调整神经网络中间层的输出，提高训练的稳定性和性能
 */
class NormalizationOperator {
public:
    virtual ~NormalizationOperator() = default;
    
    /**
     * 前向传播
     * @param input 输入数据
     * @param training 是否为训练模式
     * @return 归一化后的输出
     */
    virtual std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            bool training = true) = 0;
    
    /**
     * 反向传播
     * @param grad_output 输出梯度
     * @return 输入梯度
     */
    virtual std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output) = 0;
};

/**
 * 批量归一化 (Batch Normalization)
 * 对每个特征通道进行归一化
 */
class BatchNormalization : public NormalizationOperator {
private:
    std::vector<float> gamma;      // 缩放参数
    std::vector<float> beta;       // 偏移参数
    std::vector<float> running_mean;   // 运行时的均值
    std::vector<float> running_var;    // 运行时的方差
    std::vector<float> mean;       // 当前批次的均值
    std::vector<float> var;        // 当前批次的方差
    std::vector<float> x_normalized;   // 归一化后的输入
    float momentum = 0.9f;         // 动量参数
    float epsilon = 1e-5f;         // 防止除零的小常数
    
public:
    BatchNormalization(int num_features) {
        gamma.resize(num_features, 1.0f);
        beta.resize(num_features, 0.0f);
        running_mean.resize(num_features, 0.0f);
        running_var.resize(num_features, 1.0f);
        mean.resize(num_features, 0.0f);
        var.resize(num_features, 1.0f);
        x_normalized.resize(num_features, 0.0f);
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            bool training = true) override {
        
        int batch_size = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        int channels = input[0][0][0].size();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> output = input;
        
        if (training) {
            // 计算当前批次的均值和方差
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            sum += input[b][h][w][c];
                        }
                    }
                }
                mean[c] = sum / (batch_size * height * width);
                
                float sum_sq = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float diff = input[b][h][w][c] - mean[c];
                            sum_sq += diff * diff;
                        }
                    }
                }
                var[c] = sum_sq / (batch_size * height * width);
                
                // 更新运行时的均值和方差
                running_mean[c] = momentum * running_mean[c] + (1 - momentum) * mean[c];
                running_var[c] = momentum * running_var[c] + (1 - momentum) * var[c];
            }
        } else {
            // 推理时使用运行时的均值和方差
            mean = running_mean;
            var = running_var;
        }
        
        // 归一化和缩放
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        x_normalized[c] = (input[b][h][w][c] - mean[c]) / std::sqrt(var[c] + epsilon);
                        output[b][h][w][c] = gamma[c] * x_normalized[c] + beta[c];
                    }
                }
            }
        }
        
        return output;
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output) override {
        
        int batch_size = grad_output.size();
        int height = grad_output[0].size();
        int width = grad_output[0][0].size();
        int channels = grad_output[0][0][0].size();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> grad_input = grad_output;
        
        // 计算输入梯度
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        grad_input[b][h][w][c] = gamma[c] * grad_output[b][h][w][c] / std::sqrt(var[c] + epsilon);
                    }
                }
            }
        }
        
        return grad_input;
    }
    
    // 设置参数
    void set_gamma(const std::vector<float>& g) { gamma = g; }
    void set_beta(const std::vector<float>& b) { beta = b; }
    void set_momentum(float m) { momentum = m; }
    void set_epsilon(float e) { epsilon = e; }
};

/**
 * 层归一化 (Layer Normalization)
 * 对每个样本的所有特征进行归一化
 */
class LayerNormalization : public NormalizationOperator {
private:
    std::vector<float> gamma;      // 缩放参数
    std::vector<float> beta;       // 偏移参数
    float epsilon = 1e-5f;         // 防止除零的小常数
    
public:
    LayerNormalization(int num_features) {
        gamma.resize(num_features, 1.0f);
        beta.resize(num_features, 0.0f);
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input,
            bool training = true) override {
        
        int batch_size = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        int channels = input[0][0][0].size();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> output = input;
        
        // 对每个样本进行归一化
        for (int b = 0; b < batch_size; ++b) {
            // 计算当前样本的均值和方差
            float sum = 0.0f;
            int total_elements = height * width * channels;
            
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        sum += input[b][h][w][c];
                    }
                }
            }
            float mean = sum / total_elements;
            
            float sum_sq = 0.0f;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        float diff = input[b][h][w][c] - mean;
                        sum_sq += diff * diff;
                    }
                }
            }
            float var = sum_sq / total_elements;
            
            // 归一化和缩放
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        float normalized = (input[b][h][w][c] - mean) / std::sqrt(var + epsilon);
                        output[b][h][w][c] = gamma[c] * normalized + beta[c];
                    }
                }
            }
        }
        
        return output;
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>>
    backward(const std::vector<std::vector<std::vector<std::vector<float>>>>& grad_output) override {
        
        int batch_size = grad_output.size();
        int height = grad_output[0].size();
        int width = grad_output[0][0].size();
        int channels = grad_output[0][0][0].size();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> grad_input = grad_output;
        
        // 简化的反向传播
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < channels; ++c) {
                        grad_input[b][h][w][c] = gamma[c] * grad_output[b][h][w][c];
                    }
                }
            }
        }
        
        return grad_input;
    }
    
    // 设置参数
    void set_gamma(const std::vector<float>& g) { gamma = g; }
    void set_beta(const std::vector<float>& b) { beta = b; }
    void set_epsilon(float e) { epsilon = e; }
};

/**
 * 实例归一化 (Instance Normalization)
 * 对每个样本的每个通道分别进行归一化
 */
class InstanceNormalization {
private:
    std::vector<float> gamma;
    std::vector<float> beta;
    float epsilon = 1e-5f;
    
public:
    InstanceNormalization(int num_features) {
        gamma.resize(num_features, 1.0f);
        beta.resize(num_features, 0.0f);
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) {
        
        int batch_size = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        int channels = input[0][0][0].size();
        
        std::vector<std::vector<std::vector<std::vector<float>>>> output = input;
        
        // 对每个样本的每个通道进行归一化
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                // 计算当前通道的均值和方差
                float sum = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        sum += input[b][h][w][c];
                    }
                }
                float mean = sum / (height * width);
                
                float sum_sq = 0.0f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float diff = input[b][h][w][c] - mean;
                        sum_sq += diff * diff;
                    }
                }
                float var = sum_sq / (height * width);
                
                // 归一化和缩放
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float normalized = (input[b][h][w][c] - mean) / std::sqrt(var + epsilon);
                        output[b][h][w][c] = gamma[c] * normalized + beta[c];
                    }
                }
            }
        }
        
        return output;
    }
};

/**
 * 组归一化 (Group Normalization)
 * 将通道分组后进行归一化
 */
class GroupNormalization {
private:
    std::vector<float> gamma;
    std::vector<float> beta;
    int num_groups;
    float epsilon = 1e-5f;
    
public:
    GroupNormalization(int num_features, int groups) : num_groups(groups) {
        gamma.resize(num_features, 1.0f);
        beta.resize(num_features, 0.0f);
    }
    
    std::vector<std::vector<std::vector<std::vector<float>>>> 
    forward(const std::vector<std::vector<std::vector<std::vector<float>>>>& input) {
        
        int batch_size = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        int channels = input[0][0][0].size();
        int channels_per_group = channels / num_groups;
        
        std::vector<std::vector<std::vector<std::vector<float>>>> output = input;
        
        // 对每个样本的每个组进行归一化
        for (int b = 0; b < batch_size; ++b) {
            for (int g = 0; g < num_groups; ++g) {
                // 计算当前组的均值和方差
                float sum = 0.0f;
                int group_size = channels_per_group * height * width;
                
                for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            sum += input[b][h][w][c];
                        }
                    }
                }
                float mean = sum / group_size;
                
                float sum_sq = 0.0f;
                for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float diff = input[b][h][w][c] - mean;
                            sum_sq += diff * diff;
                        }
                    }
                }
                float var = sum_sq / group_size;
                
                // 归一化和缩放
                for (int c = g * channels_per_group; c < (g + 1) * channels_per_group; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float normalized = (input[b][h][w][c] - mean) / std::sqrt(var + epsilon);
                            output[b][h][w][c] = gamma[c] * normalized + beta[c];
                        }
                    }
                }
            }
        }
        
        return output;
    }
};

} // namespace DeepLearning

#endif // NORMALIZATION_H
