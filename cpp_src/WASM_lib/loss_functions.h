#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace DeepLearning {

/**
 * 损失函数基类
 * 用于在训练过程中评估模型的性能
 */
class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    /**
     * 计算损失值
     * @param predictions 模型预测值
     * @param targets 真实标签
     * @return 损失值
     */
    virtual float compute_loss(const std::vector<std::vector<float>>& predictions,
                              const std::vector<std::vector<float>>& targets) = 0;
    
    /**
     * 计算梯度
     * @param predictions 模型预测值
     * @param targets 真实标签
     * @return 梯度
     */
    virtual std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) = 0;
};

/**
 * 交叉熵损失 (Cross-Entropy Loss)
 * 常用于分类任务
 */
class CrossEntropyLoss : public LossFunction {
private:
    float epsilon = 1e-7f;  // 防止log(0)的小常数
    
public:
    CrossEntropyLoss(float eps = 1e-7f) : epsilon(eps) {}
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_classes = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            // 应用softmax
            std::vector<float> softmax_probs = softmax(predictions[i]);
            
            // 计算交叉熵
            float sample_loss = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                if (targets[i][j] > 0) {  // 只对正标签计算损失
                    sample_loss -= targets[i][j] * std::log(softmax_probs[j] + epsilon);
                }
            }
            total_loss += sample_loss;
        }
        
        return total_loss / batch_size;
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_classes = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_classes, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> softmax_probs = softmax(predictions[i]);
            
            for (int j = 0; j < num_classes; ++j) {
                gradients[i][j] = softmax_probs[j] - targets[i][j];
            }
        }
        
        return gradients;
    }
    
private:
    std::vector<float> softmax(const std::vector<float>& logits) {
        int size = logits.size();
        std::vector<float> exp_values(size);
        
        // 计算指数值并找到最大值（数值稳定性）
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        
        for (int i = 0; i < size; ++i) {
            exp_values[i] = std::exp(logits[i] - max_val);
            sum_exp += exp_values[i];
        }
        
        // 归一化
        for (int i = 0; i < size; ++i) {
            exp_values[i] /= sum_exp;
        }
        
        return exp_values;
    }
};

/**
 * 均方误差损失 (Mean Squared Error)
 * 常用于回归任务
 */
class MSELoss : public LossFunction {
public:
    MSELoss() = default;
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float diff = predictions[i][j] - targets[i][j];
                total_loss += diff * diff;
            }
        }
        
        return total_loss / (batch_size * num_features);
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_features, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                gradients[i][j] = 2.0f * (predictions[i][j] - targets[i][j]) / (batch_size * num_features);
            }
        }
        
        return gradients;
    }
};

/**
 * 二元交叉熵损失 (Binary Cross-Entropy Loss)
 * 用于二分类任务
 */
class BinaryCrossEntropyLoss : public LossFunction {
private:
    float epsilon = 1e-7f;
    
public:
    BinaryCrossEntropyLoss(float eps = 1e-7f) : epsilon(eps) {}
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float pred = std::max(epsilon, std::min(1.0f - epsilon, predictions[i][j]));
                float target = targets[i][j];
                
                total_loss -= target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred);
            }
        }
        
        return total_loss / (batch_size * num_features);
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_features, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float pred = std::max(epsilon, std::min(1.0f - epsilon, predictions[i][j]));
                float target = targets[i][j];
                
                gradients[i][j] = (pred - target) / (pred * (1.0f - pred)) / (batch_size * num_features);
            }
        }
        
        return gradients;
    }
};

/**
 * 平滑L1损失 (Smooth L1 Loss)
 * Huber损失的变种，对异常值更鲁棒
 */
class SmoothL1Loss : public LossFunction {
private:
    float beta = 1.0f;  // 阈值参数
    
public:
    SmoothL1Loss(float b = 1.0f) : beta(b) {}
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float diff = std::abs(predictions[i][j] - targets[i][j]);
                
                if (diff < beta) {
                    total_loss += 0.5f * diff * diff / beta;
                } else {
                    total_loss += diff - 0.5f * beta;
                }
            }
        }
        
        return total_loss / (batch_size * num_features);
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_features, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float diff = predictions[i][j] - targets[i][j];
                float abs_diff = std::abs(diff);
                
                if (abs_diff < beta) {
                    gradients[i][j] = diff / beta / (batch_size * num_features);
                } else {
                    gradients[i][j] = (diff > 0 ? 1.0f : -1.0f) / (batch_size * num_features);
                }
            }
        }
        
        return gradients;
    }
};

/**
 * KL散度损失 (Kullback-Leibler Divergence Loss)
 * 用于分布匹配任务
 */
class KLDivLoss : public LossFunction {
private:
    float epsilon = 1e-7f;
    
public:
    KLDivLoss(float eps = 1e-7f) : epsilon(eps) {}
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float pred = std::max(epsilon, predictions[i][j]);
                float target = std::max(epsilon, targets[i][j]);
                
                total_loss += target * std::log(target / pred);
            }
        }
        
        return total_loss / (batch_size * num_features);
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_features = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_features, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_features; ++j) {
                float pred = std::max(epsilon, predictions[i][j]);
                gradients[i][j] = -targets[i][j] / pred / (batch_size * num_features);
            }
        }
        
        return gradients;
    }
};

/**
 * 焦点损失 (Focal Loss)
 * 用于处理类别不平衡问题
 */
class FocalLoss : public LossFunction {
private:
    float alpha = 1.0f;    // 权重参数
    float gamma = 2.0f;    // 聚焦参数
    float epsilon = 1e-7f;
    
public:
    FocalLoss(float a = 1.0f, float g = 2.0f, float eps = 1e-7f) 
        : alpha(a), gamma(g), epsilon(eps) {}
    
    float compute_loss(const std::vector<std::vector<float>>& predictions,
                      const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_classes = predictions[0].size();
        float total_loss = 0.0f;
        
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> softmax_probs = softmax(predictions[i]);
            
            for (int j = 0; j < num_classes; ++j) {
                float p = std::max(epsilon, softmax_probs[j]);
                float target = targets[i][j];
                
                float ce_loss = -target * std::log(p);
                float weight = std::pow(1.0f - p, gamma);
                
                total_loss += alpha * weight * ce_loss;
            }
        }
        
        return total_loss / batch_size;
    }
    
    std::vector<std::vector<float>> compute_gradient(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets) override {
        
        int batch_size = predictions.size();
        int num_classes = predictions[0].size();
        std::vector<std::vector<float>> gradients(batch_size, std::vector<float>(num_classes, 0.0f));
        
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> softmax_probs = softmax(predictions[i]);
            
            for (int j = 0; j < num_classes; ++j) {
                float p = std::max(epsilon, softmax_probs[j]);
                float target = targets[i][j];
                
                float weight = std::pow(1.0f - p, gamma);
                float focal_weight = alpha * weight * (gamma * p * std::log(p) + p - target);
                
                gradients[i][j] = focal_weight / batch_size;
            }
        }
        
        return gradients;
    }
    
private:
    std::vector<float> softmax(const std::vector<float>& logits) {
        int size = logits.size();
        std::vector<float> exp_values(size);
        
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        
        for (int i = 0; i < size; ++i) {
            exp_values[i] = std::exp(logits[i] - max_val);
            sum_exp += exp_values[i];
        }
        
        for (int i = 0; i < size; ++i) {
            exp_values[i] /= sum_exp;
        }
        
        return exp_values;
    }
};

} // namespace DeepLearning

#endif // LOSS_FUNCTIONS_H
