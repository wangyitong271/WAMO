#include "cnn.h"
#include "lstm.h"
#include "rnn.h"
#include "transformer.h"
#include "network.h"
#include "read_model.h"
#include <emscripten/bind.h>
#include <memory>
#include <string>
#include <map>

using namespace std;
using namespace emscripten;

// 全局网络对象
Network network;

// 模型类型枚举
enum class ModelType {
    IMAGE_CLASSIFICATION,
    GAN,
    LSTM,
    RNN,
    TRANSFORMER
};

// 当前加载的模型类型
ModelType currentModelType;

// 初始化网络
bool initializeNetwork(const string& modelType) {
    try {
        network = buildNetwork();
        
        if (modelType == "image_classification") {
            currentModelType = ModelType::IMAGE_CLASSIFICATION;
        } else if (modelType == "gan") {
            currentModelType = ModelType::GAN;
        } else if (modelType == "lstm") {
            currentModelType = ModelType::LSTM;
        } else if (modelType == "rnn") {
            currentModelType = ModelType::RNN;
        } else if (modelType == "transformer") {
            currentModelType = ModelType::TRANSFORMER;
        } else {
            return false;
        }
        
        return true;
    } catch (const exception& e) {
        cerr << "Error initializing network: " << e.what() << endl;
        return false;
    }
}

// 图像分类预测
int predictImage(const vector<vector<vector<double>>>& input) {
    if (currentModelType != ModelType::IMAGE_CLASSIFICATION) {
        cerr << "Wrong model type for image classification" << endl;
        return -1;
    }

    vector<vector<vector<double>>> processed = input;
    vector<double> flattened_input;

    for (auto& layer : network.layers) {
        if (layer->type == "Conv2D") {
            auto conv2dLayer = dynamic_pointer_cast<Conv2DLayer>(layer);
            processed = conv2d(processed, conv2dLayer->weight, conv2dLayer->bias,
                             conv2dLayer->stride, conv2dLayer->padding);
        } 
        else if (layer->type == "MaxPooling2D") {
            auto maxPooling2DLayer = dynamic_pointer_cast<MaxPooling2DLayer>(layer);
            processed = max_pooling(processed, maxPooling2DLayer->pool_size,
                                  maxPooling2DLayer->strides);
        }
        else if (layer->type == "Flatten") {
            flattened_input = flatten(processed);
        }
        else if (layer->type == "Dense") {
            auto denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            if (!flattened_input.empty()) {
                auto output = dense(flattened_input, denseLayer->weight,
                                  denseLayer->bias, denseLayer->activation);
                return my_argmax(output);
            }
        }
    }
    return -1;
}

// GAN波形生成预测
vector<double> predictWave(double wave_type) {
    if (currentModelType != ModelType::GAN) {
        cerr << "Wrong model type for GAN wave generation" << endl;
        return vector<double>();
    }

    vector<double> input = {wave_type};
    vector<double> output;

    for (auto& layer : network.layers) {
        if (layer->type == "Dense") {
            auto denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            if (output.empty()) {
                output = dense(input, denseLayer->weight, denseLayer->bias,
                             denseLayer->activation);
            } else {
                output = dense(output, denseLayer->weight, denseLayer->bias,
                             denseLayer->activation);
            }
        }
    }
    return output;
}

// LSTM预测
vector<double> predictLSTM(const vector<vector<vector<double>>>& input) {
    if (currentModelType != ModelType::LSTM) {
        cerr << "Wrong model type for LSTM prediction" << endl;
        return vector<double>();
    }

    vector<vector<double>> output;

    for (auto& layer : network.layers) {
        if (layer->type == "LSTM") {
            auto lstmLayer = dynamic_pointer_cast<LSTMLayer>(layer);
            int batch_size = input.size();
            vector<vector<double>> h(batch_size, vector<double>(lstmLayer->units, 0.0));
            vector<vector<double>> c(batch_size, vector<double>(lstmLayer->units, 0.0));
            output = custom_lstm_last_output(input, lstmLayer->kernel,
                                           lstmLayer->recurrent_kernel,
                                           lstmLayer->bias, h, c);
        }
        else if (layer->type == "Dense") {
            auto denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            output = custom_dense(output, denseLayer->weight, denseLayer->bias);
        }
    }

    // 转换为一维数组返回
    vector<double> result;
    for (const auto& row : output) {
        for (const auto& val : row) {
            result.push_back(denormalize_data(val));
        }
    }
    return result;
}

// Transformer预测
int predictTransformer(const vector<double>& input) {
    if (currentModelType != ModelType::TRANSFORMER) {
        cerr << "Wrong model type for Transformer prediction" << endl;
        return -1;
    }

    vector<double> processed = input;

    for (auto& layer : network.layers) {
        if (layer->type == "Dense") {
            auto denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            vector<double> output = dense(processed, denseLayer->weight,
                                        denseLayer->bias, "relu");
            if (output.size() == 1) {
                return static_cast<int>(output[0]);
            }
            processed = output;
        }
        else if (layer->type == "GlobalAveragePooling1D") {
            processed = global_average_pooling1d(processed);
        }
        else if (layer->type == "Attention") {
            vector<vector<double>> attention_output = attention({processed});
            if (!attention_output.empty()) {
                processed = attention_output[0];
            }
        }
    }
    return -1;
}

// RNN天气预测
vector<double> predictWeather(const vector<vector<vector<double>>>& input) {
    if (currentModelType != ModelType::RNN) {
        cerr << "Wrong model type for weather prediction" << endl;
        return vector<double>();
    }

    vector<double> flattened_input;
    Matrix output;
    
    for (auto& layer : network.layers) {
        if (layer->type == "SimpleRNN") {
            auto rnnLayer = dynamic_pointer_cast<SimpleRNNLayer>(layer);
            output = simple_rnn(input, rnnLayer->weight, rnnLayer->recurrent_weight,
                              rnnLayer->bias);
        }
        else if (layer->type == "Dense") {
            auto denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            output = dense_matrix(output, denseLayer->weight, denseLayer->bias,
                                denseLayer->activation);
        }
    }

    return matrix_to_vector(output);
}

// Emscripten绑定
EMSCRIPTEN_BINDINGS(common_model_module) {
    // 导出主要函数
    emscripten::function("initializeNetwork", &initializeNetwork);
    emscripten::function("predictImage", &predictImage);
    emscripten::function("predictWave", &predictWave);
    emscripten::function("predictLSTM", &predictLSTM);
    emscripten::function("predictTransformer", &predictTransformer);
    emscripten::function("predictWeather", &predictWeather);

    // 注册各种向量类型
    register_vector<double>("VectorDouble");
    register_vector<int>("VectorInt");
    register_vector<vector<double>>("Matrix");
    register_vector<vector<vector<double>>>("Matrix3D");
    register_vector<vector<vector<vector<double>>>>("Matrix4D");
}
