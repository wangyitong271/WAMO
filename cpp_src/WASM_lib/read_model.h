#ifndef READ_MODEL_H
#define READ_MODEL_H

#include <iostream>
#include <string>
#include <vector>
// #include "model_weights.h"
#include "model_weights_loader.h"

#include "json.hpp" // 引入 nlohmann/json 库
#include "network.h"

using json = nlohmann::json;
using namespace std;

#define DEBUG_INFO 0

// 函数：递归获取JSON数组的维度
std::vector<int> get_dimensions(const json &j)
{
    std::vector<int> dims;
    const json *current = &j;
    while (current->is_array())
    {
        dims.push_back(current->size());
        if (current->empty())
        {
            break;
        }
        current = &(*current)[0];
    }
    return dims;
}

// 函数：格式化维度为字符串，例如 [3, 3] -> "3x3"
std::string format_dimensions(const std::vector<int> &dims)
{
    std::string dim_str = "";
    for (size_t i = 0; i < dims.size(); ++i)
    {
        dim_str += std::to_string(dims[i]);
        if (i != dims.size() - 1)
        {
            dim_str += "x";
        }
    }
    return dim_str;
}

Network buildNetwork()
{
    // 运行时解密获取模型配置JSON
    string model_config_json = getDecryptedModelConfig();
    json config = json::parse(model_config_json);
    Network network;
    // 输出所有层的信息
    for (const auto &layer : config["layers"])
    {
        // 获取层的名称和类型
        string layer_name = layer.value("name", "Unnamed");
        string layer_type = layer.value("type", "Unknown");

        if (layer_type == "Conv2D")
        {
            Conv2DLayer conv2dLayer;
            conv2dLayer.name = layer_name;
            conv2dLayer.type = layer_type;
            // cout << "Layer Name: " << conv2dLayer.name << ", Type: " << conv2dLayer.type << endl;

            const json &weights = layer.value("weights", json::array());
            for (size_t i = 0; i < weights.size(); ++i)
            {
                const json &weight = weights[i];
                std::vector<int> dims = get_dimensions(weight);
                std::string dim_str = format_dimensions(dims);
                // cout << "  " << layer_type << "_weight_" << (i + 1) << ", size(" << dim_str << ")" << dims.size() << endl;
                if (dims.size() == 4)
                {
                    conv2dLayer.weight.resize(dims[0], vector<vector<vector<double>>>(dims[1], vector<vector<double>>(dims[2], vector<double>(dims[3]))));
                    for (size_t i = 0; i < dims[0]; ++i)
                    {
                        for (size_t j = 0; j < dims[1]; ++j)
                        {
                            for (size_t k = 0; k < dims[2]; ++k)
                            {
                                for (size_t l = 0; l < dims[3]; ++l)
                                {
                                    conv2dLayer.weight[i][j][k][l] = weight[i][j][k][l];
                                }
                            }
                        }
                    }
                }
                else if (dims.size() == 1)
                {
                    conv2dLayer.bias.resize(dims[0]);
                    for (size_t i = 0; i < dims[0]; ++i)
                    {
                        conv2dLayer.bias[i] = weight[i];
                    }
                }

                conv2dLayer.padding = layer.value("padding", "valid");

                const json &strides = layer.value("strides", json::array());
                if (strides.size() == 2)
                {
                    conv2dLayer.stride = strides[0];
                }
            }

            network.layers.push_back(make_shared<Conv2DLayer>(conv2dLayer));
        }
        else if (layer_type == "Dense")
        {
            DenseLayer denseLayer;
            denseLayer.name = layer_name;
            denseLayer.type = layer_type;
            // cout << "Layer Name: " << denseLayer.name << ", Type: " << denseLayer.type << endl;

            const json &weights = layer.value("weights", json::array());
            for (size_t i = 0; i < weights.size(); ++i)
            {
                const json &weight = weights[i];
                std::vector<int> dims = get_dimensions(weight);
                std::string dim_str = format_dimensions(dims);
                // cout << "  " << layer_type << "_weight_" << (i + 1) << ", size(" << dim_str << ")" << dims.size() << endl;
                if (dims.size() == 2)
                {
                    denseLayer.weight.resize(dims[0], vector<double>(dims[1]));
                    for (size_t i = 0; i < dims[0]; ++i)
                    {
                        for (size_t j = 0; j < dims[1]; ++j)
                        {
                            denseLayer.weight[i][j] = weight[i][j];
                        }
                    }
                }
                else if (dims.size() == 1)
                {
                    denseLayer.bias.resize(dims[0]);
                    for (size_t i = 0; i < dims[0]; ++i)
                    {
                        denseLayer.bias[i] = weight[i];
                    }
                }
            }

            denseLayer.activation = layer.value("activation", "linear");

            network.layers.push_back(make_shared<DenseLayer>(denseLayer));
        }
        else if (layer_type == "MaxPooling2D")
        {
            MaxPooling2DLayer maxPooling2DLayer;
            maxPooling2DLayer.name = layer_name;
            maxPooling2DLayer.type = layer_type;
            // std::cout << "Layer Name: " << maxPooling2DLayer.name << ", Type: " << maxPooling2DLayer.type << std::endl;

            const json &pool_size = layer.value("pool_size", json::array());
            if (pool_size.size() == 2)
            {
                maxPooling2DLayer.pool_size = pool_size[0];
            }

            const json &strides = layer.value("strides", json::array());
            if (strides.size() == 2)
            {
                maxPooling2DLayer.strides = strides[0];
            }

            network.layers.push_back(make_shared<MaxPooling2DLayer>(maxPooling2DLayer));
        }
        else if (layer_type == "LSTM")
        {
            LSTMLayer lstmLayer;
            lstmLayer.name = layer_name;
            lstmLayer.type = layer_type;
            // std::cout << "Layer Name: " << lstmLayer.name << ", Type: " << lstmLayer.type << std::endl;

            const json &kernel = layer.value("kernel", json::array());

            std::vector<int> dims = get_dimensions(kernel);
            std::string dim_str = format_dimensions(dims);
            // std::cout << "  " << layer_type << "_weight_" << (i + 1) << ", size(" << dim_str << ")" << dims.size() << std::endl;
            if (dims.size() == 2)
            {
                lstmLayer.kernel.resize(dims[0], vector<double>(dims[1]));
                for (size_t i = 0; i < dims[0]; ++i)
                {
                    for (size_t j = 0; j < dims[1]; ++j)
                    {
                        lstmLayer.kernel[i][j] = kernel[i][j];
                    }
                }
            }
            else if (dims.size() == 1)
            {
                // 假设kernel是一维数组那么改为1xN的二维数组
                lstmLayer.kernel.resize(1, vector<double>(dims[0]));
                for (size_t i = 0; i < dims[0]; ++i)
                {
                    lstmLayer.kernel[0][i] = kernel[i];
                }
            }

            const json &recurrent_kernel = layer.value("recurrent_kernel", json::array());

            dims = get_dimensions(recurrent_kernel);
            dim_str = format_dimensions(dims);
            // std::cout << "  " << layer_type << "_recurrent_weight_" << ", size(" << dim_str << ")" << dims.size() << std::endl;
            if (dims.size() == 2)
            {
                lstmLayer.recurrent_kernel.resize(dims[0], vector<double>(dims[1]));
                for (size_t i = 0; i < dims[0]; ++i)
                {
                    for (size_t j = 0; j < dims[1]; ++j)
                    {
                        lstmLayer.recurrent_kernel[i][j] = recurrent_kernel[i][j];
                        //cout << lstmLayer.recurrent_kernel[i][j] << ", ";
                    }
                }
            }

            const json &bias = layer.value("bias", json::array());
            dims = get_dimensions(bias);
            dim_str = format_dimensions(dims);
            // std::cout << "  " << layer_type << "_bias, size(" << dim_str << ")" << dims.size() << std::endl;
            if (dims.size() == 1)
            {
                lstmLayer.bias.resize(dims[0]);
                for (size_t i = 0; i < dims[0]; ++i)
                {
                    lstmLayer.bias[i] = bias[i];
                    // cout << lstmLayer.bias[i] << ", ";
                }
            }
            // h是隐藏状态，c是细胞状态
            const int units = layer.value("units", 0);
            lstmLayer.units = units;
            lstmLayer.h.resize(1, vector<double>(units));
            lstmLayer.c.resize(1, vector<double>(units));

            network.layers.push_back(make_shared<LSTMLayer>(lstmLayer));
        }
        else if (layer_type == "SimpleRNN")
        {
            SimpleRNNLayer simpleRNNLayer;
            simpleRNNLayer.name = layer_name;
            simpleRNNLayer.type = layer_type;

            // 打印层的名称和类型
            // std::cout << "Layer Name: " << simpleRNNLayer.name << ", Type: " << simpleRNNLayer.type << std::endl;
            
            const json &weights = layer.value("weights", json::array());
            // 打印weights的大小，大小为3，分别是Wx, Wh, brnn
            // cout << "Weights Size: " << weights.size() << endl;
            for (size_t i = 0; i < weights.size(); ++i)
            {
                const json &weight = weights[i];
                std::vector<int> dims = get_dimensions(weight);
                std::string dim_str = format_dimensions(dims);
                // 打印权重的名称和大小
                // std::cout << "  " << layer_type << "_weight_" << (i + 1) << ", size(" << dim_str << ")" << dims.size() << std::endl;
                if (dims.size() == 2)
                {
                    if (i == 0)
                    {
                        simpleRNNLayer.Wx.resize(dims[0], vector<double>(dims[1]));
                        for (size_t i = 0; i < dims[0]; ++i)
                        {
                            for (size_t j = 0; j < dims[1]; ++j)
                            {
                                simpleRNNLayer.Wx[i][j] = weight[i][j];
                            }
                        }
                    }
                    else if (i == 1)
                    {
                        simpleRNNLayer.Wh.resize(dims[0], vector<double>(dims[1]));
                        for (size_t i = 0; i < dims[0]; ++i)
                        {
                            for (size_t j = 0; j < dims[1]; ++j)
                            {
                                simpleRNNLayer.Wh[i][j] = weight[i][j];
                            }
                        }
                    }
                }
                else if (dims.size() == 1)
                {
                    simpleRNNLayer.brnn.resize(dims[0]);
                    for (size_t i = 0; i < dims[0]; ++i)
                    {
                        simpleRNNLayer.brnn[i] = weight[i];
                    }
                }
                
            }

            // 激活函数
            simpleRNNLayer.activation = layer.value("activation", "tanh");
            

            network.layers.push_back(make_shared<SimpleRNNLayer>(simpleRNNLayer));
        }
        else
        {
            BaseLayer baseLayer;
            baseLayer.name = layer_name;
            baseLayer.type = layer_type;
            // std::cout << "Layer Name: " << baseLayer.name << ", Type: " << baseLayer.type << std::endl;

            network.layers.push_back(make_shared<BaseLayer>(baseLayer));
        }

#if DEBUG_INFO
        // 遍历层的所有键
        for (auto it = layer.begin(); it != layer.end(); ++it)
        {
            std::string key = it.key();
            const json &value = it.value();

            if (key == "weights")
            {
                // 处理权重和偏置
                const json &weights = value;
                for (size_t i = 0; i < weights.size(); ++i)
                {
                    const json &weight = weights[i];
                    std::vector<int> dims = get_dimensions(weight);
                    std::string dim_str = format_dimensions(dims);
                    // 构造权重名称，例如 "Conv2D_weight_1"
                    std::string weight_name = layer_type + "_" + key + "_" + std::to_string(i + 1);
                    std::cout << "  " << weight_name << ", size(" << dim_str << ")" << dims.size() << std::endl;
                }
            }
            else if (value.is_primitive()) // 如果值是原始类型（如字符串、数字等）
            {
                std::cout << "  " << key << ": " << value << std::endl;
            }
            else
            {
                // 如果值是复杂类型（如数组或对象）
                std::cout << "  " << key << ": " << value.dump() << std::endl;
            }
        }
#endif // DEBUG_INFO

        // std::cout << std::endl; // 分隔每一层的输出
    }
    return network;
}

void showFileContent()
{
    Network network1 = buildNetwork();

    cout << "Total Layers: " << network1.layers.size() << endl;
    // 输出层的所有信息
    network1.showModelInfo();
}

#endif