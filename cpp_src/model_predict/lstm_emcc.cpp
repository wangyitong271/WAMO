#include "lstm.h"
#include "stock_BABA.h"
//#include "model_weights.h"
#include <iostream>
#include "read_model.h"
#include <emscripten/bind.h>


using namespace emscripten;

// 构建网络
Network network = buildNetwork();

std::vector<double> predict(int data_size)
{
    std::vector<double> x = read_data(data_size);

    // 转为 LSTM 输入格式
    std::vector<std::vector<std::vector<double>>> Xi = create_dataset(x, 7);

    vector<vector<double>> output;

    for (auto &layer : network.layers)
    {
        if (layer->type == "LSTM")
        {
            shared_ptr<LSTMLayer> lstmLayer = dynamic_pointer_cast<LSTMLayer>(layer);
            int batch_size = Xi.size();
            vector<vector<double>> h(batch_size, vector<double>(lstmLayer->units, 0.0));
            vector<vector<double>> c(batch_size, vector<double>(lstmLayer->units, 0.0));

            output = custom_lstm_last_output(Xi, lstmLayer->kernel, lstmLayer->recurrent_kernel, lstmLayer->bias, h, c);

        }
        else if (layer->type == "Dense")
        {
            shared_ptr<DenseLayer> denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            output = custom_dense(output, denseLayer->weight, denseLayer->bias);
        }
    }

    // 整理为一维数组
    std::vector<double> result;
    for (size_t i = 0; i < output.size(); ++i)
    {
        for (size_t j = 0; j < output[i].size(); ++j)
        {
            result.push_back(denormalize_data(output[i][j]));
        }
    }
    return result;
}

std::vector<double> read_dataset(int data_size)
{
    std::vector<double> x = read_data(data_size);
    return x;
}

val predict_js(int data_size) {
    std::vector<double> vec = predict(data_size);
    val jsArray = val::array();
    for(auto &d : vec) {
        jsArray.call<void>("push", d);
    }
    return jsArray;
}

val read_dataset_js(int data_size) {
    std::vector<double> vec = read_dataset(data_size);
    // denormalize_data
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = denormalize_data(vec[i]);
    }

    val jsArray = val::array();
    for(auto &d : vec) {
        jsArray.call<void>("push", d);
    }
    return jsArray;
}

// Emscripten 绑定部分
EMSCRIPTEN_BINDINGS(my_module)
{
    // 导出
    emscripten::function("test1_js", &predict_js);
    emscripten::function("read_dataset_js", &read_dataset_js);

    // 导出 std::vector 类型
    register_vector<double>("VectorDouble");
    register_vector<std::vector<double>>("MatrixDouble");
}
    