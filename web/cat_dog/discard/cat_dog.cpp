#include <emscripten/bind.h>
#include <vector>
#include <iostream>
#include <algorithm> 
#include <numeric>   
#include "model_weights.h"
using namespace emscripten;


typedef std::vector<std::vector<std::vector<double>>> Image;


Image createIamge(int height, int width, int channels)
{
    return Image(height, std::vector<std::vector<double>>(width, std::vector<double>(channels, 0.0)));
}

void setIamgeData(Image &image, int x, int y, int z, double value)
{
    image[x][y][z] = value;
}

double getImageData(Image &image, int x, int y, int z)
{
    return image[x][y][z];
}

std::vector<std::vector<std::vector<double>>> relu(const std::vector<std::vector<std::vector<double>>> &input)
{
    auto output = input;
    for (auto &matrix : output)
    {
        for (auto &row : matrix)
        {
            std::transform(row.begin(), row.end(), row.begin(), [](double x)
                           { return std::max(0.0, x); });
        }
    }
    return output;
}

std::vector<double> softmax(const std::vector<double> &input)
{
    std::vector<double> output(input.size());
    double max_val = *std::max_element(input.begin(), input.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }
    for (double &val : output)
    {
        val /= sum_exp;
    }
    return output;
}

std::vector<std::vector<std::vector<double>>> conv2d(
    const std::vector<std::vector<std::vector<double>>> &input,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &weight,
    const std::vector<double> &bias,
    int stride,
    int padding)
{
    // 确定输出维度
    size_t input_height = input.size();
    size_t input_width = input[0].size();
    size_t kernel_height = weight.size();
    size_t kernel_width = weight[0].size();
    size_t output_channels = weight[0][0][0].size();

    size_t output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    size_t output_width = (input_width - kernel_width + 2 * padding) / stride + 1;


    std::vector<std::vector<std::vector<double>>> output(output_height, std::vector<std::vector<double>>(output_width, std::vector<double>(output_channels, 0)));


    for (size_t o = 0; o < output_channels; ++o)
    {
        for (size_t i = 0; i < output_height; ++i)
        {
            for (size_t j = 0; j < output_width; ++j)
            {
                double conv_sum = 0.0;
                for (size_t m = 0; m < kernel_height; ++m)
                {
                    for (size_t n = 0; n < kernel_width; ++n)
                    {
                        for (size_t c = 0; c < weight[m][n].size(); ++c)
                        {
                            size_t h_index = i * stride + m - padding;
                            size_t w_index = j * stride + n - padding;
                            if (h_index < input_height && w_index < input_width)
                            {
                                conv_sum += input[h_index][w_index][c] * weight[m][n][c][o];
                            }
                        }
                    }
                }
                output[i][j][o] = conv_sum + bias[o];
            }
        }
    }

    return relu(output); 
}

std::vector<std::vector<std::vector<double>>> max_pooling(
    const std::vector<std::vector<std::vector<double>>> &input,
    int pool_size,
    int stride)
{
    size_t input_height = input.size();
    size_t input_width = input[0].size();
    size_t input_channels = input[0][0].size();

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(output_height, std::vector<std::vector<double>>(output_width, std::vector<double>(input_channels, 0)));

    for (size_t c = 0; c < input_channels; ++c)
    {
        for (size_t i = 0; i < output_height; ++i)
        {
            for (size_t j = 0; j < output_width; ++j)
            {
                double max_val = std::numeric_limits<double>::lowest();
                for (int m = 0; m < pool_size; ++m)
                {
                    for (int n = 0; n < pool_size; ++n)
                    {
                        size_t h_index = i * stride + m;
                        size_t w_index = j * stride + n;
                        if (h_index < input_height && w_index < input_width)
                        {
                            max_val = std::max(max_val, input[h_index][w_index][c]);
                        }
                    }
                }
                output[i][j][c] = max_val;
            }
        }
    }
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
    else if (activation == "softmax")
    {
        output = softmax(output);
    }

    return output;
}

std::vector<double> flatten(const std::vector<std::vector<std::vector<double>>> &input)
{
    std::vector<double> output;
    for (const auto &matrix : input)
    {
        for (const auto &row : matrix)
        {
            for (double val : row)
            {
                output.push_back(val);
            }
        }
    }
    return output;
}

int my_argmax(const std::vector<double> &input)
{
    return std::distance(input.begin(), std::max_element(input.begin(), input.end()));
}


std::vector<std::vector<std::vector<double>>> convertImage() {
    std::vector<std::vector<std::vector<double>>> vec(200, std::vector<std::vector<double>>(200, std::vector<double>(3)));
    for (int i = 0; i < 200; ++i) {
        for (int j = 0; j < 200; ++j) {
            for (int k = 0; k < 3; ++k) {
                vec[i][j][k] = image0[i][j][k];
            }
        }
    }
    return vec;
}

std::vector<std::vector<std::vector<std::vector<double>>>> convertLayer1Weights() {
    std::vector<std::vector<std::vector<std::vector<double>>>> vec(3, std::vector<std::vector<std::vector<double>>>(3, std::vector<std::vector<double>>(3, std::vector<double>(4))));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 4; ++l) {
                    vec[i][j][k][l] = layer1_weights[i][j][k][l];
                }
            }
        }
    }
    return vec;
}

std::vector<std::vector<double>> convertLayer2Weights() {
    std::vector<std::vector<double>> vec(9604, std::vector<double>(2));
    for (int i = 0; i < 9604; ++i) {
        for (int j = 0; j < 2; ++j) {
            vec[i][j] = layer2_weights[i][j];
        }
    }
    return vec;
}

std::vector<double> convertLayer1Biases() {
    return std::vector<double>(std::begin(layer1_biases), std::end(layer1_biases));
}

std::vector<double> convertLayer2Biases() {
    return std::vector<double>(std::begin(layer2_biases), std::end(layer2_biases));
}


int predict(const std::vector<std::vector<std::vector<double>>> &input)
{
    std::vector<std::vector<std::vector<std::vector<double>>>> weight1 = convertLayer1Weights();
    std::vector<double> bias1 = convertLayer1Biases();
    int stride1 = 1;
    int padding1 = 0;

    auto output1 = conv2d(input, weight1, bias1, stride1, padding1);

    std::vector<std::vector<double>> weight2 = convertLayer2Weights();
    std::vector<double> bias2 = convertLayer2Biases();

    auto output2 = max_pooling(output1, 4, 4);

    auto output3 = flatten(output2);

    auto output4 = dense(output3, weight2, bias2, "softmax");

    // 打印输出
    for (size_t i = 0; i < output4.size(); ++i)
    {
        std::cout << "Output " << i << ": " << output4[i] << std::endl;
    }

    return my_argmax(output4);
}

// 释放内存
void freeImage(Image &image)
{
    image.clear();
}

void test_predict()
{
    std::vector<std::vector<std::vector<double>>> input = convertImage();
    int result = predict(input);
    std::cout << "Predicted: " << result << std::endl;
}

EMSCRIPTEN_BINDINGS(my_module)
{

    function("test_predict", &test_predict);
    function("predict", &predict);
    function("createIamge", &createIamge);
    function("setIamgeData", &setIamgeData);
    function("getImageData", &getImageData);
    function("freeImage", &freeImage);

    register_vector<std::vector<std::vector<double>>>("Matrix");
    register_vector<std::vector<double>>("Vector");
    register_vector<int>("VectorInt");
    register_vector<double>("VectorDouble");
    register_vector<std::vector<std::vector<std::vector<std::vector<double>>>>>("Matrix4D");
    register_vector<std::vector<std::vector<std::vector<double>>>>("Matrix3D");
}