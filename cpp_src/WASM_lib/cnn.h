#ifndef CNN_H
#define CNN_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>


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
    const std::string &padding_mode) // 使用字符串来指定padding模式
{
    size_t input_height = input.size();
    size_t input_width = input[0].size();
    size_t kernel_height = weight.size();
    size_t kernel_width = weight[0].size();
    size_t output_channels = weight[0][0][0].size();

    int padding_height = 0, padding_width = 0;
    if (padding_mode == "same") {
        // 计算填充以使输出维度与输入维度相同
        padding_height = (kernel_height - 1) / 2;
        padding_width = (kernel_width - 1) / 2;
    } else if (padding_mode == "valid") {
        // 无填充
        padding_height = 0;
        padding_width = 0;
    }

    size_t output_height = (input_height - kernel_height + 2 * padding_height) / stride + 1;
    size_t output_width = (input_width - kernel_width + 2 * padding_width) / stride + 1;
    std::vector<std::vector<std::vector<double>>> output(output_height, std::vector<std::vector<double>>(output_width, std::vector<double>(output_channels, 0)));

    for (size_t o = 0; o < output_channels; ++o) {
        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                double conv_sum = 0.0;
                for (size_t m = 0; m < kernel_height; ++m) {
                    for (size_t n = 0; n < kernel_width; ++n) {
                        for (size_t c = 0; c < weight[m][n].size(); ++c) {
                            int h_index = i * stride + m - padding_height;
                            int w_index = j * stride + n - padding_width;
                            if (h_index >= 0 && h_index < input_height && w_index >= 0 && w_index < input_width) {
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
std::vector<double> linear(const std::vector<double> &input)
{
    return input;  // 直接返回输入，不做任何变换
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
    else if (activation == "linear")
    {
        output = linear(output);
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


#endif 