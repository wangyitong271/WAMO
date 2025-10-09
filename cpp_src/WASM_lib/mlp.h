#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

typedef std::vector<std::vector<std::vector<double>>> Matrix3D;
typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;
typedef std::vector<std::vector<std::vector<double>>> Matrix3D;
typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

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
Matrix createMatrix(int rows, int cols)
{
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

std::vector<double> createVector(int size)
{
    std::vector<double> v(size, 0.0);
    return v;
}

Matrix3D createMatrix3D(int dim1, int dim2, int dim3)
{
    return Matrix3D(dim1, Matrix(dim2, std::vector<double>(dim3, 0.0)));
}

void setVectorData(std::vector<double> &vector, int index, double value)
{
    vector[index] = value;
}

double getVectorData(std::vector<double> &vector, int index)
{
    return vector[index];
}

double getMatrixData(Matrix &matrix, int x, int y)
{
    return matrix[x][y];
}

void setMatrixData(Matrix &matrix, int x, int y, double value)
{
    matrix[x][y] = value;
}

void setMatrix3DData(Matrix3D &matrix, int x, int y, int z, double value)
{
    matrix[x][y][z] = value;
}

double getMatrix3DData(Matrix3D &matrix, int x, int y, int z)
{
    return matrix[x][y][z];
}


#endif 