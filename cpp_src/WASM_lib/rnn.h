#ifndef RNN_H
#define RNN_H

#include <vector>
#include <iostream>
#include <string>
#include <optional>
#include <cmath>

typedef std::vector<std::vector<std::vector<double>>> Matrix3D;
typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

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

double getMatrix3DData (Matrix3D &matrix, int x, int y, int z)
{
    return matrix[x][y][z];
}

void setMatrix3DData(Matrix3D &matrix, int x, int y, int z, double value)
{
    matrix[x][y][z] = value;
}

Matrix add_bias(const Matrix &matrix, const std::vector<double> &bias)
{
    Matrix result = matrix;
    for (size_t i = 0; i < matrix.size(); i++)
    {
        for (size_t j = 0; j < matrix[0].size(); j++)
        {
            result[i][j] += bias[j];
        }
    }
    return result;
}

Matrix matrix_multiply(const Matrix &A, const Matrix &B)
{
    std::size_t a_rows = A.size();
    std::size_t a_cols = A[0].size();
    std::size_t b_cols = B[0].size();
    Matrix result(a_rows, std::vector<double>(b_cols, 0.0));

    for (std::size_t i = 0; i < a_rows; ++i)
    {
        for (std::size_t j = 0; j < b_cols; ++j)
        {
            for (std::size_t k = 0; k < a_cols; ++k)
            {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

double my_tanh(double x)
{
    return std::tanh(x);
}

Matrix apply_tanh(const Matrix &M)
{
    Matrix result = M;
    for (auto &row : result)
    {
        for (auto &element : row)
        {
            element = my_tanh(element);
        }
    }
    return result;
}

Matrix apply_linear(const Matrix &M)
{
    return M;
}

Matrix dense_forward(const Matrix &h, const Matrix &Wd, const std::vector<double> &bd)
{
    return apply_linear(add_bias(matrix_multiply(h, Wd), bd));
}


Matrix simple_rnn_forward(const Matrix3D &X, const Matrix &Wx, const Matrix &Wh, const Vector &brnn)
{
    std::size_t batch_size = X.size();
    std::size_t hidden_size = Wh.size();
    Matrix h_next(batch_size, std::vector<double>(hidden_size, 0));
    for (std::size_t t = 0; t < X[0].size(); ++t)
    {
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            for (std::size_t j = 0; j < hidden_size; ++j)
            {
                double sum = brnn[j];
                for (std::size_t k = 0; k < X[0][0].size(); ++k)
                {
                    sum += X[i][t][k] * Wx[k][j];
                }
                for (std::size_t k = 0; k < hidden_size; ++k)
                {
                    sum += h_next[i][k] * Wh[k][j];
                }
                h_next[i][j] = my_tanh(sum);
            }
        }
    }
    return h_next;
}

Vector matrix_to_vector(const Matrix &matrix)
{
    Vector result;
    for (const auto &row : matrix)
    {
        for (const auto &val : row)
        {
            result.push_back(val);
        }
    }
    return result;
}

#endif 