#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

class BaseLayer
{
public:
    string name;
    string type;
    virtual ~BaseLayer() = default;
};

class Conv2DLayer : public BaseLayer
{
public:
    vector<vector<vector<double>>> input;
    vector<vector<vector<vector<double>>>> weight;
    vector<double> bias;
    int stride;
    string padding;
    Conv2DLayer()
    {
        type = "Conv2D";
    }
};

class FlattenLayer : public BaseLayer
{
public:
    vector<vector<vector<double>>> input;
    FlattenLayer()
    {
        type = "Flatten";
    }
};

class MaxPooling2DLayer : public BaseLayer
{
public:
    vector<vector<vector<double>>> input;
    int pool_size;
    int strides;
    MaxPooling2DLayer()
    {
        type = "MaxPooling2D";
    }
};

class LSTMLayer : public BaseLayer
{
public:
    vector<vector<vector<double>>> input;
    vector<vector<double>> kernel;
    vector<vector<double>> recurrent_kernel;
    vector<double> bias;
    vector<vector<double>> h;
    vector<vector<double>> c;
    int units;
    LSTMLayer()
    {
        type = "LSTM";
    }
};

class DenseLayer : public BaseLayer
{
public:
    vector<double> input;
    vector<vector<double>> weight;
    vector<double> bias;
    string activation;

    DenseLayer()
    {
        type = "Dense";
    }
};

class SimpleRNNLayer : public BaseLayer
{
public:
    // (const Matrix3D &X, const Matrix &Wx, const Matrix &Wh, const Vector &brnn)
    vector<vector<vector<double>>> input;
    vector<vector<double>> Wx;
    vector<vector<double>> Wh;
    vector<double> brnn;

    string activation;

    SimpleRNNLayer()
    {
        type = "SimpleRNN";
    }
};
class GlobalAveragePooling1DLayer : public BaseLayer
{
public:
    vector<double> input;
    GlobalAveragePooling1DLayer()
    {
        type = "GlobalAveragePooling1D";
    }
};

class AttentionLayer : public BaseLayer
{
public:
    vector<vector<double>> input;

    AttentionLayer()
    {
        type = "Attention";
    }
};

class GeneratorLayer : public BaseLayer
{
public:
    vector<double> input;
    vector<double> output;
    vector<vector<double>> weight;
    vector<double> bias;
    string activation;
    int noise_dim;
    int output_dim;

    GeneratorLayer()
    {
        type = "Generator";
        activation = "relu";
        noise_dim = 100;
        output_dim = 784;
    }

    GeneratorLayer(const vector<vector<double>>& w,
                   const vector<double>& b,
                   const string& act = "relu")
        : weight(w), bias(b), activation(act)
    {
        type = "Generator";
        noise_dim = w.size() > 0 ? w[0].size() : 100;
        output_dim = w.size();
    }
};

class DiscriminatorLayer : public BaseLayer
{
public:
    vector<double> input;
    vector<double> output;
    vector<vector<double>> weight;
    vector<double> bias;
    string activation;
    int input_dim;
    int output_dim;

    DiscriminatorLayer()
    {
        type = "Discriminator";
        activation = "sigmoid";
        input_dim = 784;
        output_dim = 1;
    }

    DiscriminatorLayer(const vector<vector<double>>& w,
                       const vector<double>& b,
                       const string& act = "sigmoid")
        : weight(w), bias(b), activation(act)
    {
        type = "Discriminator";
        input_dim = w.size() > 0 ? w[0].size() : 784;
        output_dim = w.size();
    }
};

class Network
{
public:
    vector<shared_ptr<BaseLayer>> layers;

    Network()
    {
        cout << "Network created!" << endl;
    }

    ~Network()
    {
        cout << "Network destroyed!" << endl;
    }

    void showModelInfo()
    {
        for (auto &layer : layers)
        {

            if (layer->type == "Conv2D")
            {
                cout << "Conv2D Layer" << endl;
                // 将BaseLayer类型的对象转换为Conv2DLayer类型的对象
                shared_ptr<Conv2DLayer> conv2dLayer = dynamic_pointer_cast<Conv2DLayer>(layer);

                // 打印卷积层的信息
                cout << "Weights: " << endl;
                for (size_t i = 0; i < conv2dLayer->weight.size(); ++i)
                {
                    for (size_t j = 0; j < conv2dLayer->weight[i].size(); ++j)
                    {
                        for (size_t k = 0; k < conv2dLayer->weight[i][j].size(); ++k)
                        {
                            for (size_t l = 0; l < conv2dLayer->weight[i][j][k].size(); ++l)
                            {
                                cout << conv2dLayer->weight[i][j][k][l] << ", ";
                            }
                            cout << endl;
                        }
                    }
                }
                cout << "Bias: " << endl;
                for (size_t i = 0; i < conv2dLayer->bias.size(); ++i)
                {
                    cout << conv2dLayer->bias[i] << ", ";
                }

                // 展示stride和padding的信息
                cout << "Stride: " << conv2dLayer->stride << endl;

                cout << "Padding: " << conv2dLayer->padding << endl;
            }
            else if (layer->type == "Flatten")
            {
                cout << "Flatten Layer" << endl;
            }
            else if (layer->type == "SimpleRNN")
            {
                cout << "SimpleRNN Layer" << endl;
                shared_ptr<SimpleRNNLayer> simpleRNNLayer = dynamic_pointer_cast<SimpleRNNLayer>(layer);
                if (simpleRNNLayer)
                {
                    cout << "Wx: " << endl;
                    for (size_t i = 0; i < simpleRNNLayer->Wx.size(); ++i)
                    {
                        for (size_t j = 0; j < simpleRNNLayer->Wx[i].size(); ++j)
                        {
                            cout << simpleRNNLayer->Wx[i][j] << ", ";
                        }
                        cout << endl;
                    }
                    cout << "Wh: " << endl;
                    for (size_t i = 0; i < simpleRNNLayer->Wh.size(); ++i)
                    {
                        for (size_t j = 0; j < simpleRNNLayer->Wh[i].size(); ++j)
                        {
                            cout << simpleRNNLayer->Wh[i][j] << ", ";
                        }
                        cout << endl;
                    }
                    cout << "brnn: " << endl;
                    for (size_t i = 0; i < simpleRNNLayer->brnn.size(); ++i)
                    {
                        cout << simpleRNNLayer->brnn[i] << ", ";
                    }
                }
            }
            else if (layer->type == "Dense")
            {
                cout << "Dense Layer" << endl;
                shared_ptr<DenseLayer> denseLayer = dynamic_pointer_cast<DenseLayer>(layer);

                if (denseLayer)
                {
                    // 打印全连接层的信息
                    cout << "Weights: " << endl;
                    for (size_t i = 0; i < denseLayer->weight.size(); ++i)
                    {
                        for (size_t j = 0; j < denseLayer->weight[i].size(); ++j)
                        {
                            cout << denseLayer->weight[i][j] << ", ";
                        }
                        cout << endl;
                    }
                    cout << "Bias: " << endl;
                    for (size_t i = 0; i < denseLayer->bias.size(); ++i)
                    {
                        cout << denseLayer->bias[i] << ", ";
                    }

                    cout << "Activation: " << denseLayer->activation << endl;
                }
            }
            else if (layer->type == "MaxPooling2D")
            {
                cout << "MaxPooling2D Layer" << endl;
                shared_ptr<MaxPooling2DLayer> maxPooling2DLayer = dynamic_pointer_cast<MaxPooling2DLayer>(layer);
                if (maxPooling2DLayer)
                {
                    cout << "Pool Size: " << maxPooling2DLayer->pool_size << endl;
                    cout << "Strides: " << maxPooling2DLayer->strides << endl;
                }
            }
            else if (layer->type == "LSTM")
            {
                cout << "LSTM Layer" << endl;
                shared_ptr<LSTMLayer> lstmLayer = dynamic_pointer_cast<LSTMLayer>(layer);
                if (lstmLayer)
                {
                    cout << "Kernel: " << endl;
                    for (size_t i = 0; i < lstmLayer->kernel.size(); ++i)
                    {
                        for (size_t j = 0; j < lstmLayer->kernel[i].size(); ++j)
                        {
                            cout << lstmLayer->kernel[i][j] << ", ";
                        }
                        cout << endl;
                    }
                    cout << "Recurrent Kernel: " << endl;
                    for (size_t i = 0; i < lstmLayer->recurrent_kernel.size(); ++i)
                    {
                        for (size_t j = 0; j < lstmLayer->recurrent_kernel[i].size(); ++j)
                        {
                            cout << lstmLayer->recurrent_kernel[i][j] << ", ";
                        }
                        cout << endl;
                    }
                    cout << "Bias: " << endl;
                    for (size_t i = 0; i < lstmLayer->bias.size(); ++i)
                    {
                        cout << lstmLayer->bias[i] << ", ";
                    }

                    cout << "Hidden State: " << endl;
                    for (size_t i = 0; i < lstmLayer->h.size(); ++i)
                    {
                        for (size_t j = 0; j < lstmLayer->h[i].size(); ++j)
                        {
                            cout << lstmLayer->h[i][j] << ", ";
                        }
                        cout << endl;
                    }

                    cout << "Cell State: " << endl;
                    for (size_t i = 0; i < lstmLayer->c.size(); ++i)
                    {
                        for (size_t j = 0; j < lstmLayer->c[i].size(); ++j)
                        {
                            cout << lstmLayer->c[i][j] << ", ";
                        }
                        cout << endl;
                    }
                }
            }
        }
    }
};

#endif // !NETWORK_H