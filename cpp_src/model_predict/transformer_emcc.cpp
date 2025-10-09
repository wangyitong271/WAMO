#include "transformer.h"
#include "X_data.h"
#include <iostream>
#include <vector>
#include "read_model.h"

#include <emscripten/bind.h>

using namespace std;
using namespace emscripten;

Network network = buildNetwork();

int predict(int data_no)
{
    std::vector<double> input(16, 0.0);
    for (int i = 0; i < 16; ++i)
    {
        input[i] = X_scaled[data_no][0][i];
    }

    for (auto &layer : network.layers)
    {
        if (layer->type == "Dense")
        {
            std::shared_ptr<DenseLayer> denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
            if (denseLayer)
            {
                std::vector<double> output = dense(input, denseLayer->weight, denseLayer->bias, "relu");
                if (output.size() == 1)
                {
                    int return_value = output[0];

                    return return_value;
                }
            }
        }
        else if (layer->type == "GlobalAveragePooling1D")
        {
            std::shared_ptr<GlobalAveragePooling1DLayer> globalAveragePooling1DLayer = std::dynamic_pointer_cast<GlobalAveragePooling1DLayer>(layer);
            if (globalAveragePooling1DLayer)
            {
                std::vector<double> output = global_average_pooling1d(input);
                if (output.size() == 16)
                {
                    input = output;
                }
            }
        }
        else if (layer->type == "Attention")
        {
            std::shared_ptr<AttentionLayer> attentionLayer = std::dynamic_pointer_cast<AttentionLayer>(layer);
            if (attentionLayer)
            {
                std::vector<std::vector<double>> output = attention({input});
                if (output.size() == 1 && output[0].size() == 16)
                {
                    input = output[0];
                }
            }
        }
    }
}

EMSCRIPTEN_BINDINGS(my_module)
{
    emscripten::function("predict", &predict);
}