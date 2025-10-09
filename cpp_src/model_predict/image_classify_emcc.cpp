#include "cnn.h"
//#include "model_weights.h"
#include "network.h"
#include "read_model.h"
#include <emscripten/bind.h>


using namespace std;
using namespace emscripten;

Network network = buildNetwork();

int predict(vector<vector<vector<double>>> &input) {

    cout << "Total Layers: " << network.layers.size() << endl;

    vector<double> flattened_input;

    for (auto &layer : network.layers) {
        if (layer->type == "Conv2D") {
            shared_ptr<Conv2DLayer> conv2dLayer = dynamic_pointer_cast<Conv2DLayer>(layer);
            input = conv2d(input, conv2dLayer->weight, conv2dLayer->bias, conv2dLayer->stride, conv2dLayer->padding);
        } else if (layer->type == "MaxPooling2D") {
            shared_ptr<MaxPooling2DLayer> maxPooling2DLayer = dynamic_pointer_cast<MaxPooling2DLayer>(layer);
            input = max_pooling(input, maxPooling2DLayer->pool_size, maxPooling2DLayer->strides);
        } else if (layer->type == "Flatten") {
            flattened_input = flatten(input);
        } else if (layer->type == "Dense") {
            shared_ptr<DenseLayer> denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            if (!flattened_input.empty()) {
                auto output = dense(flattened_input, denseLayer->weight, denseLayer->bias, denseLayer->activation);
                return my_argmax(output);
            } else {
                cerr << "Error: Flatten layer output is empty." << endl;
                return -1;
            }
        }
    }
    return -1;
}


EMSCRIPTEN_BINDINGS(my_module)
{
    emscripten::function("predict", &predict);
    emscripten::function("createIamge", &createIamge);
    emscripten::function("setIamgeData", &setIamgeData);
    emscripten::function("getImageData", &getImageData);

    register_vector<std::vector<std::vector<double>>>("Matrix");
    register_vector<std::vector<double>>("Vector");
    register_vector<int>("VectorInt");
    register_vector<double>("VectorDouble");
    register_vector<std::vector<std::vector<std::vector<std::vector<double>>>>>("Matrix4D");
    register_vector<std::vector<std::vector<std::vector<double>>>>("Matrix3D");
}