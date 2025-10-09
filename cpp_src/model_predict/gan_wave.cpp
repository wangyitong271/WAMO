#include "gan.h"
#include "cnn.h"
//#include "model_weights.h"
#include "network.h"
#include "read_model.h"
#include <emscripten/bind.h>


using namespace std;
using namespace emscripten;

Network network = buildNetwork();

vector<double> predict(double wave_type) {
    // 将输入转换为vector
    vector<double> input = {wave_type};
    vector<double> output;

    cout << "Total Layers: " << network.layers.size() << endl;

    for (auto &layer : network.layers) {
        if (layer->type == "Dense") {
            shared_ptr<DenseLayer> denseLayer = dynamic_pointer_cast<DenseLayer>(layer);
            if (output.empty()) {
                output = dense(input, denseLayer->weight, denseLayer->bias, denseLayer->activation);
            } else {
                output = dense(output, denseLayer->weight, denseLayer->bias, denseLayer->activation);
            }
        } else if (layer->type == "Generator") {
            shared_ptr<GeneratorLayer> genLayer = dynamic_pointer_cast<GeneratorLayer>(layer);
            if (output.empty()) {
                output = generator_forward(input, genLayer->weight, genLayer->bias, genLayer->activation);
            } else {
                output = generator_forward(output, genLayer->weight, genLayer->bias, genLayer->activation);
            }
        } else if (layer->type == "Discriminator") {
            shared_ptr<DiscriminatorLayer> discLayer = dynamic_pointer_cast<DiscriminatorLayer>(layer);
            if (output.empty()) {
                output = discriminator_forward(input, discLayer->weight, discLayer->bias, discLayer->activation);
            } else {
                output = discriminator_forward(output, discLayer->weight, discLayer->bias, discLayer->activation);
            }
        }
    }
    for (int i = 0; i < output.size(); i++) {
        cout << output[i] << " ";
    }
    cout << endl;
    return output;  // 返回100个波形点
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