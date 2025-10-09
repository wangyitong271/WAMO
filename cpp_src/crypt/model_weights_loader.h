#ifndef MODEL_WEIGHTS_LOADER_H
#define MODEL_WEIGHTS_LOADER_H

#include <string>
#include <iostream>
#include "feistel_crypto.h"
#include "RSAKeyStore.h"
#include "rsa.h"
#include "model_weights_cry.h"

// 运行时解密模型权重数据的完整解决方案
class ModelWeightsLoader {
private:
    static std::string cached_json;
    static bool initialized;
    
    // 从RSA加密的key中解密获取K1和K2，然后解密JSON数据
    static std::string decryptModelConfig() {
        // 从RSA加密的key中解密获取原始密钥字符串
        std::string decrypted_key = decrypt(rsa_ciphertext, keys.d, keys.n);
        
        // 从解密的key中提取K1和K2（使用与encode_rsa.cpp相同的算法）
        uint32_t hash = 0;
        for (char c : decrypted_key) {
            hash = hash * 31 + static_cast<uint32_t>(c);
        }
        
        uint32_t K1 = (hash & 0xFFF) % 4096;        // 取低12位作为K1
        uint32_t K2 = ((hash >> 12) & 0x7FF) % 2048; // 取13-23位作为K2
        
        // 使用K1和K2创建解密器
        FeistelCrypto crypto(K1, K2);
        
        // 解密model_weights_cry.h中的加密JSON数据
        return crypto.decryptJsonString(model_config_json_encrypted);
    }
    
public:
    // 获取解密后的模型配置JSON（运行时解密）
    static const std::string& getModelConfig() {
        if (!initialized) {
            cached_json = decryptModelConfig();
            initialized = true;
        }
        return cached_json;
    }
    
    // 强制重新解密（清除缓存）
    static void refreshCache() {
        initialized = false;
        cached_json.clear();
    }
    
    // 验证解密是否成功
    static bool verify() {
        const std::string& json = getModelConfig();
        return !json.empty() && 
               json.find("\"layers\"") != std::string::npos &&
               json.find("\"weights\"") != std::string::npos;
    }
    
    // 调试信息
    static void printInfo() {
        const std::string& json = getModelConfig();
        std::cout << "模型配置JSON长度: " << json.length() << " 字符\n";
        std::cout << "前100个字符预览:\n" << json.substr(0, 100) << "...\n";
        std::cout << "验证状态: " << (verify() ? "成功" : "失败") << "\n";
    }
};

// 静态成员初始化
std::string ModelWeightsLoader::cached_json = "";
bool ModelWeightsLoader::initialized = false;

// 运行时解密函数 - 每次调用都会重新解密
inline std::string getDecryptedModelConfig() {
    // 从RSA加密的key中解密获取原始密钥字符串
    std::string decrypted_key = decrypt(rsa_ciphertext, keys.d, keys.n);
    
    // 从解密的key中提取K1和K2
    uint32_t hash = 0;
    for (char c : decrypted_key) {
        hash = hash * 31 + static_cast<uint32_t>(c);
    }
    
    uint32_t K1 = (hash & 0xFFF) % 4096;
    uint32_t K2 = ((hash >> 12) & 0x7FF) % 2048;
    
    // 使用K1和K2创建解密器
    FeistelCrypto crypto(K1, K2);
    
    // 解密并返回JSON数据
    return crypto.decryptJsonString(model_config_json_encrypted);
}

#endif // MODEL_WEIGHTS_LOADER_H
