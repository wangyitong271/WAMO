#ifndef FEISTEL_CRYPTO_H
#define FEISTEL_CRYPTO_H

#include <string>
#include <cstdint>
#include "json.hpp"

using json = nlohmann::json;

/**
 * Feistel 网络浮点数加密/解密系统
 * 
 * 特性：
 * - 对 JSON 字符串中的所有浮点数进行加密/解密
 * - 使用两个密钥进行两轮 Feistel 网络加密
 * - 只加密 IEEE754 浮点数的尾数部分，保持符号位和指数位不变
 * - 加密是可逆的，支持完整的解密功能
 */
class FeistelCrypto {
private:
    uint32_t K1;  // 12位密钥
    uint32_t K2;  // 11位密钥
    
    // F 函数（XOR 操作）
    uint32_t F(uint32_t R, uint32_t K);
    
    // 对单个浮点数进行加密
    float encryptFloat(float value);
    
    // 对单个浮点数进行解密
    float decryptFloat(float value);
    
    // 递归加密 JSON 对象中的浮点数
    void encryptJsonRecursive(json& j);
    
    // 递归解密 JSON 对象中的浮点数
    void decryptJsonRecursive(json& j);

public:
    /**
     * 构造函数
     * @param key1 12位密钥1 (范围: 0-4095)
     * @param key2 11位密钥2 (范围: 0-2047)
     */
    FeistelCrypto(uint32_t key1 = 0b011010101011, uint32_t key2 = 0b10110110011);
    
    /**
     * 设置密钥
     * @param key1 12位密钥1
     * @param key2 11位密钥2
     */
    void setKeys(uint32_t key1, uint32_t key2);
    
    /**
     * 加密 JSON 字符串
     * @param jsonStr 输入的 JSON 字符串
     * @return 加密后的 JSON 字符串
     */
    std::string encryptJsonString(const std::string& jsonStr);
    
    /**
     * 解密 JSON 字符串
     * @param encryptedJsonStr 加密的 JSON 字符串
     * @return 解密后的 JSON 字符串
     */
    std::string decryptJsonString(const std::string& encryptedJsonStr);
    
    /**
     * 获取当前密钥信息
     * @return 密钥信息字符串
     */
    std::string getKeyInfo() const;
    
    /**
     * 生成随机密钥对
     * @return 包含两个密钥的 pair<K1, K2>
     */
    static std::pair<uint32_t, uint32_t> generateRandomKeys();
};

#endif // FEISTEL_CRYPTO_H
