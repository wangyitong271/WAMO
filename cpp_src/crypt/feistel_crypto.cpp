#include "feistel_crypto.h"
#include <bitset>
#include <iostream>
#include <random>
#include <iomanip>
#include <sstream>

// ---- 新增: 局部辅助函数，保证位宽一致 ----
static inline uint32_t F12(uint32_t x, uint32_t k) { // 输出 12 位
    return (x ^ k) & 0x0FFF; // 12 bits
}
static inline uint32_t F11(uint32_t x, uint32_t k) { // 输出 11 位
    return (x ^ k) & 0x07FF; // 11 bits
}

FeistelCrypto::FeistelCrypto(uint32_t key1, uint32_t key2) {
    setKeys(key1, key2);
}

void FeistelCrypto::setKeys(uint32_t key1, uint32_t key2) {
    K1 = key1 & 0xFFF;   // 12位
    K2 = key2 & 0x7FF;   // 11位
}

uint32_t FeistelCrypto::F(uint32_t R, uint32_t K) { // 保留原接口（不直接再用）
    return R ^ K;
}

float FeistelCrypto::encryptFloat(float value) {
    union { float f; uint32_t i; } u; u.f = value;
    uint32_t bits = u.i;

    uint32_t S = (bits >> 31) & 0x1;
    uint32_t E = (bits >> 23) & 0xFF;
    uint32_t M = bits & 0x7FFFFF; // 23 位

    uint32_t L0 = (M >> 11) & 0x0FFF; // 12 位
    uint32_t R0 = M & 0x07FF;         // 11 位

    // Round 1
    uint32_t L1 = R0;                                  // 11 位
    uint32_t R1 = F12(L0, F12(R0, K1));                // 12 位: R1 = L0 XOR (R0 XOR K1) (都限制为 12 位)

    // Round 2
    uint32_t L2 = R1;                                  // 12 位
    uint32_t R2 = F11(L1, F11(R1, K2));                // 11 位: R2 = L1 XOR (R1 XOR K2) (都限制为 11 位)

    uint32_t M_new = ((L2 & 0x0FFF) << 11) | (R2 & 0x07FF);
    uint32_t newBits = (S << 31) | (E << 23) | M_new;
    u.i = newBits;
    return u.f;
}

float FeistelCrypto::decryptFloat(float value) {
    union { float f; uint32_t i; } u; u.f = value;
    uint32_t bits = u.i;

    uint32_t S = (bits >> 31) & 0x1;
    uint32_t E = (bits >> 23) & 0xFF;
    uint32_t M = bits & 0x7FFFFF; // 23 位 (加密后)

    uint32_t L2 = (M >> 11) & 0x0FFF; // 12 位 (= R1)
    uint32_t R2 = M & 0x07FF;         // 11 位 (= R2)

    // Inverse Round 2:
    // R2 = L1 XOR (R1 XOR K2)_11  => L1 = R2 XOR (R1 XOR K2)_11
    uint32_t R1 = L2;                              // 12 位
    uint32_t L1 = F11(R2, F11(R1, K2));            // 11 位

    // Inverse Round 1:
    // R1 = L0 XOR (R0 XOR K1)_12  => L0 = R1 XOR (R0 XOR K1)_12
    // 且 R0 = L1
    uint32_t R0 = L1;                              // 11 位
    uint32_t L0 = F12(R1, F12(R0, K1));            // 12 位

    uint32_t M_orig = ((L0 & 0x0FFF) << 11) | (R0 & 0x07FF);
    uint32_t origBits = (S << 31) | (E << 23) | M_orig;
    u.i = origBits;
    return u.f;
}

void FeistelCrypto::encryptJsonRecursive(json& j) {
    if (j.is_object()) {
        for (auto& [k, v] : j.items()) encryptJsonRecursive(v);
    } else if (j.is_array()) {
        for (auto& v : j) encryptJsonRecursive(v);
    } else if (j.is_number_float()) {
        float f = j.get<float>();
        j = encryptFloat(f);
    }
}

void FeistelCrypto::decryptJsonRecursive(json& j) {
    if (j.is_object()) {
        for (auto& [k, v] : j.items()) decryptJsonRecursive(v);
    } else if (j.is_array()) {
        for (auto& v : j) decryptJsonRecursive(v);
    } else if (j.is_number_float()) {
        float f = j.get<float>();
        j = decryptFloat(f);
    }
}

std::string FeistelCrypto::encryptJsonString(const std::string& jsonStr) {
    try { json j = json::parse(jsonStr); encryptJsonRecursive(j); return j.dump(2); } catch (const std::exception& e) { std::cerr << "加密错误: " << e.what() << "\n"; return ""; }
}
std::string FeistelCrypto::decryptJsonString(const std::string& encryptedJsonStr) {
    try { json j = json::parse(encryptedJsonStr); decryptJsonRecursive(j); return j.dump(2); } catch (const std::exception& e) { std::cerr << "解密错误: " << e.what() << "\n"; return ""; }
}

std::string FeistelCrypto::getKeyInfo() const {
    std::ostringstream oss;
    oss << "密钥信息:\n";
    oss << "K1 (12位): " << std::bitset<12>(K1) << " (十进制: " << K1 << ")\n";
    oss << "K2 (11位): " << std::bitset<11>(K2) << " (十进制: " << K2 << ")";
    return oss.str();
}

std::pair<uint32_t, uint32_t> FeistelCrypto::generateRandomKeys() {
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> d1(0, 0x0FFF), d2(0, 0x07FF);
    return { d1(gen), d2(gen) };
}
