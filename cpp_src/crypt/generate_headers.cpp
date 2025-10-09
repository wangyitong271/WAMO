#include <iostream>
#include <fstream>
#include <string>
#include "feistel_crypto.h"
#include "model_weights.h"
#include "RSAKeyStore.h"  // 包含RSA加密的密钥信息
#include "rsa.h"         // 包含RSA解密函数

// 生成加密与解密后的头文件:
//  model_weights_cry.h  (包含加密后的 JSON: model_config_json_encrypted)
//  model_weights_dec.h  (包含解密验证后的 JSON: model_config_json_decrypted, 应与原始等价)
// 密钥从RSAKeyStore.h中的加密信息中解密获得

// 从RSA加密的key中解密获取K1和K2
std::pair<uint32_t, uint32_t> decryptKeysFromRSA() {
    // 解密原始key字符串
    std::string decrypted_key = decrypt(rsa_ciphertext, keys.d, keys.n);
    
    std::cout << "解密得到的原始密钥字符串: \"" << decrypted_key << "\"" << std::endl;
    
    // 从解密的key中提取K1和K2（使用与encode_rsa.cpp相同的算法）
    uint32_t hash = 0;
    for (char c : decrypted_key) {
        hash = hash * 31 + static_cast<uint32_t>(c);
    }
    
    uint32_t K1 = hash & 0xFFF;        // 取低12位作为K1
    uint32_t K2 = (hash >> 12) & 0x7FF; // 取13-23位作为K2
    
    // 确保K1和K2在有效范围内
    K1 = K1 % 4096;  // 确保K1在0-4095范围内
    K2 = K2 % 2048;  // 确保K2在0-2047范围内
    
    return std::make_pair(K1, K2);
}

// 解密model_weights_cry.h中的加密JSON数据
std::string decryptModelWeights(const std::string& encrypted_json, uint32_t K1, uint32_t K2) {
    FeistelCrypto crypto(K1, K2);
    return crypto.decryptJsonString(encrypted_json);
}

static void writeHeader(const std::string& path, const std::string& guard, const std::string& varName, const std::string& jsonText, const std::string& comment) {
    std::ofstream ofs(path, std::ios::binary);
    if(!ofs){
        std::cerr << "写入文件失败: " << path << "\n";
        return;
    }
    ofs << "#ifndef " << guard << "\n#define " << guard << "\n\n#include <string>\n\n// " << comment << "\nconst std::string " << varName << " = R\"json(\n";
    ofs << jsonText << "\n)json\";\n\n#endif // " << guard << "\n";
}

int main() {
    std::cout << "开始从RSA加密信息中解密密钥...\n";
    
    // 从RSA加密的信息中解密获取K1和K2
    auto keys_pair = decryptKeysFromRSA();
    uint32_t K1 = keys_pair.first;
    uint32_t K2 = keys_pair.second;

    std::cout << "解密得到的密钥: K1=" << K1 << " K2=" << K2 << "\n";

    FeistelCrypto crypto(K1, K2);

    // 加密
    std::string encrypted = crypto.encryptJsonString(model_config_json);
    if(encrypted.empty()) {
        std::cerr << "加密失败, 退出\n";
        return 1;
    }

    // 只生成加密头文件
    writeHeader("model_weights_cry.h", "MODEL_WEIGHTS_CRY_H", "model_config_json_encrypted", encrypted, "加密后的模型 JSON (Feistel 尾数加密) 密钥来自RSA解密");

    std::cout << "已生成: model_weights_cry.h\n";
    std::cout << "使用方法: #include \"model_weights_loader.h\" 然后直接使用 model_config_json 变量\n";
    std::cout << "注意: model_weights_loader.h 是固定文件，无需重新生成\n";
    return 0;
}
