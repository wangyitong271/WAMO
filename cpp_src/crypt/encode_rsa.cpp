
#include "rsa.h"
#include "model_weights.h"
#include <cassert>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cstdint>
using namespace std;

// 将字符串key转换为K1和K2
pair<uint32_t, uint32_t> keyToK1K2(const string& key) {
    // 计算字符串的哈希值
    uint32_t hash = 0;
    for (char c : key) {
        hash = hash * 31 + static_cast<uint32_t>(c);
    }

    // 从哈希值中提取K1 (12位，范围0-4095) 和 K2 (11位，范围0-2047)
    uint32_t K1 = hash & 0xFFF;        // 取低12位作为K1
    uint32_t K2 = (hash >> 12) & 0x7FF; // 取13-23位作为K2

    // 确保K1和K2在有效范围内
    K1 = K1 % 4096;  // 确保K1在0-4095范围内
    K2 = K2 % 2048;  // 确保K2在0-2047范围内

    return make_pair(K1, K2);
}

// 将K1和K2编码为字符串key
string K1K2ToKey(uint32_t K1, uint32_t K2) {
    // 将K1和K2组合成一个32位数值
    uint32_t combined = (K1 & 0xFFF) | ((K2 & 0x7FF) << 12);

    // 将数值转换为字符串
    string result = "RSA_";

    // 使用Base64类似的编码，但使用可打印字符
    const string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    for (int i = 0; i < 6; i++) {
        result += charset[combined % 64];
        combined /= 64;
    }

    return result;
}


int main()
{
    // 设置命令窗口为UTF-8编码
    // system("chcp 65001");

    // 随机种子
    srand(static_cast<unsigned int>(time(0)));

    // 密钥，用于加密和解密
    string key = "Hello, RSA!";
    string txt0 = model_config_json;

    // 从key字符串中提取K1和K2
    pair<uint32_t, uint32_t> keys_pair = keyToK1K2(key);
    uint32_t K1 = keys_pair.first;
    uint32_t K2 = keys_pair.second;

//    cout << "从密钥字符串 \"" << key << "\" 提取的参数:" << endl;
//    cout << "K1 = " << K1 << " (0x" << hex << K1 << dec << ")" << endl;
//    cout << "K2 = " << K2 << " (0x" << hex << K2 << dec << ")" << endl;

    // 验证反向转换
    string reconstructed_key = K1K2ToKey(K1, K2);
    cout << "重构的密钥字符串: \"" << reconstructed_key << "\"" << endl;

    // RSA加密解密示例
    // cout << "RSA加密解密key" << endl;

    // 生成密钥
    // cout << "--------------------密钥产生----------------------------" << endl;
    Keys keys = generateKeys();
    // cout << " 公钥 e = " << keys.e << ", 公钥 n = " << keys.n << endl;
    // cout << " 私钥 d = " << keys.d << ", 私钥 n = " << keys.n << endl;

    // RSA加密
    // cout << "---------------------加   密--------------------------" << endl;
    // cout << "明文： " << key << endl;
    vector<long long> rsa_ciphertext = encrypt(key, keys.e, keys.n);

    // cout << "密文为：";
    // for (const auto &c : rsa_ciphertext)
    //     cout << c << " ";
    // cout << endl;

    // 将密钥keys和密文rsa_ciphertext以及K1、K2存储到RSAKeyStore.h文件中
    ofstream out("RSAKeyStore.h");
    out << "#ifndef RSAKeyStore_H" << endl;
    out << "#define RSAKeyStore_H" << endl;
    out << "#include <vector>" << endl;
    out << "#include <string>" << endl;
    out << "#include \"rsa.h\"" << endl;
    out << "using namespace std;" << endl;
    out << "// RSA加密的原始key字符串" << endl;
    out << "string original_key = \"" << key << "\";" << endl;
    out << "// 从key提取的Feistel密钥" << endl;
    out << "uint32_t K1 = " << K1 << ";" << endl;
    out << "uint32_t K2 = " << K2 << ";" << endl;
    out << "// RSA加密结果" << endl;
    out << "vector<long long> rsa_ciphertext = {";
    for (const auto &c : rsa_ciphertext)
        out << c << ", ";
    out << "};" << endl;
    out << "Keys keys = {" << keys.e << ", " << keys.d << ", " << keys.n << "};" << endl;
    out << "#endif" << endl;

    
    return 0;
}
