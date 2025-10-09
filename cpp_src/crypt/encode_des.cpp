#include "RSAKeyStore.h"
#include "des.h"
#include "model_weights.h"
#include <fstream>

int main(){
    // cout << "--------------------解   密----------------------------" << endl;
    string decrypted_key = decrypt(rsa_ciphertext, keys.d, keys.n);
    // cout << "解密结果为：" << decrypted_key << endl;

    string decrypted_tex = decrypted_key;
    string txt0 = model_config_json;

    // DES加密解密示例
    // cout << "DES加密解密示例" << endl;
    // cout << "从ECDH解密后的key对明文进行DES加密解密" << endl;
    DES des(decrypted_tex);
    // cout << "明文： " << txt0 << endl;
    // 密钥
    // cout << "密钥： " << decrypted_tex << endl;
    string ciphertext = des.encrypt(txt0);
    // cout << "密文（二进制）： " << ciphertext << endl;


    // 将密文ciphertext以字符串形式存储到DESInfo.h文件中
    ofstream out("DESInfo.h");
    out << "#ifndef DESInfo_H" << endl;
    out << "#define DESInfo_H" << endl;
    out << "#include <string>" << endl;
    out << "using namespace std;" << endl;
    out << "string ciphertext = \"" << ciphertext << "\";" << endl;
    out << "#endif" << endl;




    // cout << "解密后的文本（字符串）： " << des.decryptToText(ciphertext) << endl;

    return 0;
}
