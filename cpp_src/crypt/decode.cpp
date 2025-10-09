#include "RSAKeyStore.h"
#include "des.h"
#include "DESInfo.h"

#include <emscripten/bind.h>


using namespace std;
using namespace emscripten;

int main(){
    cout << "--------------------解   密----------------------------" << endl;
    string decrypted_key = decrypt(rsa_ciphertext, keys.d, keys.n);
    cout << "解密结果为：" << decrypted_key << endl;

    // 密文是DESIfo.h中的ciphertext
    DES des(decrypted_key);
    cout << "明文： " << des.decryptToText(ciphertext) << endl;
    
}

EMSCRIPTEN_BINDINGS(my_module)
{
    // function("main", &main);
}