#ifndef _RSA_H_
#define _RSA_H_



#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <sys/timeb.h>

using namespace std;

// 结构体用于存储公钥和私钥
struct Keys {
    int e;
    int d;
    int n;
};

// 判断一个数是否为素数
bool isPrime(int num) {
    if (num < 10000) return false; // 根据原代码逻辑，过滤小于10000的数
    for (int i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}

// 生成两个大素数p和q
vector<int> generatePrimes() {
    vector<int> primes(2);
    primes[0] = rand();
    primes[1] = rand();
    while (!(isPrime(primes[0]) && isPrime(primes[1]))) {
        primes[0] = rand();
        primes[1] = rand();
    }
    return primes;
}

// 使用扩展欧几里得算法计算 gcd 和 x, y 使得 ax + by = gcd(a, b)
int extendedGCD(int a, int b, int &x, int &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int gcd = extendedGCD(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return gcd;
}

// 快速幂算法计算 (base^exp) % mod
long long quickPow(long long base, int exp, int mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    if (result < 0)
        result += mod;
    return result;
}

// 生成 RSA 密钥对
Keys generateKeys() {
    vector<int> primes = generatePrimes();
    int p = primes[0];
    int q = primes[1];
    int n = p * q;
    int phi_n = (p - 1) * (q - 1);

    // 选择公钥 e
    int e = rand() % phi_n;
    int x, y;
    while (extendedGCD(phi_n, e, x, y) != 1) {
        e = rand() % phi_n;
    }

    // 计算私钥 d
    int d = y % phi_n;
    if (d < 0)
        d += phi_n;

    Keys keys = {e, d, n};
    return keys;
}

// 加密函数
vector<long long> encrypt(const string &message, int e, int n) {
    int length = message.size();
    vector<long long> ciphertext(length);
    for (int i = 0; i < length; ++i) {
        ciphertext[i] = quickPow(static_cast<long long>(message[i]), e, n);
    }
    return ciphertext;
}

// 解密函数
string decrypt(const vector<long long> &ciphertext, int d, int n) {
    string decrypted;
    for (auto &c : ciphertext) {
        long long decrypted_char = quickPow(c, d, n);
        decrypted += static_cast<char>(decrypted_char);
    }
    return decrypted;
}

#endif