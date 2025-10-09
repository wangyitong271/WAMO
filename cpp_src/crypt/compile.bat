@REM 设置字符集
chcp 65001
@echo off


@REM g++ -o encode.exe encode.cpp
@REM encode.exe

g++ -o encode_rsa.exe encode_rsa.cpp
encode_rsa.exe

@REM g++ -o decode_rsa.exe decode_rsa.cpp
@REM decode_rsa.exe

@REM emcc -lembind -o decode_rsa.js decode_rsa.cpp -O3

g++ -o encode_des.exe encode_des.cpp
encode_des.exe

@REM g++ -o decode_des.exe decode_des.cpp
@REM decode_des.exe
del *.exe
@REM emcc -lembind -o decode.js decode.cpp -O3

emcc -lembind -o mnist.js mnist.cpp -O3

