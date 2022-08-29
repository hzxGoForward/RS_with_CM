## 编译方案
假设已经在当前程序主目录下
###　Linux/MAC OS/Windows
1. cd build
2. cmake ..
3. make -j8

### Windows 平台生成 makefile
1. cd build
2. cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -G "Unix Makefiles" (该命令生成 makefile)
3. cmake ..
4. make -j8 