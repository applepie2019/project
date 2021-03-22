all:
	icpc -qopenmp -g -O3 -DUSE_SNIPER -xCORE-AVX512 -std=c++11 main.cpp -o sharedmicro
