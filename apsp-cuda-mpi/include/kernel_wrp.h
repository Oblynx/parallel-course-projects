#pragma once
#include <cuda_runtime_api.h>

void phase1(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N);
void phase2(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);

void phase1_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N);
void phase2_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);

