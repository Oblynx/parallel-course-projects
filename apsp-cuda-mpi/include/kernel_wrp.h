#pragma once
#include <cuda_runtime_api.h>

void phase1(const dim3 gs, const dim3 bs, int* g, const int tileStart, const int pitch);
void phase2Row(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int rowStart, const int pitch);
void phase2Col(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int colStart, const int pitch);
void phase3(const dim3 gs, const dim3 bs, int* g, const int* row, const int* col, const int b, const int xStart,
    const int yStart, const int pitch, const int rowpitch, const int rcFlag);
  

void phase1_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N);
void phase2_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3_multiy(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);

void phase1_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N);
void phase2_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
