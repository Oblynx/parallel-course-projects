#pragma once
#include <cuda_runtime_api.h>
#include "utils.h"

void phase1(const dim3 gs, const dim3 bs, int* g, const xy tileStart, const int pitch);
void phase2Row(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int rowStart, const int pitch, const int t_pitch);
void phase2Col(const dim3 gs, const dim3 bs, int* g, const int* primaryTile, const int colStart, const int pitch, const int t_pitch);
void phase3(const dim3 gs, const dim3 bs, int* g, const int* row, const int* col, const int b, const xy start,
    const int pitch, const int r_pitch, const int c_pitch, const int rcFlag);
  

void phase1_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int N);
void phase2_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3_multiy_test(const dim3 gs, const dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase1_test(dim3 gs, dim3 bs, int* g, const int pstart, const int N);
void phase2_test(dim3 gs, dim3 bs, int* g, const int pstart, const int primary_n, const int N);
void phase3_test(dim3 gs, dim3 bs, int* g, const int pstart, const int primary_n, const int N);
