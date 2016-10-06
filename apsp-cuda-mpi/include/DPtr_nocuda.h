#pragma once
#include <cstdio>

enum Dir { H2D, D2H };
struct DPtr{
  DPtr(int N);
  ~DPtr();
  void copy (int* a, const int N, const Dir dir, const int devOffset=0);
  int* get() const;
  operator int*() const;
  private:
  int* data_;
};

