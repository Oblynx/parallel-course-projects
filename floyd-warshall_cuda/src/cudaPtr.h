#pragma once
#include <stdexcept>

enum CopyDir{ H2D= cudaMemcpyHostToDevice, D2H= cudaMemcpyDeviceToHost };

template<typename T> class cudaPtr;
/*! `std::unique_ptr`-like CUDA smart pointer for arrays of type T.
  @param T non-const, non-ref arithmetic type
  
  - Declared as cudaPtr<T[]>, where T={int, float, double, ...}
  - __NOT__ thread safe.
*/
template<typename T> class cudaPtr<T[]>{
public:
  /*! Default constructor for creating arrays of type T elements
    @param elementN_ Number of elements in array. If 0, no allocation occurs.
    @param flag Leave at default (`host`) for normal use cases.
    See also CUDA C Programming Guide J.2.2.6
    If used without specifying element number, a call to reset() is required for
    memory allocation.
    Allocation failures result in bad_alloc exception.
  */
  cudaPtr(unsigned elementN= 0): elementN_(elementN),errorState_(cudaSuccess),ownershipReleased_(false){
    if(elementN) allocate();
  }
  //!@{ Delete copy semantics to enforce unique memory ownership
  cudaPtr(const cudaPtr&) =delete;
  cudaPtr& operator=(const cudaPtr&) =delete;
  //!@}
  //! Free memory. If CUDA runtime returns an error code here, it will be lost (destructor doesn't throw).
  ~cudaPtr() noexcept{ if (!ownershipReleased_) deallocate(); }
  //! Allocate the requested memory after freeing any previous allocation
  void reset(int elementN){
    if (!ownershipReleased_){
      deallocate();
      elementN_= elementN;
      allocate();
    }
  }
  //! Release ownership and return contained pointer
  T* release(){
    ownershipReleased_= true; T* tmpData= data_; data_= nullptr;
    return tmpData;
  }
  //! Get contained pointer -- __unsafe__
  T* get() const { return data_; }
  void releaseAndFree() { 
    if (!ownershipReleased_){
      ownershipReleased_= true; deallocate();
    }
  }
  inline T& operator[](int idx) const { return data_[idx]; }
  void copy(T* other, const CopyDir dir){
    if(dir==H2D) cudaMemcpy(data_, other, elementN_*sizeof(T), H2D);
    else         cudaMemcpy(other, data_, elementN_*sizeof(T), D2H);
  }
  cudaError_t getErrorState() const { return errorState_; }
private:
  inline void allocate(){
    errorState_= cudaMalloc((void**)&data_, elementN_*sizeof(T));
    if (errorState_ != cudaSuccess) throw new std::bad_alloc();
  }
  inline void deallocate() noexcept{ errorState_= cudaFree(data_); }
      //---$$$---//
  //! The contained data_.
  T* data_;
  unsigned elementN_;
  cudaError_t errorState_;
  bool ownershipReleased_;
};
