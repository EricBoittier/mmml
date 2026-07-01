#ifndef NOCUDAC
#ifndef FORCE_H
#define FORCE_H

#include <cuda.h>
#include "cuda_utils.h"

//
// Simple strided storage class for forces
//
template <typename T>
class Force {

private:

  // Stride of the force data:
  // x data is in data[0...size-1];
  // y data is in data[stride...stride+size-1];
  // z data is in data[stride*2...stride*2+size-1];
  //int stride;

  // Force data
  //int data_len;
  //T *data;

  int _size;
  int _stride;
  int _capacity;
  T *_xyz;

  //cudaXYZ<T> xyz;

 public:

  Force() {
    _size = 0;
    _stride = 0;
    _capacity = 0;
    _xyz = NULL;
  }

  Force(const int size) {
    _size = 0;
    _stride = 0;
    _capacity = 0;
    _xyz = NULL;
    this->realloc(size);
  }

  Force(const char *filename);

  ~Force() {
    if (_xyz != NULL) deallocate<T>(&_xyz);
  }


  void clear(cudaStream_t stream=0) {
    clear_gpu_array<T>(this->_xyz, 3*this->_stride, stream);
  }

  bool compare(Force<T>& force, const double tol, double& max_diff);

  // Re-allocates array, does not preserve content
  void realloc(int size, float fac=1.0f) {
    this->_size = size;
    // Returns stride that aligns with 256 byte boundaries
    this->_stride = (( (size-1+32)*sizeof(T) - 1)/256 + 1)*256/sizeof(T);
    int new_capacity = (int)((double)(3*this->_stride)*(double)fac);
    reallocate<T>(&this->_xyz, &this->_capacity, new_capacity, fac);
  }

  int stride() {return _stride;}
  int size() {return _size;}
  T* xyz() {return _xyz;}
  T* x() {return &_xyz[0];}
  T* y() {return &_xyz[_stride];}
  T* z() {return &_xyz[_stride*2];}

  void getXYZ(T* h_x, T* h_y, T* h_z);
  
  int stride() const {return _stride;}
  int size() const {return _size;}
  const T* xyz() const {return _xyz;}
  const T* x() const {return &_xyz[0];}
  const T* y() const {return &_xyz[_stride];}
  const T* z() const {return &_xyz[_stride*2];}

  template <typename T2> void convert(Force<T2>& force, cudaStream_t stream=0);
  template <typename T2> void convert(cudaStream_t stream=0);
  template <typename T2, typename T3> void convert_to(Force<T3>& force, cudaStream_t stream=0);
  template <typename T2, typename T3> void convert_add(Force<T3>& force, cudaStream_t stream=0);
  template <typename T2, typename T3> void add(Force<T3>& force, cudaStream_t stream=0);
  template <typename T2> void add(float3 *force_data, int force_n, cudaStream_t stream=0);
  template <typename T2> void save(const char* filename);
};


#endif // FORCE_H
#endif //NOCUDAC
