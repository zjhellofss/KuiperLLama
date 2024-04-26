#include "base/buffer.h"

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
  if (!ptr_ && allocator_) {
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);
  }
}

Buffer::~Buffer() {
  if (!use_external_) {
    if (ptr_ && allocator_) {
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}

void* Buffer::get_ptr() { return ptr_; }

const void* Buffer::get_ptr() const { return ptr_; }

size_t Buffer::get_byte_size() const { return byte_size_; }