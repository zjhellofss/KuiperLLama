#include "base/buffer.h"

#include <glog/logging.h>

namespace base {
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator,
               void* ptr, bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();
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

void* Buffer::ptr() { return ptr_; }

const void* Buffer::ptr() const { return ptr_; }

size_t Buffer::byte_size() const { return byte_size_; }

bool Buffer::allocate() {
  if (allocator_ && byte_size_ != 0) {
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size_);
    if (!ptr_) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}

void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr && buffer.allocator_ != nullptr);
  CHECK(this->device_type() == buffer.device_type());

  size_t copy_size =
      byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
  return allocator_->memcpy(this->ptr_, buffer.ptr_, copy_size);
}

void Buffer::copy_from(const Buffer* buffer) const {
  if (!buffer) {
    return;
  }
  CHECK(allocator_ != nullptr && buffer->allocator_ != nullptr);
  CHECK(this->device_type() == buffer->device_type());

  size_t src_size = byte_size_;
  size_t dest_size = buffer->byte_size_;
  size_t copy_size = src_size < dest_size ? src_size : dest_size;
  return allocator_->memcpy(buffer->ptr_, this->ptr_, copy_size);
}

DeviceType Buffer::device_type() const { return device_type_; }

void Buffer::set_device_type(DeviceType device_type) {
  device_type_ = device_type;
}

}  // namespace base