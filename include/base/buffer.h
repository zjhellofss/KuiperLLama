#ifndef LC_INCLUDE_BASE_BUFFER_H_
#define LC_INCLUDE_BASE_BUFFER_H_
#include <memory>

#include "base/alloc.h"
class Buffer : public Noncopyable {
 private:
  size_t byte_size_ = 0;
  void *ptr_ = nullptr;
  bool use_external_ = false;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void *ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  void *get_ptr();

  const void *get_ptr() const;

  size_t get_byte_size() const;
};

#endif