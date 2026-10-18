#pragma once
#include "rfc/storage/rfcx.hpp"
#include <cstring>

namespace rexgraph::python {
// An owning byte image for exact comparisons of retained tensor storage.
class ModelByteImage {
public:
    ModelByteImage(const unsigned char* data,std::size_t size,std::size_t maximum) {
        if (size>maximum) throw std::length_error("model witness byte budget");
        if (size && !data) throw std::invalid_argument("missing model witness bytes");
        if (size) writer_.raw({reinterpret_cast<const std::byte*>(data),size});
    }
    bool matches(const unsigned char* data,std::size_t size) const noexcept {
        return size==writer_.size() && (!size || (data && std::memcmp(writer_.bytes().data(),data,size)==0));
    }
private:
    rfc::storage::rfcx_detail::Writer writer_;
};
// Emit existing canonical value bytes through the native storage writer.
class ModelValueWriter {
public:
    void raw(const unsigned char* data,std::size_t size) {
        if(size && !data) throw std::invalid_argument("missing native value bytes");
        (void)rfc::machine::checked_size_add(writer_.size(),size);
        if(size) writer_.raw({reinterpret_cast<const std::byte*>(data),size});
    }
    void byte(unsigned char value) {writer_.u8(value);}
    void uint(std::size_t value) {
        while(value>=128) {byte(static_cast<unsigned char>((value&127)|128));value>>=7;}
        byte(static_cast<unsigned char>(value));
    }
    const char* data() const noexcept {return reinterpret_cast<const char*>(writer_.bytes().data());}
    std::size_t size() const noexcept {return writer_.size();}
private:
    rfc::storage::rfcx_detail::Writer writer_;
};
}
