#include "model_storage_bridge.hpp"
#include <array>
#include <cassert>
#include <stdexcept>

int main() {
    using rexgraph::python::ModelByteImage;
    std::array<unsigned char,4> data{0,1,128,255};
    ModelByteImage image(data.data(),data.size(),data.size());
    assert(image.matches(data.data(),data.size()));
    const auto copy=data;
    data[2]=0;
    assert(!image.matches(data.data(),data.size()));
    assert(image.matches(copy.data(),copy.size()));
    assert(!image.matches(copy.data(),copy.size()-1));
    assert(!image.matches(nullptr,copy.size()));
    ModelByteImage empty(nullptr,0,0);
    assert(empty.matches(nullptr,0));
    assert(!empty.matches(copy.data(),copy.size()));
    bool bytes=false,missing=false;
    try {const ModelByteImage rejected(copy.data(),copy.size(),copy.size()-1);} catch(const std::length_error&) {bytes=true;}
    try {const ModelByteImage rejected(nullptr,1,1);} catch(const std::invalid_argument&) {missing=true;}
    assert(bytes && missing);
}
