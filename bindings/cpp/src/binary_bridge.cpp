#include "binary_bridge.hpp"
#include "rfc/learning/binary_gate.hpp"
#include <stdexcept>
#include <utility>

namespace rexgraph::python::binary {
namespace {
std::size_t byte_count(std::size_t lanes) { return lanes / 8 + (lanes % 8 != 0 ? 1 : 0); }
void validate(const std::string& raw, std::size_t lanes) {
    if (raw.size() != byte_count(lanes)) throw std::invalid_argument("packed Boolean plane byte count differs from its declared lanes");
    if (lanes % 8 != 0 && (static_cast<unsigned char>(raw.back()) >> (lanes % 8)) != 0)
        throw std::invalid_argument("packed Boolean plane has nonzero tail padding");
}
rfc::machine::PackedBits decode(const std::string& raw, std::size_t lanes, std::size_t maximum) {
    validate(raw, lanes);
    rfc::machine::PackedBits result(lanes, false, rfc::machine::PackedBits::words_for_size(maximum));
    for (std::size_t word = 0; word < result.word_count(); ++word) {
        std::uint64_t value = 0;
        for (std::size_t byte = 0; byte < 8 && word * 8 + byte < raw.size(); ++byte)
            value |= std::uint64_t{static_cast<unsigned char>(raw[word * 8 + byte])} << (byte * 8);
        result.assign_word(word, value);
    }
    return result;
}
std::string encode(const rfc::machine::PackedBits& plane) {
    std::string raw(byte_count(plane.size()), '\0');
    auto* bytes = reinterpret_cast<unsigned char*>(raw.data());
    for (std::size_t byte = 0; byte < raw.size(); ++byte)
        bytes[byte] = static_cast<unsigned char>((plane.word(byte / 8) >> ((byte % 8) * 8)) & 0xffu);
    return raw;
}
}
struct Input::State {
    rfc::machine::PackedBits a, b;
    std::size_t maximum;
    State(rfc::machine::PackedBits first, rfc::machine::PackedBits second, std::size_t max)
        : a(std::move(first)), b(std::move(second)), maximum(max) {}
};
Input::Input(const std::string& a, const std::string& b, std::size_t lanes, std::size_t maximum) {
    if (!maximum || maximum > 64000000) throw std::invalid_argument("packed Boolean lane admission requires 1 through 64000000 lanes");
    if (lanes > maximum) throw std::length_error("packed Boolean lane admission exceeded");
    state_ = std::make_shared<const State>(decode(a, lanes, maximum), decode(b, lanes, maximum), maximum);
}
std::size_t Input::lanes() const { return state_->a.size(); }
std::size_t Input::words() const { return state_->a.word_count(); }
std::size_t Input::max_lanes() const { return state_->maximum; }
std::string Input::first() const { return encode(state_->a); }
std::string Input::second() const { return encode(state_->b); }
std::string forward(unsigned table, const std::shared_ptr<Input>& input) {
    if (!input) throw std::invalid_argument("packed Boolean input owner is absent");
    if (table > 15) throw std::invalid_argument("binary truth table must fit four bits");
    return encode(rfc::learning::BinaryTruthGate2(static_cast<std::uint8_t>(table)).forward(input->state_->a, input->state_->b));
}
struct FitAccess {
    static Fit run(const std::shared_ptr<Input>& input, const std::string& target) {
        if (!input) throw std::invalid_argument("packed Boolean input owner is absent");
        const auto labels = decode(target, input->lanes(), input->max_lanes());
        const auto result = rfc::learning::fit_binary_truth_gate2(input->state_->a, input->state_->b, labels);
        return {result.gate.truth_table(), result.hamming_error};
    }
};
Fit fit(const std::shared_ptr<Input>& input, const std::string& target) { return FitAccess::run(input, target); }
std::string engine_version() { return RFC_ENGINE_VERSION; }
std::string engine_digest() { return RFC_ENGINE_DIGEST; }
std::string binding_digest() { return RFC_BINARY_BINDING_DIGEST; }
}
