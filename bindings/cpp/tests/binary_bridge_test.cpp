#include "binary_bridge.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::size_t checks = 0;
void require(bool value) { ++checks; if (!value) throw std::runtime_error("binary bridge assertion failed"); }
std::string pack(const std::vector<unsigned>& bits) {
    std::string result(bits.size() / 8 + (bits.size() % 8 ? 1 : 0), '\0');
    auto* bytes = reinterpret_cast<unsigned char*>(result.data());
    for (std::size_t i = 0; i < bits.size(); ++i) bytes[i / 8] |= static_cast<unsigned char>(bits[i] << (i % 8));
    return result;
}
template<class F> void rejects(F&& action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    require(failed);
}
}
int main() {
    using namespace rexgraph::python::binary;
    for (const std::size_t count : std::array<std::size_t, 12>{0,1,7,8,9,63,64,65,127,128,129,513}) {
        std::vector<unsigned> a(count), b(count), labels(count);
        for (std::size_t i = 0; i < count; ++i) { a[i] = static_cast<unsigned>((i / 2) % 2); b[i] = static_cast<unsigned>(i % 2); }
        const auto input = std::make_shared<Input>(pack(a), pack(b), count, 1024);
        require(input->lanes() == count); require(input->words() == (count + 63) / 64);
        require(input->first() == pack(a)); require(input->second() == pack(b));
        for (unsigned table = 0; table < 16; ++table) {
            for (std::size_t i = 0; i < count; ++i) labels[i] = (table >> (2 * a[i] + b[i])) & 1u;
            require(forward(table, input) == pack(labels));
            auto result = fit(input, pack(labels));
            unsigned expected = 0;
            for (std::size_t i = 0; i < count; ++i) expected |= labels[i] << (2 * a[i] + b[i]);
            require(result.truth_table == expected); require(result.hamming_error == 0);
            for (std::size_t i = 0; i < count; ++i) labels[i] = static_cast<unsigned>((i / 4) % 2);
            result = fit(input, pack(labels));
            std::size_t best_loss = count + 1; unsigned best_table = 0;
            for (unsigned candidate = 0; candidate < 16; ++candidate) {
                std::size_t loss = 0;
                for (std::size_t i = 0; i < count; ++i) loss += ((candidate >> (2 * a[i] + b[i])) & 1u) != labels[i] ? 1 : 0;
                if (loss < best_loss) { best_loss = loss; best_table = candidate; }
            }
            require(result.truth_table == best_table); require(result.hamming_error == best_loss);
        }
    }
    rejects([] { Input input("\1", "\0", 1, 0); });
    rejects([] { Input input("\1", "\0", 2, 1); });
    rejects([] { Input input("\377", "\0", 1, 8); });
    rejects([] { Input input("", "\0", 1, 8); });
    rejects([] { forward(0, nullptr); });
    rejects([] { fit(nullptr, ""); });
    const auto input = std::make_shared<Input>(std::string(1, '\0'), std::string(1, '\0'), 1, 8);
    rejects([&] { forward(16, input); });
    rejects([&] { fit(input, std::string(1, '\377')); });
    require(!engine_version().empty()); require(!engine_digest().empty()); require(!binding_digest().empty());
    std::cout << "{\"assertions\":" << checks << "}\n";
}
