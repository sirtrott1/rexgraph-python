#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rexgraph::python::binary {
class Input {
public:
    Input(const std::string&, const std::string&, std::size_t, std::size_t);
    std::size_t lanes() const;
    std::size_t words() const;
    std::size_t max_lanes() const;
    std::string first() const;
    std::string second() const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    friend std::string forward(unsigned, const std::shared_ptr<Input>&);
    friend struct FitAccess;
};
struct Fit { unsigned truth_table{}; std::size_t hamming_error{}; };
std::string forward(unsigned, const std::shared_ptr<Input>&);
Fit fit(const std::shared_ptr<Input>&, const std::string&);
std::string engine_version();
std::string engine_digest();
std::string binding_digest();
}
