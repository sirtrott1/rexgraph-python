#pragma once
#include "bridge.hpp"

namespace rexgraph::python {
struct ChainTerm {
    std::uint32_t target{};
    std::string coefficient;
};
using ChainFamily = std::vector<std::vector<std::vector<ChainTerm>>>;

std::vector<std::vector<std::uint64_t>> observe_context_counts(
    const std::vector<std::vector<std::uint64_t>>&,
    const std::vector<std::uint32_t>&, const std::vector<std::uint32_t>&,
    std::uint64_t);

class ChainMap {
public:
    static std::shared_ptr<ChainMap> prepare(const std::shared_ptr<Source>&,
        const std::shared_ptr<Source>&, const ChainFamily&, std::uint64_t,
        const std::vector<std::size_t>&);
    std::vector<std::string> push(std::uint16_t, const std::vector<std::string>&) const;
    std::vector<std::int64_t> push_integer(std::uint16_t, const std::vector<std::int64_t>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit ChainMap(std::shared_ptr<const State>);
};

struct LiteralTokenSpan { std::size_t begin{}, end{}; };
std::vector<LiteralTokenSpan> literal_whitespace_spans(const std::string&);
struct LiteralTokenChart {
    std::shared_ptr<Source> tokens, bytes;
    std::shared_ptr<ChainMap> map;
    ChainFamily family;
    std::vector<LiteralTokenSpan> spans;
};
LiteralTokenChart literal_token_chart(const std::string&, const std::vector<std::string>&,
    const std::vector<LiteralTokenSpan>&, std::uint64_t, std::size_t,
    const std::vector<std::size_t>&);
LiteralTokenChart literal_whitespace_chart(const std::string&, std::uint64_t,
    std::size_t, const std::vector<std::size_t>&);
}
