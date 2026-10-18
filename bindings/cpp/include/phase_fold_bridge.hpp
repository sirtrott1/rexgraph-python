#pragma once
#include "bridge.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rexgraph::python {
struct PhaseFoldOperationInfo {
    std::uint64_t identity{};
    bool active{}, forward{};
    std::size_t crease{}, edge{};
    std::vector<std::uint32_t> moving;
    std::vector<std::string> phase;
};
struct PhaseFoldProposalInfo {
    std::uint64_t key{};
    std::size_t position{};
    bool active{};
    std::vector<std::string> phase;
};
struct PhaseFoldInfo {
    unsigned kind{}, grade{}, index{};
    bool has_cell{};
    std::size_t operations{}, proposals{}, degree{}, endpoint_coordinates{}, observation_coordinates{}, branch{};
    std::vector<std::size_t> source_sizes, shape;
    std::vector<std::string> polynomial, involution, endpoint, positions;
    std::vector<PhaseFoldOperationInfo> operation_views;
    std::vector<PhaseFoldProposalInfo> proposal_views;
};
class PhaseFoldArchive {
public:
    static std::vector<std::size_t> default_limits();
    static std::shared_ptr<PhaseFoldArchive> load(const char*, std::size_t, const std::vector<std::size_t>&);
    static std::shared_ptr<PhaseFoldArchive> from_source(const std::shared_ptr<Source>&,
        const std::vector<std::uint64_t>&, const std::vector<std::size_t>&);
    std::string image() const;
    std::string source_image() const;
    PhaseFoldInfo info(std::size_t, bool) const;
    std::shared_ptr<PhaseFoldArchive> prefix(std::size_t) const;
    std::shared_ptr<PhaseFoldArchive> propose(std::uint64_t, std::size_t, const std::vector<std::string>&, bool) const;
    std::shared_ptr<PhaseFoldArchive> apply(std::uint64_t) const;
    bool same_carrier(const std::shared_ptr<PhaseFoldArchive>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit PhaseFoldArchive(std::shared_ptr<const State>);
};
}
