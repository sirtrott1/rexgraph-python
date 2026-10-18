#pragma once

#include "field_interface_bridge.hpp"
#include <rfc/inference/relational_literal_input_codec.hpp>

namespace rexgraph::python {
struct LiteralCoordinate {
    std::uint64_t key{}, direction{}, use{};
    std::size_t grade{}, cell{}, span_index{}, begin{}, end{};
    unsigned kind{};
    std::string operand;
};
struct LiteralDeclaration {
    std::uint64_t key{}, revision{}, use{};
    std::vector<LiteralCoordinate> coordinates;
};
class LiteralInputBridge {
public:
    LiteralInputBridge(std::shared_ptr<InputAccessionBridge> accession,
            const LiteralDeclaration& value, const std::vector<std::size_t>& bounds,
            std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        if (!accession) throw std::invalid_argument("actual native input accession owner required");
        rfc::inference::RelationalLiteralInputDeclaration declaration;
        declaration.key = value.key; declaration.revision = value.revision; declaration.use = value.use;
        for (const auto& row : value.coordinates) {
            rfc::inference::RelationalLiteralCoordinate coordinate;
            coordinate.key = row.key; coordinate.cell = program_bridge_detail::cell(row.grade, row.cell);
            coordinate.direction = row.direction; coordinate.use = row.use; coordinate.span_index = row.span_index;
            coordinate.span = {row.begin, row.end};
            coordinate.kind = static_cast<rfc::inference::RelationalLiteralPredicateKind>(row.kind);
            const auto bytes = program_bridge_detail::bytes(row.operand);
            coordinate.operand.assign(bytes.begin(), bytes.end());
            declaration.coordinates.push_back(std::move(coordinate));
        }
        owner_ = std::make_shared<const rfc::inference::OwnedRelationalLiteralInput>(
            accession->native_owner(), declaration, program_bridge_detail::limits(bounds));
    }
    static std::shared_ptr<LiteralInputBridge> restore(const std::string& image,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        auto owner = rfc::inference::decode_relational_literal_input(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds, maximum), program_bridge_detail::limits(bounds));
        return std::shared_ptr<LiteralInputBridge>(new LiteralInputBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<LiteralInputBridge> adopt(rfc::inference::OwnedRelationalLiteralInput value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<LiteralInputBridge>(new LiteralInputBridge(std::move(value), bounds, maximum));
    }
    const rfc::inference::OwnedRelationalLiteralInput& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalLiteralInput& native_owner() const && = delete;
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_literal_input(*owner_, maximum_), maximum_);
    }
    LiteralDeclaration info(std::size_t maximum) const {
        const auto& d = owner_->declaration();
        if (d.coordinates.size() > maximum) throw std::length_error("literal coordinate metadata bound");
        LiteralDeclaration out; out.key = d.key; out.revision = d.revision; out.use = d.use;
        for (const auto& c : d.coordinates)
            out.coordinates.push_back({c.key, c.direction, c.use, c.cell.grade.value, c.cell.index,
                c.span_index, c.span.begin, c.span.end, static_cast<unsigned>(c.kind),
                program_bridge_detail::text(c.operand)});
        return out;
    }
    std::shared_ptr<AccessionInputBridge> input() const {
        return AccessionInputBridge::adopt(owner_->input(), bounds_, maximum_);
    }
    std::shared_ptr<InputUseAssetBridge> asset() const {
        return InputUseAssetBridge::adopt(owner_->input().accession().asset(), bounds_, maximum_);
    }
    std::shared_ptr<FieldProgramBridge> program() const {
        return FieldProgramBridge::adopt(owner_->input().accession().program(), bounds_, maximum_);
    }
    bool same_owner(const LiteralInputBridge& other) const { return owner_->same_owner(*other.owner_); }
private:
    LiteralInputBridge(rfc::inference::OwnedRelationalLiteralInput value,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalLiteralInput>(std::move(value))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalLiteralInput> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
} // namespace rexgraph::python
