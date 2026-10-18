#pragma once
#include "finite_field_bridge.hpp"
#include "literal_input_bridge.hpp"
#include <rfc/learning/relational_literal_finite_field_codec.hpp>

namespace rexgraph::python {
struct LiteralFiniteRule {
    std::uint64_t key{};
    std::vector<std::uint64_t> bindings, symbols;
    bool has_target{};
    std::size_t target{};
    std::string unsupported;
    std::vector<std::shared_ptr<LiteralInputBridge>> inputs;
};
struct LiteralFiniteDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<LiteralFiniteRule> rules;
};
inline std::string literal_use_value(std::shared_ptr<InputUseAssetBridge> asset,
    std::uint64_t use, const std::vector<std::size_t>& bounds, std::size_t maximum) {
    if (!asset) throw std::invalid_argument("actual native byte asset required");
    return program_bridge_detail::bounded(rfc::learning::encode_relational_literal_use_value(
        asset->native_owner(), use, maximum, program_bridge_detail::limits(bounds)), maximum);
}
class LiteralFiniteResultBridge;
class LiteralFiniteMapBridge {
public:
    LiteralFiniteMapBridge(std::shared_ptr<FiniteSourceBridge> source,
        const std::vector<FiniteTarget>& targets, const LiteralFiniteDeclaration& declaration,
        const std::vector<std::size_t>& bounds, std::size_t maximum): bounds_(bounds), maximum_(maximum) {
        if (!source) throw std::invalid_argument("actual finite source owner required");
        std::vector<rfc::learning::RelationalFiniteFieldTarget> native_targets;
        for (const auto& value : targets) native_targets.push_back(finite_bridge_detail::target(value));
        rfc::learning::RelationalLiteralFiniteFieldDeclaration d{declaration.key, declaration.revision, {}};
        for (const auto& row : declaration.rules) {
            rfc::learning::RelationalLiteralFiniteFieldRule rule;
            rule.key = row.key; rule.bindings = row.bindings; rule.symbols = row.symbols;
            rule.unsupported = row.unsupported;
            if (row.has_target) rule.target = row.target;
            for (const auto& input : row.inputs) {
                if (!input) throw std::invalid_argument("actual native literal derivation required");
                rule.inputs.push_back(input->native_owner());
            }
            d.rules.push_back(std::move(rule));
        }
        owner_ = std::make_shared<const rfc::learning::PreparedRelationalLiteralFiniteFieldMap>(
            source->native_owner(), std::move(native_targets), std::move(d), program_bridge_detail::limits(bounds));
    }
    static std::shared_ptr<LiteralFiniteMapBridge> adopt(rfc::learning::PreparedRelationalLiteralFiniteFieldMap value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<LiteralFiniteMapBridge>(new LiteralFiniteMapBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<LiteralFiniteMapBridge> restore(const std::string& image,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return adopt(rfc::learning::decode_relational_literal_finite_field_map(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds, maximum), program_bridge_detail::limits(bounds)), bounds, maximum);
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::learning::encode_relational_literal_finite_field_map(*owner_, maximum_), maximum_);
    }
    const rfc::learning::PreparedRelationalLiteralFiniteFieldMap& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedRelationalLiteralFiniteFieldMap& native_owner() const && = delete;
    std::shared_ptr<FiniteSourceBridge> source() const { return FiniteSourceBridge::adopt(owner_->source(), bounds_, maximum_); }
    std::shared_ptr<FiniteMapBridge> raw_map() const { return FiniteMapBridge::adopt(owner_->map(), bounds_, maximum_); }
    bool same_owner(const LiteralFiniteMapBridge& other) const { return owner_->same_owner(*other.owner_); }
    LiteralFiniteDeclaration info(std::size_t maximum) const {
        const auto& d = owner_->declaration();
        LiteralFiniteDeclaration out{d.key, d.revision, {}};
        std::size_t count = d.rules.size(); evidence_bridge_detail::bound(count, maximum);
        for (const auto& r : d.rules) {
            count = rfc::machine::checked_size_add(count, r.bindings.size());
            count = rfc::machine::checked_size_add(count, r.symbols.size());
            count = rfc::machine::checked_size_add(count, r.inputs.size());
            count = rfc::machine::checked_size_add(count, r.unsupported.size());
            evidence_bridge_detail::bound(count, maximum);
            LiteralFiniteRule rule;
            rule.key = r.key; rule.bindings = r.bindings; rule.symbols = r.symbols;
            rule.has_target = r.target.has_value(); rule.target = r.target.value_or(0); rule.unsupported = r.unsupported;
            for (const auto& literal : r.inputs) rule.inputs.push_back(LiteralInputBridge::adopt(literal, bounds_, maximum_));
            out.rules.push_back(std::move(rule));
        }
        return out;
    }
    std::shared_ptr<LiteralFiniteResultBridge> execute() const;
private:
    LiteralFiniteMapBridge(rfc::learning::PreparedRelationalLiteralFiniteFieldMap value,
        std::vector<std::size_t> bounds, std::size_t maximum):
        owner_(std::make_shared<const rfc::learning::PreparedRelationalLiteralFiniteFieldMap>(std::move(value))),
        bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedRelationalLiteralFiniteFieldMap> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class LiteralFiniteResultBridge {
public:
    static std::shared_ptr<LiteralFiniteResultBridge> restore(const std::string& image,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return adopt(rfc::learning::decode_relational_literal_finite_field_result(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds, maximum), program_bridge_detail::limits(bounds)), bounds, maximum);
    }
    static std::shared_ptr<LiteralFiniteResultBridge> adopt(rfc::learning::RelationalLiteralFiniteFieldResult value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<LiteralFiniteResultBridge>(new LiteralFiniteResultBridge(std::move(value), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::learning::encode_relational_literal_finite_field_result(*owner_, maximum_), maximum_);
    }
    std::shared_ptr<LiteralFiniteMapBridge> map() const { return LiteralFiniteMapBridge::adopt(owner_->map(), bounds_, maximum_); }
    std::shared_ptr<FiniteResultBridge> family() const { return FiniteResultBridge::adopt(owner_->family(), bounds_, maximum_); }
private:
    LiteralFiniteResultBridge(rfc::learning::RelationalLiteralFiniteFieldResult value,
        std::vector<std::size_t> bounds, std::size_t maximum):
        owner_(std::make_shared<const rfc::learning::RelationalLiteralFiniteFieldResult>(std::move(value))),
        bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalLiteralFiniteFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<LiteralFiniteResultBridge> LiteralFiniteMapBridge::execute() const {
    return LiteralFiniteResultBridge::adopt(owner_->execute(program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
