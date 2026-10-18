#pragma once
#include "finite_field_admission_bridge.hpp"
#include "literal_finite_field_bridge.hpp"
#include <rfc/learning/relational_available_literal_finite_field_codec.hpp>

namespace rexgraph::python {
using LiteralOriginRows = std::vector<std::vector<std::uint64_t>>;
class AvailableLiteralResultBridge;
class AvailableLiteralBranchBridge;
class AvailableLiteralEvaluationBridge;
class AvailableLiteralMapBridge {
public:
    AvailableLiteralMapBridge(std::shared_ptr<AvailableFiniteMapBridge> base,
        std::shared_ptr<LiteralFiniteMapBridge> reader, const LiteralOriginRows& inputs,
        std::uint64_t interface, std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        if (!base || !reader) throw std::invalid_argument("actual available map and literal reader required");
        owner_ = std::make_shared<const rfc::learning::PreparedAvailableRelationalLiteralFiniteFieldMap>(
            base->native_owner(), reader->native_owner(), rfc::learning::RelationalLiteralFiniteFieldOrigins{inputs, interface},
            available_finite_bridge_detail::authority(context), maximum);
    }
    static std::string interface_payload(std::shared_ptr<AvailableFiniteMapBridge> base,
        std::shared_ptr<LiteralFiniteMapBridge> reader, const LiteralOriginRows& inputs,
        std::shared_ptr<AvailabilityContextBridge> context, std::size_t maximum) {
        if (!base || !reader) throw std::invalid_argument("actual available map and literal reader required");
        return program_bridge_detail::bounded(rfc::learning::relational_literal_finite_interface_payload(
            base->native_owner(), reader->native_owner(), inputs, available_finite_bridge_detail::authority(context), maximum), maximum);
    }
    static std::shared_ptr<AvailableLiteralMapBridge> adopt(rfc::learning::PreparedAvailableRelationalLiteralFiniteFieldMap value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableLiteralMapBridge>(new AvailableLiteralMapBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableLiteralMapBridge> restore(const std::string& image,
        std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return adopt(rfc::learning::decode_available_relational_literal_finite_field_map(program_bridge_detail::bytes(image),
            available_finite_bridge_detail::authority(context), program_bridge_detail::reader(bounds, maximum),
            program_bridge_detail::limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(
        rfc::learning::encode_available_relational_literal_finite_field_map(*owner_, maximum_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    bool same_owner(const AvailableLiteralMapBridge& other) const { return owner_->same_owner(*other.owner_); }
    std::shared_ptr<AvailableFiniteMapBridge> base() const { return AvailableFiniteMapBridge::adopt(owner_->base(), bounds_, maximum_); }
    std::shared_ptr<LiteralFiniteMapBridge> reader() const { return LiteralFiniteMapBridge::adopt(owner_->reader(), bounds_, maximum_); }
    LiteralOriginRows inputs(std::size_t maximum) const {
        const auto pin = *owner_; const auto& rows = pin.origins().inputs;
        std::size_t count = rows.size(); evidence_bridge_detail::bound(count, maximum);
        for (const auto& row : rows) { count = rfc::machine::checked_size_add(count, row.size()); evidence_bridge_detail::bound(count, maximum); }
        return rows;
    }
    std::uint64_t interface() const { return owner_->origins().interface; }
    std::vector<std::string> dependencies(std::size_t maximum) const {
        const auto pin = *owner_; evidence_bridge_detail::bound(pin.obligations().size(), maximum);
        return admission_bridge_detail::dependencies(pin.obligations(), maximum_);
    }
    std::shared_ptr<AvailableLiteralResultBridge> execute(std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableLiteralMapBridge(rfc::learning::PreparedAvailableRelationalLiteralFiniteFieldMap value,
        std::vector<std::size_t> bounds, std::size_t maximum) :
        owner_(std::make_shared<const rfc::learning::PreparedAvailableRelationalLiteralFiniteFieldMap>(std::move(value))),
        bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedAvailableRelationalLiteralFiniteFieldMap> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableLiteralResultBridge {
public:
    static std::shared_ptr<AvailableLiteralResultBridge> adopt(rfc::learning::AvailableRelationalLiteralFiniteFieldResult value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableLiteralResultBridge>(new AvailableLiteralResultBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableLiteralResultBridge> restore(const std::string& image,
        std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return adopt(rfc::learning::decode_available_relational_literal_finite_field_result(program_bridge_detail::bytes(image),
            available_finite_bridge_detail::authority(context), program_bridge_detail::reader(bounds, maximum),
            program_bridge_detail::limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(
        rfc::learning::encode_available_relational_literal_finite_field_result(*owner_, maximum_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableLiteralMapBridge> model() const { return AvailableLiteralMapBridge::adopt(owner_->model(), bounds_, maximum_); }
    std::shared_ptr<FiniteResultBridge> raw() const { return FiniteResultBridge::adopt(owner_->raw_result(), bounds_, maximum_); }
    std::size_t size() const { return owner_->size(); }
    std::vector<std::shared_ptr<AvailableLiteralBranchBridge>> select(std::size_t begin, std::size_t count,
        std::shared_ptr<AvailabilityContextBridge> context) const;
    std::string completion_payload(std::size_t ordinal, const std::vector<std::string>& values, std::uint64_t setting,
        std::shared_ptr<AvailabilityContextBridge> context) const {
        return program_bridge_detail::bounded(rfc::learning::relational_literal_finite_completion_payload(*owner_, ordinal,
            available_evidence_bridge_detail::free_values(values, bounds_), setting,
            available_finite_bridge_detail::authority(context), maximum_), maximum_);
    }
    std::shared_ptr<AvailableLiteralEvaluationBridge> evaluate(std::size_t ordinal, const std::vector<std::string>& values,
        std::uint64_t setting, std::uint64_t choice, std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableLiteralResultBridge(rfc::learning::AvailableRelationalLiteralFiniteFieldResult value,
        std::vector<std::size_t> bounds, std::size_t maximum) :
        owner_(std::make_shared<const rfc::learning::AvailableRelationalLiteralFiniteFieldResult>(std::move(value))),
        bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalLiteralFiniteFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableLiteralBranchBridge {
public:
    AvailableLiteralBranchBridge(rfc::learning::AvailableRelationalLiteralFiniteFieldBranch owner,
        std::vector<std::size_t> bounds, std::size_t maximum) : owner_(std::move(owner)), bounds_(std::move(bounds)), maximum_(maximum) {}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_.admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableLiteralResultBridge> family() const { return AvailableLiteralResultBridge::adopt(owner_.parent(), bounds_, maximum_); }
    std::shared_ptr<FiniteBranchBridge> raw() const { return std::make_shared<FiniteBranchBridge>(owner_.parent().raw_result(), owner_.branch().ordinal, bounds_, maximum_); }
private:
    rfc::learning::AvailableRelationalLiteralFiniteFieldBranch owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableLiteralEvaluationBridge {
public:
    static std::shared_ptr<AvailableLiteralEvaluationBridge> adopt(rfc::learning::AvailableRelationalLiteralFiniteFieldEvaluation value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableLiteralEvaluationBridge>(new AvailableLiteralEvaluationBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableLiteralEvaluationBridge> restore(const std::string& image,
        std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return adopt(rfc::learning::decode_available_relational_literal_finite_field_evaluation(program_bridge_detail::bytes(image),
            available_finite_bridge_detail::authority(context), program_bridge_detail::reader(bounds, maximum),
            program_bridge_detail::limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(
        rfc::learning::encode_available_relational_literal_finite_field_evaluation(*owner_, maximum_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableLiteralResultBridge> family() const { return AvailableLiteralResultBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<FiniteEvaluationBridge> raw() const { return FiniteEvaluationBridge::adopt(owner_->raw_result(), bounds_, maximum_); }
    std::string setting_receipt() const {
        const auto value = owner_->parameter_setting(); return value ? program_bridge_detail::bounded(value->receipt(), maximum_) : std::string{};
    }
    std::string choice_receipt() const {
        const auto value = owner_->origin(); return program_bridge_detail::bounded(value.receipt(), maximum_);
    }
    std::vector<std::string> dependencies(std::size_t maximum) const {
        const auto pin = *owner_; const auto family = pin.family(); const auto model = family.model();
        auto all = model.obligations(); const auto setting = pin.parameter_setting();
        if (setting) all.push_back(*setting);
        all.push_back(pin.origin()); evidence_bridge_detail::bound(all.size(), maximum);
        return admission_bridge_detail::dependencies(all, maximum_);
    }
private:
    AvailableLiteralEvaluationBridge(rfc::learning::AvailableRelationalLiteralFiniteFieldEvaluation value,
        std::vector<std::size_t> bounds, std::size_t maximum) :
        owner_(std::make_shared<const rfc::learning::AvailableRelationalLiteralFiniteFieldEvaluation>(std::move(value))),
        bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalLiteralFiniteFieldEvaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AvailableLiteralResultBridge> AvailableLiteralMapBridge::execute(std::shared_ptr<AvailabilityContextBridge> context) const {
    return AvailableLiteralResultBridge::adopt(owner_->execute(available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
inline std::vector<std::shared_ptr<AvailableLiteralBranchBridge>> AvailableLiteralResultBridge::select(std::size_t begin,
    std::size_t count, std::shared_ptr<AvailabilityContextBridge> context) const {
    const auto selected = owner_->select(begin, count, available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_));
    std::vector<std::shared_ptr<AvailableLiteralBranchBridge>> out;
    for (const auto& branch : selected) out.push_back(std::make_shared<AvailableLiteralBranchBridge>(branch, bounds_, maximum_));
    return out;
}
inline std::shared_ptr<AvailableLiteralEvaluationBridge> AvailableLiteralResultBridge::evaluate(std::size_t ordinal,
    const std::vector<std::string>& values, std::uint64_t setting, std::uint64_t choice,
    std::shared_ptr<AvailabilityContextBridge> context) const {
    return AvailableLiteralEvaluationBridge::adopt(owner_->evaluate(ordinal,
        available_evidence_bridge_detail::free_values(values, bounds_), setting, choice,
        available_finite_bridge_detail::authority(context), maximum_, program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
