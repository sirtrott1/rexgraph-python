#pragma once
#include "finite_field_bridge.hpp"
#include "field_evidence_admission_bridge.hpp"
#include <rfc/learning/relational_available_finite_field_codec.hpp>

namespace rexgraph::python {
struct AvailableFiniteTarget {
    std::shared_ptr<AvailableProgramBridge> fixed;
    std::shared_ptr<AvailableEvidenceBridge> affine;
};
struct AvailableFiniteInput {
    std::shared_ptr<InputUseAssetBridge> asset;
    InterfaceAccession accession;
    std::uint64_t use{};
    std::string field;
    std::vector<std::uint64_t> origins;
};
struct AvailableFiniteRule {
    std::uint64_t key{};
    std::vector<std::uint64_t> bindings, symbols;
    bool has_target{};
    std::size_t target{};
    std::string unsupported;
    std::vector<AvailableFiniteInput> inputs;
};
struct AvailableFiniteMapDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<AvailableFiniteRule> rules;
};
namespace available_finite_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
using available_evidence_bridge_detail::authority;
inline learning::RelationalFiniteRelationshipStatus status(unsigned value) {
    if (value > 1) throw std::invalid_argument("finite relationship requires observed or prospective status");
    return static_cast<learning::RelationalFiniteRelationshipStatus>(value);
}
inline learning::RelationalFiniteSourceOrigins source_keys(const std::vector<std::uint64_t>& values) {
    if (values.size() != 2) throw std::invalid_argument("finite asset and constraint origin keys required");
    return {values[0], values[1]};
}
inline learning::RelationalFiniteMapOrigins map_keys(const std::vector<std::uint64_t>& values) {
    if (values.size() != 2) throw std::invalid_argument("finite Maps and relationship origin keys required");
    return {values[0], values[1]};
}
inline learning::AvailableRelationalFiniteFieldTarget target(const AvailableFiniteTarget& value) {
    if (bool(value.fixed) == bool(value.affine)) throw std::invalid_argument("one actual admitted fixed or evidence target required");
    if (value.fixed) return value.fixed->native_owner();
    return value.affine->native_owner();
}
inline std::vector<learning::AvailableRelationalFiniteFieldTarget> targets(const std::vector<AvailableFiniteTarget>& rows,
        const std::vector<std::size_t>& bounds) {
    evidence_bridge_detail::bound(rows.size(), bounds.at(0));
    std::vector<learning::AvailableRelationalFiniteFieldTarget> out;
    for (const auto& row : rows) out.push_back(target(row));
    return out;
}
inline learning::RelationalAvailableFiniteFieldMapDeclaration declaration(const AvailableFiniteMapDeclaration& value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
    numeric::ExactActionGuard guard(limits(bounds)); guard.entries(value.rules.size());
    learning::RelationalAvailableFiniteFieldMapDeclaration out{value.key, value.revision, {}};
    for (const auto& row : value.rules) {
        guard.dimension(row.bindings.size()); guard.dimension(row.symbols.size()); guard.dimension(row.inputs.size());
        learning::RelationalAvailableFiniteFieldRule rule{row.key, row.bindings, row.symbols,
            row.has_target ? std::optional<std::size_t>{row.target} : std::nullopt, {}, row.unsupported};
        for (const auto& input : row.inputs) {
            if (!input.asset) throw std::invalid_argument("actual finite source asset owner required for raw input submission");
            rule.inputs.push_back({{input.asset->native_owner(), interface_bridge_detail::accession_declaration(input.accession, guard),
                input.use, interface_bridge_detail::field(input.field, bounds, maximum)}, admission_bridge_detail::input_keys(input.origins)});
        }
        out.rules.push_back(std::move(rule));
    }
    return out;
}
inline std::vector<std::string> receipts(const std::vector<inference::OwnedRelationalValueOrigin>& rows,
        std::size_t records, std::size_t maximum) {
    evidence_bridge_detail::bound(rows.size(), records); std::vector<std::string> out;
    for (const auto& row : rows) out.push_back(bounded(row.receipt(), maximum));
    return out;
}
} // namespace available_finite_bridge_detail

class AvailableFiniteSectionsBridge;
class AvailableFiniteSourceBridge {
public:
    AvailableFiniteSourceBridge(std::shared_ptr<InputUseAssetBridge> asset, const FiniteDeclaration& declaration,
            unsigned status, const std::vector<std::uint64_t>& origins, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace available_finite_bridge_detail;
        if (!asset) throw std::invalid_argument("actual finite source asset owner required");
        owner_ = std::make_shared<const learning::PreparedAvailableRelationalFiniteSource>(asset->native_owner(),
            finite_bridge_detail::declaration(declaration), available_finite_bridge_detail::status(status), source_keys(origins), authority(context), limits(bounds));
    }
    static std::shared_ptr<AvailableFiniteSourceBridge> adopt(rfc::learning::PreparedAvailableRelationalFiniteSource value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableFiniteSourceBridge>(new AvailableFiniteSourceBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableFiniteSourceBridge> restore(const std::string& image, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        return adopt(learning::decode_available_relational_finite_source(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    static std::string constraint_payload(std::shared_ptr<InputUseAssetBridge> asset, const FiniteDeclaration& declaration,
            unsigned status, std::size_t maximum) {
        if (!asset) throw std::invalid_argument("actual finite source asset owner required");
        const auto image = rfc::inference::encode_relational_input_use_asset(asset->native_owner());
        return program_bridge_detail::bounded(rfc::learning::relational_finite_constraint_payload(image,
            finite_bridge_detail::declaration(declaration), available_finite_bridge_detail::status(status)), maximum);
    }
    const rfc::learning::PreparedAvailableRelationalFiniteSource& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedAvailableRelationalFiniteSource& native_owner() const && = delete;
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_source(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<FiniteSourceBridge> raw() const { return FiniteSourceBridge::adopt(owner_->finite(), bounds_, maximum_); }
    unsigned status() const { return static_cast<unsigned>(owner_->status()); }
    std::vector<std::uint64_t> origins() const { const auto keys = owner_->origins(); return {keys.asset, keys.constraints}; }
    bool same_owner(const AvailableFiniteSourceBridge& other) const { return owner_->same_owner(*other.owner_); }
    std::vector<std::string> receipts(std::size_t maximum) const { return available_finite_bridge_detail::receipts(owner_->obligations(), maximum, maximum_); }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        evidence_bridge_detail::bound(rfc::machine::checked_size_add(owner_->obligations().size(), 1), maximum);
        auto out = admission_bridge_detail::dependencies(owner_->obligations(), maximum_);
        out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->finite().asset().history()), maximum_)); return out;
    }
    std::shared_ptr<AvailableFiniteSectionsBridge> sections(std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableFiniteSourceBridge(rfc::learning::PreparedAvailableRelationalFiniteSource value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::PreparedAvailableRelationalFiniteSource>(std::move(value))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedAvailableRelationalFiniteSource> owner_;
    std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
class AvailableFiniteSectionBridge;
class AvailableFiniteSectionsBridge {
public:
    static std::shared_ptr<AvailableFiniteSectionsBridge> adopt(rfc::learning::AvailableRelationalFiniteSections value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableFiniteSectionsBridge>(new AvailableFiniteSectionsBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableFiniteSectionsBridge> restore(const std::string& image, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        return adopt(learning::decode_available_relational_finite_sections(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_sections(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteSourceBridge> source() const { return AvailableFiniteSourceBridge::adopt(owner_->source(), bounds_, maximum_); }
    std::shared_ptr<FiniteSectionsBridge> raw() const { return std::make_shared<FiniteSectionsBridge>(owner_->sections(), bounds_, maximum_); }
    std::vector<std::shared_ptr<AvailableFiniteSectionBridge>> select(std::size_t begin, std::size_t count, std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableFiniteSectionsBridge(rfc::learning::AvailableRelationalFiniteSections value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::AvailableRelationalFiniteSections>(std::move(value))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalFiniteSections> owner_;
    std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
class AvailableFiniteSectionBridge {
public:
    AvailableFiniteSectionBridge(rfc::learning::AvailableRelationalFiniteSection value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::move(value)), bounds_(std::move(bounds)), maximum_(maximum) {}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_.admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteSectionsBridge> family() const { return AvailableFiniteSectionsBridge::adopt(owner_.parent(), bounds_, maximum_); }
    std::size_t ordinal() const { return owner_.ordinal(); }
    std::vector<std::uint64_t> assignment(std::size_t maximum) const { evidence_bridge_detail::bound(owner_.assignment().size(), maximum); return owner_.assignment(); }
private:
    rfc::learning::AvailableRelationalFiniteSection owner_; std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
inline std::shared_ptr<AvailableFiniteSectionsBridge> AvailableFiniteSourceBridge::sections(std::shared_ptr<AvailabilityContextBridge> context) const {
    return AvailableFiniteSectionsBridge::adopt(owner_->sections(available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
inline std::vector<std::shared_ptr<AvailableFiniteSectionBridge>> AvailableFiniteSectionsBridge::select(std::size_t begin, std::size_t count,
        std::shared_ptr<AvailabilityContextBridge> context) const {
    const auto rows = owner_->select(begin, count, available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_));
    std::vector<std::shared_ptr<AvailableFiniteSectionBridge>> out;
    for (const auto& row : rows) { out.push_back(std::make_shared<AvailableFiniteSectionBridge>(row, bounds_, maximum_)); }
    return out;
}
class AvailableFiniteResultBridge;
class AvailableFiniteMapBridge {
public:
    AvailableFiniteMapBridge(std::shared_ptr<AvailableFiniteSourceBridge> source, const std::vector<AvailableFiniteTarget>& targets,
            const AvailableFiniteMapDeclaration& declaration, unsigned status, const std::vector<std::uint64_t>& origins,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace available_finite_bridge_detail;
        if (!source) throw std::invalid_argument("actual admitted finite source required");
        owner_ = std::make_shared<const learning::PreparedAvailableRelationalFiniteFieldMap>(source->native_owner(),
            available_finite_bridge_detail::targets(targets, bounds), available_finite_bridge_detail::declaration(declaration, bounds, maximum),
            available_finite_bridge_detail::status(status), map_keys(origins), authority(context), limits(bounds));
    }
    static std::shared_ptr<AvailableFiniteMapBridge> adopt(rfc::learning::PreparedAvailableRelationalFiniteFieldMap value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableFiniteMapBridge>(new AvailableFiniteMapBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableFiniteMapBridge> restore(const std::string& image, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        return adopt(learning::decode_available_relational_finite_field_map(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    static std::string interpretation_payload(std::shared_ptr<AvailableFiniteSourceBridge> source, const std::vector<AvailableFiniteTarget>& targets,
            const AvailableFiniteMapDeclaration& declaration, unsigned status, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        if (!source) throw std::invalid_argument("actual admitted finite source required");
        return bounded(learning::relational_finite_interpretation_payload(source->native_owner(), available_finite_bridge_detail::targets(targets, bounds),
            available_finite_bridge_detail::declaration(declaration, bounds, maximum), available_finite_bridge_detail::status(status), authority(context)), maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_field_map(*owner_), maximum_); }
    const rfc::learning::PreparedAvailableRelationalFiniteFieldMap& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedAvailableRelationalFiniteFieldMap& native_owner() const && = delete;
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteSourceBridge> source() const { return AvailableFiniteSourceBridge::adopt(owner_->source(), bounds_, maximum_); }
    std::shared_ptr<FiniteMapBridge> raw() const { return FiniteMapBridge::adopt(owner_->map(), bounds_, maximum_); }
    unsigned status() const { return static_cast<unsigned>(owner_->status()); }
    std::vector<std::uint64_t> origins() const { const auto keys = owner_->origins(); return {keys.maps, keys.relationship}; }
    std::vector<std::string> receipts(std::size_t maximum) const { return available_finite_bridge_detail::receipts(owner_->obligations(), maximum, maximum_); }
    bool same_owner(const AvailableFiniteMapBridge& other) const { return owner_->same_owner(*other.owner_); }
    std::vector<AvailableFiniteTarget> targets(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_->targets().size(), maximum); std::vector<AvailableFiniteTarget> out;
        for (const auto& target : owner_->targets()) {
            if (const auto* fixed = std::get_if<rfc::inference::PreparedAvailableRelationalProgram>(&target)) out.push_back({AvailableProgramBridge::adopt(*fixed, bounds_, maximum_), {}});
            else out.push_back({{}, AvailableEvidenceBridge::adopt(std::get<rfc::learning::PreparedAvailableRelationalEvidence>(target), bounds_, maximum_)});
        }
        return out;
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_->obligations().size(), maximum);
        auto out = admission_bridge_detail::dependencies(owner_->obligations(), maximum_);
        evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), 1), maximum);
        const auto source_history = rfc::storage::encode_native_series(owner_->source().finite().asset().history());
        out.push_back(program_bridge_detail::bounded(source_history, maximum_));
        for (const auto& target : targets(maximum)) {
            const auto rows = target.fixed ? target.fixed->dependency_archives(maximum) : target.affine->dependency_archives(maximum);
            evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), rows.size()), maximum); out.insert(out.end(), rows.begin(), rows.end());
        }
        return out;
    }
    std::shared_ptr<AvailableFiniteResultBridge> execute(std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableFiniteMapBridge(rfc::learning::PreparedAvailableRelationalFiniteFieldMap value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::PreparedAvailableRelationalFiniteFieldMap>(std::move(value))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedAvailableRelationalFiniteFieldMap> owner_;
    std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
class AvailableFiniteBranchBridge;
class AvailableFiniteEvaluationBridge;
class AvailableFiniteResultBridge {
public:
    static std::shared_ptr<AvailableFiniteResultBridge> adopt(rfc::learning::AvailableRelationalFiniteFieldResult value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableFiniteResultBridge>(new AvailableFiniteResultBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableFiniteResultBridge> restore(const std::string& image, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        return adopt(learning::decode_available_relational_finite_field_result(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_field_result(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteMapBridge> model() const { return AvailableFiniteMapBridge::adopt(owner_->model(), bounds_, maximum_); }
    std::shared_ptr<FiniteResultBridge> raw() const { return FiniteResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::size_t size() const { return owner_->size(); }
    std::string completion_payload(std::size_t ordinal, const std::vector<std::string>& free, std::uint64_t setting,
            std::shared_ptr<AvailabilityContextBridge> context) const {
        using namespace available_finite_bridge_detail;
        return bounded(learning::relational_finite_completion_payload(*owner_, ordinal, available_evidence_bridge_detail::free_values(free, bounds_), setting, authority(context)), maximum_);
    }
    std::vector<std::shared_ptr<AvailableFiniteBranchBridge>> select(std::size_t begin, std::size_t count, std::shared_ptr<AvailabilityContextBridge> context) const;
    std::shared_ptr<AvailableFiniteEvaluationBridge> evaluate(std::size_t ordinal, const std::vector<std::string>& free, std::uint64_t setting,
        std::uint64_t choice, std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableFiniteResultBridge(rfc::learning::AvailableRelationalFiniteFieldResult value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::AvailableRelationalFiniteFieldResult>(std::move(value))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalFiniteFieldResult> owner_;
    std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
class AvailableFiniteBranchBridge {
public:
    AvailableFiniteBranchBridge(rfc::learning::AvailableRelationalFiniteFieldBranch value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::move(value)), bounds_(std::move(bounds)), maximum_(maximum) {}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_.admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteResultBridge> family() const { return AvailableFiniteResultBridge::adopt(owner_.parent(), bounds_, maximum_); }
    std::size_t ordinal() const { return owner_.ordinal(); }
    std::shared_ptr<FiniteBranchBridge> raw() const { return std::make_shared<FiniteBranchBridge>(owner_.parent().result(), owner_.ordinal(), bounds_, maximum_); }
private:
    rfc::learning::AvailableRelationalFiniteFieldBranch owner_; std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
class AvailableFiniteEvaluationBridge {
public:
    static std::shared_ptr<AvailableFiniteEvaluationBridge> adopt(rfc::learning::AvailableRelationalFiniteFieldEvaluation value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableFiniteEvaluationBridge>(new AvailableFiniteEvaluationBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableFiniteEvaluationBridge> restore(const std::string& image, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_finite_bridge_detail;
        return adopt(learning::decode_available_relational_finite_field_evaluation(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_field_evaluation(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_finite_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableFiniteResultBridge> family() const { return AvailableFiniteResultBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<FiniteEvaluationBridge> raw() const { return FiniteEvaluationBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::size_t ordinal() const { return owner_->ordinal(); }
    std::string setting_receipt() const { const auto setting = owner_->parameter_setting(); return setting ? program_bridge_detail::bounded(setting->receipt(), maximum_) : std::string{}; }
    std::string choice_receipt() const { const auto origin = owner_->origin(); return program_bridge_detail::bounded(origin.receipt(), maximum_); }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out = family()->model()->dependency_archives(maximum);
        const auto setting = owner_->parameter_setting(); const auto extra = setting ? 2u : 1u;
        evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), extra), maximum);
        if (setting) out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(setting->publication().history()), maximum_));
        out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->origin().publication().history()), maximum_)); return out;
    }
private:
    AvailableFiniteEvaluationBridge(rfc::learning::AvailableRelationalFiniteFieldEvaluation value, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::AvailableRelationalFiniteFieldEvaluation>(std::move(value))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalFiniteFieldEvaluation> owner_;
    std::vector<std::size_t> bounds_; std::size_t maximum_{};
};
inline std::shared_ptr<AvailableFiniteResultBridge> AvailableFiniteMapBridge::execute(std::shared_ptr<AvailabilityContextBridge> context) const {
    return AvailableFiniteResultBridge::adopt(owner_->execute(available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
inline std::vector<std::shared_ptr<AvailableFiniteBranchBridge>> AvailableFiniteResultBridge::select(std::size_t begin, std::size_t count,
        std::shared_ptr<AvailabilityContextBridge> context) const {
    const auto rows = owner_->select(begin, count, available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_));
    std::vector<std::shared_ptr<AvailableFiniteBranchBridge>> out;
    for (const auto& row : rows) { out.push_back(std::make_shared<AvailableFiniteBranchBridge>(row, bounds_, maximum_)); }
    return out;
}
inline std::shared_ptr<AvailableFiniteEvaluationBridge> AvailableFiniteResultBridge::evaluate(std::size_t ordinal, const std::vector<std::string>& free,
        std::uint64_t setting, std::uint64_t choice, std::shared_ptr<AvailabilityContextBridge> context) const {
    return AvailableFiniteEvaluationBridge::adopt(owner_->evaluate(ordinal, available_evidence_bridge_detail::free_values(free, bounds_), setting,
        choice, available_finite_bridge_detail::authority(context), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
