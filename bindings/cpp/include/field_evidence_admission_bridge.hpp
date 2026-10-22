#pragma once

#include "field_admission_bridge.hpp"
#include "field_evidence_bridge.hpp"
#include <rfc/learning/relational_available_evidence_codec.hpp>

namespace rexgraph::python {
struct AvailableEvidenceObservation {
    std::uint64_t key{}, target_use{};
    std::vector<AdmissionInput> inputs;
    std::string target_asset, target;
    std::vector<std::vector<std::size_t>> observed;
    std::vector<std::uint64_t> origins;
    std::shared_ptr<OccurrenceTargetBridge> port;
    std::vector<std::uint64_t> observed_coordinates;
};
struct AvailableEvidenceBatchInfo {
    std::size_t origin_count{};
    std::vector<std::string> dependencies;
    std::vector<std::shared_ptr<ValueOriginBridge>> origins;
};
namespace available_evidence_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline const inference::RelationalAvailabilityContext& authority(
        const std::shared_ptr<AvailabilityContextBridge>& context) {
    if (!context) throw std::invalid_argument("independent context owner required");
    return context->native_owner();
}
inline std::vector<inference::RelationalAvailableInputSubmission> submissions(
        const std::vector<AdmissionInput>& rows, const inference::RelationalAvailabilityContext& context,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
    evidence_bridge_detail::bound(rows.size(), bounds.at(0));
    std::vector<inference::RelationalAvailableInputSubmission> out;
    for (const auto& row : rows) {
        const auto image = bytes(row.asset); const auto keys = admission_bridge_detail::input_keys(row.origins);
        (void)context.require(keys.asset, inference::RelationalDependencyRole::InputAsset, {image.begin(), image.end()});
        auto asset = InputUseAssetBridge::restore(row.asset, bounds, maximum);
        numeric::ExactActionGuard guard(limits(bounds));
        auto declaration = interface_bridge_detail::accession_declaration(row.accession, guard);
        auto field = interface_bridge_detail::field(row.field, bounds, maximum);
        out.push_back({{asset->native_owner(), std::move(declaration), row.use, std::move(field)}, keys});
    }
    return out;
}
inline numeric::RationalVector free_values(const std::vector<std::string>& values,
        const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds)); guard.dimension(values.size());
    numeric::RationalVector out;
    for (const auto& value : values) {
        numeric::Rational q;
        if (q.set_str(value, 16) != 0 || q.get_den() == 0) throw std::invalid_argument("exact complete free values required");
        q.canonicalize(); guard.value(q); out.push_back(std::move(q));
    }
    return out;
}
} // namespace available_evidence_bridge_detail

class AvailableAffineBridge;
class AvailableEvaluationBridge;
class AvailableEvidenceBridge {
public:
    const rfc::learning::PreparedAvailableRelationalEvidence& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedAvailableRelationalEvidence& native_owner() const && = delete;
    AvailableEvidenceBridge(std::shared_ptr<AvailableProgramBridge> program, const EvidenceDeclaration& declaration,
            std::uint64_t origin, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace available_evidence_bridge_detail;
        if (!program) throw std::invalid_argument("actual available program owner required");
        const auto& actual = authority(context); numeric::ExactActionGuard guard(limits(bounds));
        auto value = evidence_bridge_detail::declaration(declaration, guard);
        owner_ = std::make_shared<const learning::PreparedAvailableRelationalEvidence>(
            program->native_owner(), std::move(value), origin, actual, limits(bounds));
    }
    static std::shared_ptr<AvailableEvidenceBridge> adopt(rfc::learning::PreparedAvailableRelationalEvidence owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableEvidenceBridge>(new AvailableEvidenceBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AvailableEvidenceBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        return adopt(learning::decode_available_relational_evidence(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    static std::string root_payload(std::shared_ptr<FieldProgramBridge> program, const EvidenceDeclaration& declaration,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        if (!program) throw std::invalid_argument("actual original program owner required");
        numeric::ExactActionGuard guard(limits(bounds));
        return bounded(learning::relational_evidence_root_payload(inference::encode_relational_field_program(program->native_owner()),
            evidence_bridge_detail::declaration(declaration, guard), limits(bounds)), maximum);
    }
    static std::string target_payload(std::shared_ptr<FieldProgramBridge> program, std::uint64_t node,
            const std::string& asset, std::uint64_t use, const std::string& target,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        if (!program) throw std::invalid_argument("actual original program owner required");
        return bounded(learning::relational_target_value_payload(inference::encode_relational_field_program(program->native_owner()),
            node, bytes(asset), use, interface_bridge_detail::field(target, bounds, maximum)), maximum);
    }
    std::string setting_payload(const std::vector<std::string>& values) const {
        using namespace available_evidence_bridge_detail;
        return bounded(learning::relational_parameter_setting_payload(*owner_, free_values(values, bounds_)), maximum_);
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::learning::encode_available_relational_evidence(*owner_), maximum_);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        owner_->admit(available_evidence_bridge_detail::authority(context));
    }
    std::shared_ptr<FieldEvidenceBridge> evidence() const { return FieldEvidenceBridge::adopt(owner_->evidence(), bounds_, maximum_); }
    std::shared_ptr<AvailableProgramBridge> program() const { return AvailableProgramBridge::adopt(owner_->program(), bounds_, maximum_); }
    std::shared_ptr<AvailableEvidenceBridge> parent() const {
        const auto value = owner_->parent(); return value ? adopt(*value, bounds_, maximum_) : nullptr;
    }
    std::size_t origin_count() const { return owner_->obligations().size(); }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_->obligations().size(), maximum);
        auto out = admission_bridge_detail::dependencies(owner_->obligations(), maximum_);
        const auto assets = evidence()->dependency_archives(maximum);
        evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), assets.size()), maximum);
        out.insert(out.end(), assets.begin(), assets.end()); return out;
    }
    std::shared_ptr<AvailableEvidenceBridge> append(const std::vector<AvailableEvidenceObservation>& rows,
            std::shared_ptr<AvailabilityContextBridge> context) const {
        using namespace available_evidence_bridge_detail;
        const auto pin = owner_; const auto frozen = rows; const auto authority_pin = context;
        const auto& actual = authority(authority_pin); pin->admit(actual);
        evidence_bridge_detail::bound(frozen.size(), bounds_.at(0));
        std::vector<learning::RelationalAvailableObservationSubmission> suffix;
        for (const auto& row : frozen) {
            if (row.origins.size() != (row.port ? 3U : 2U)) throw std::invalid_argument("complete typed target origin keys required");
            const learning::RelationalTargetOrigins keys{row.origins[0], row.origins[1],row.port?row.origins[2]:0}; const auto image = bytes(row.target_asset);
            (void)actual.require(keys.asset, inference::RelationalDependencyRole::TargetAsset, {image.begin(), image.end()});
            auto target_asset = InputUseAssetBridge::restore(row.target_asset, bounds_, maximum_);
            learning::RelationalAvailableObservationSubmission value{row.key, submissions(row.inputs, actual, bounds_, maximum_),
                target_asset->native_owner(), row.target_use, interface_bridge_detail::field(row.target, bounds_, maximum_), {}, keys};
            for (const auto& cell : row.observed) {
                if (cell.size() != 2) throw std::invalid_argument("whole observed stalk grade and index required");
                value.observed.push_back(program_bridge_detail::cell(cell[0], cell[1]));
            }
            if(row.port)value.port=row.port->native_owner();
            value.observed_coordinates=row.observed_coordinates;
            suffix.push_back(std::move(value));
        }
        return adopt(pin->append_inputs(std::move(suffix), actual), bounds_, maximum_);
    }
    AvailableEvidenceBatchInfo batch_info(const std::vector<AvailableEvidenceObservation>& rows,
            std::shared_ptr<AvailabilityContextBridge> context, std::size_t maximum) const {
        using namespace available_evidence_bridge_detail;
        const auto pin = owner_; const auto frozen = rows; const auto authority_pin = context;
        const auto& actual = authority(authority_pin); pin->admit(actual);
        const auto program = pin->program(); const auto program_bytes = inference::encode_relational_field_program(program.program());
        std::vector<inference::OwnedRelationalValueOrigin> origins;
        std::vector<std::string> assets;
        evidence_bridge_detail::bound(frozen.size(), bounds_.at(0));
        for (const auto& row : frozen) {
            const auto inputs = submissions(row.inputs, actual, bounds_, maximum_); program.check_inputs(inputs, actual);
            origins.insert(origins.end(), program.obligations().begin(), program.obligations().end());
            for (const auto& input : inputs) {
                const auto asset = inference::encode_relational_input_use_asset(input.input.asset);
                const auto value = inference::relational_input_value_payload(asset, input.input.declaration, input.input.use, input.input.source_field);
                const auto accession = inference::relational_input_accession_payload(program_bytes, asset, input.input.declaration, input.input.use, input.input.source_field);
                inference::value_admission_detail::input_origins(origins, input.origins, asset, value, accession, actual);
                assets.push_back(bounded(storage::encode_native_series(input.input.asset.history()), maximum_));
            }
            if (row.origins.size() != (row.port ? 3U : 2U)) throw std::invalid_argument("complete typed target origin keys required");
            const auto image = bytes(row.target_asset);
            const auto asset = actual.require(row.origins[0], inference::RelationalDependencyRole::TargetAsset, {image.begin(), image.end()});
            const auto field = interface_bridge_detail::field(row.target, bounds_, maximum_);
            if(row.port) {
                const auto target_program=row.port->native_owner().program();const auto evidence_program=program.program();
                if(!target_program.fibers().same_owner(evidence_program.fibers()) ||
                   row.port->native_owner().declaration().node != pin->root_declaration().node)
                    throw std::invalid_argument("typed target requires actual evidence program owner/node");
            }
            const auto payload = row.port?learning::relational_occurrence_target_value_payload(program_bytes,row.port->native_owner().declaration(),image,row.target_use,field,row.observed_coordinates,
                learning::available_evidence_detail::target_inputs(program_bytes,inputs)):
                learning::relational_target_value_payload(program_bytes, pin->root_declaration().node, image, row.target_use, field);
            origins.push_back(asset); origins.push_back(actual.require(row.origins[1], inference::RelationalDependencyRole::TargetValue, payload));
            if(row.port)origins.push_back(actual.require(row.origins[2],inference::RelationalDependencyRole::Interface,
                learning::encode_relational_occurrence_target_port(row.port->native_owner())));
            const auto target = InputUseAssetBridge::restore(row.target_asset, bounds_, maximum_);
            assets.push_back(bounded(storage::encode_native_series(target->native_owner().history()), maximum_));
            evidence_bridge_detail::bound(machine::checked_size_add(origins.size(), assets.size()), maximum);
        }
        auto dependencies = admission_bridge_detail::dependencies(origins, maximum_);
        dependencies.insert(dependencies.end(), assets.begin(), assets.end());
        AvailableEvidenceBatchInfo out;out.origin_count=origins.size();out.dependencies=std::move(dependencies);
        for(const auto& origin:origins)out.origins.push_back(ValueOriginBridge::adopt(origin,maximum_));
        return out;
    }
    std::vector<std::shared_ptr<ValueOriginBridge>> observation_origins(std::size_t index,
            std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum) const {
        const auto origins=owner_->observation_dependencies(index,available_evidence_bridge_detail::authority(context),maximum);
        std::vector<std::shared_ptr<ValueOriginBridge>> out;
        for(const auto& origin:origins)out.push_back(ValueOriginBridge::adopt(origin,maximum_));
        return out;
    }
    static std::string occurrence_target_payload(std::shared_ptr<OccurrenceTargetBridge> port,
            const std::string& asset,std::uint64_t use,const std::string& target,const std::vector<AdmissionInput>& inputs,
            const std::vector<std::uint64_t>& mask,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        if(!port)throw std::invalid_argument("actual native target port required");
        evidence_bridge_detail::bound(inputs.size(),bounds.at(0));evidence_bridge_detail::bound(mask.size(),bounds.at(0));
        const auto program_bytes=inference::encode_relational_field_program(port->native_owner().program());
        std::vector<std::vector<std::byte>> accessions;
        numeric::ExactActionGuard guard(limits(bounds));
        for(const auto& input:inputs)accessions.push_back(inference::relational_input_accession_payload(program_bytes,bytes(input.asset),
            interface_bridge_detail::accession_declaration(input.accession,guard),input.use,
            interface_bridge_detail::field(input.field,bounds,maximum)));
        return bounded(learning::relational_occurrence_target_value_payload(program_bytes,port->native_owner().declaration(),bytes(asset),use,
            interface_bridge_detail::field(target,bounds,maximum),mask,accessions),maximum);
    }
    std::shared_ptr<AvailableAffineBridge> execute(const std::vector<AdmissionInput>&,
        std::shared_ptr<AvailabilityContextBridge>) const;
private:
    AvailableEvidenceBridge(rfc::learning::PreparedAvailableRelationalEvidence owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::PreparedAvailableRelationalEvidence>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedAvailableRelationalEvidence> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableAffineBridge {
public:
    const rfc::learning::AvailableRelationalAffineResult& native_owner() const & { return *owner_; }
    const rfc::learning::AvailableRelationalAffineResult& native_owner() const && = delete;
    static std::shared_ptr<AvailableAffineBridge> adopt(rfc::learning::AvailableRelationalAffineResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableAffineBridge>(new AvailableAffineBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AvailableAffineBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        return adopt(learning::decode_available_relational_affine_result(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_affine_result(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_evidence_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableEvidenceBridge> evidence() const { return AvailableEvidenceBridge::adopt(owner_->evidence(), bounds_, maximum_); }
    std::shared_ptr<AffineResultBridge> result() const { return AffineResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::shared_ptr<AvailableResultBridge> input() const { return AvailableResultBridge::adopt(owner_->input(), bounds_, maximum_); }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out = evidence()->dependency_archives(maximum);
        const auto origins = input()->dependency_archives(maximum);
        evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), origins.size()), maximum);
        out.insert(out.end(), origins.begin(), origins.end());
        const auto actual = owner_->input().result();
        for (const auto& use : actual.inputs()) {
            evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), 1), maximum);
            out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(use.accession().asset().history()), maximum_));
        }
        return out;
    }
    std::shared_ptr<AvailableEvaluationBridge> evaluate(const std::vector<std::string>&, std::uint64_t,
        std::shared_ptr<AvailabilityContextBridge>) const;
private:
    AvailableAffineBridge(rfc::learning::AvailableRelationalAffineResult owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::AvailableRelationalAffineResult>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalAffineResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableEvaluationBridge {
public:
    const rfc::learning::AvailableRelationalAffineEvaluation& native_owner() const & { return *owner_; }
    const rfc::learning::AvailableRelationalAffineEvaluation& native_owner() const && = delete;
    static std::shared_ptr<AvailableEvaluationBridge> adopt(rfc::learning::AvailableRelationalAffineEvaluation owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableEvaluationBridge>(new AvailableEvaluationBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AvailableEvaluationBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace available_evidence_bridge_detail;
        return adopt(learning::decode_available_relational_affine_evaluation(bytes(image), authority(context), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_available_relational_affine_evaluation(*owner_), maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_evidence_bridge_detail::authority(context)); }
    std::shared_ptr<AvailableAffineBridge> family() const { return AvailableAffineBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<AffineEvaluationBridge> result() const { return AffineEvaluationBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out = family()->dependency_archives(maximum); evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(), 1), maximum);
        const auto setting = owner_->setting(); out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(setting.publication().history()), maximum_));
        return out;
    }
private:
    AvailableEvaluationBridge(rfc::learning::AvailableRelationalAffineEvaluation owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::AvailableRelationalAffineEvaluation>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalAffineEvaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AvailableAffineBridge> AvailableEvidenceBridge::execute(const std::vector<AdmissionInput>& rows,
        std::shared_ptr<AvailabilityContextBridge> context) const {
    using namespace available_evidence_bridge_detail;
    const auto pin = owner_; const auto frozen = rows; const auto authority_pin = context; const auto& actual = authority(authority_pin);
    pin->admit(actual);
    return AvailableAffineBridge::adopt(pin->execute_inputs(submissions(frozen, actual, bounds_, maximum_), actual, limits(bounds_)), bounds_, maximum_);
}
inline std::shared_ptr<AvailableEvaluationBridge> AvailableAffineBridge::evaluate(const std::vector<std::string>& values,
        std::uint64_t setting, std::shared_ptr<AvailabilityContextBridge> context) const {
    using namespace available_evidence_bridge_detail;
    const auto pin = owner_; const auto frozen = values; const auto authority_pin = context;
    return AvailableEvaluationBridge::adopt(pin->evaluate(free_values(frozen, bounds_), setting, authority(authority_pin), limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
