#pragma once

#include "field_interface_bridge.hpp"
#include "occurrence_target_bridge.hpp"
#include <rfc/learning/relational_affine_program_codec.hpp>

namespace rexgraph::python {
struct EvidenceDeclaration {
    std::uint64_t key{}, revision{}, node{};
    std::vector<std::vector<std::size_t>> variables;
    ProgramMatrix basis;
    std::vector<std::uint64_t> parameters;
};
struct EvidenceIdentity {
    std::uint64_t key{}, revision{}, node{};
    std::size_t grade{}, parameters{}, free_parameters{}, observations{};
    bool has_parent{};
};
struct EvidenceDetails {
    std::vector<std::vector<std::size_t>> variables, chain_coordinates, observation_coordinates;
    std::vector<std::uint64_t> parameters, free_parameters;
    std::string basis, particular, kernel, chain, design, augmented, target, parent_shift, parent_map;
    bool has_parent{};
    std::vector<std::vector<std::size_t>> typed_coordinates;
};
struct EvidenceObservation {
    std::uint64_t key{}, target_use{};
    std::shared_ptr<AccessionResultBridge> input;
    std::shared_ptr<InputUseAssetBridge> target_asset;
    std::string target;
    std::vector<std::vector<std::size_t>> observed;
    std::shared_ptr<OccurrenceTargetBridge> port;
    std::vector<std::uint64_t> observed_coordinates;
};
struct EvidenceObservationInfo {
    std::uint64_t key{}, target_use{};
    std::string input, target_asset, target;
    std::vector<std::vector<std::size_t>> observed;
    std::shared_ptr<OccurrenceTargetBridge> port;
    std::vector<std::uint64_t> observed_coordinates;
};
struct AffineFieldInfo {
    std::string particular, directions;
    std::size_t dimension{}, supported{}, nonzeros{}, free_parameters{}, direction_nonzeros{};
};
struct AffineSelected {
    ProgramSelected particular;
    std::string directions;
    std::vector<bool> identified;
};
struct AffineMapInfo {
    std::size_t grade{}, occurrence{};
    std::string particular;
    std::vector<std::string> directions;
};
namespace evidence_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline void bound(std::size_t count, std::size_t maximum) {
    if (count > maximum) throw std::length_error("explicit evidence metadata bound exceeded");
}
inline std::string vector_bytes(const numeric::RationalVector& values) {
    storage::temporal_codec_detail::Writer writer;
    learning::relational_affine_codec_detail::vector(writer, values);
    return text(writer.bytes());
}
inline AffineFieldInfo field_info(const learning::RelationalAffineField& field) {
    return {tensor_bytes(field.particular), matrix_bytes(field.directions), field.directions.rows(),
        field.particular.support().count(), field.particular.values().nonzeros(),
        field.directions.cols(), field.directions.nonzeros()};
}
inline learning::RelationalFieldParameterDeclaration declaration(const EvidenceDeclaration& value,
        numeric::ExactActionGuard& guard) {
    learning::RelationalFieldParameterDeclaration out;
    out.key = value.key; out.revision = value.revision; out.node = value.node;
    for (const auto& v : value.variables) {
        if (v.size() != 3) throw std::invalid_argument("original parameter flag/row/column required");
        out.variables.push_back({v[0], v[1], v[2]});
    }
    out.basis = matrix(value.basis, guard); out.parameters = value.parameters;
    return out;
}
} // namespace evidence_bridge_detail

class AffineResultBridge;
class EvidenceResidualBridge;
class EvidenceMapBridge;
class FieldEvidenceBridge {
public:
    const rfc::learning::PreparedRelationalFieldEvidence& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedRelationalFieldEvidence& native_owner() const && = delete;
    FieldEvidenceBridge(std::shared_ptr<FieldProgramBridge> program, const EvidenceDeclaration& value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace evidence_bridge_detail;
        if (!program) throw std::invalid_argument("actual field evidence program owner required");
        numeric::ExactActionGuard guard(limits(bounds));
        auto d = declaration(value, guard);
        owner_ = std::make_shared<const learning::PreparedRelationalFieldEvidence>(program->native_owner(), std::move(d), limits(bounds));
    }
    static std::shared_ptr<FieldEvidenceBridge> restore(const std::string& image,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace evidence_bridge_detail;
        return adopt(learning::decode_relational_field_evidence(bytes(image), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    static std::shared_ptr<FieldEvidenceBridge> adopt(rfc::learning::PreparedRelationalFieldEvidence owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<FieldEvidenceBridge>(new FieldEvidenceBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::learning::encode_relational_field_evidence(*owner_), maximum_);
    }
    EvidenceIdentity identity() const {
        const auto& d = owner_->declaration();
        const auto program = owner_->program();
        return {d.key, d.revision, d.node, rfc::learning::relational_evidence_detail::node(program, d.node).output_grade.value,
            owner_->parameters().size(), owner_->free_parameters().size(), owner_->observations().size(), owner_->parent().has_value()};
    }
    bool has_occurrence_observations() const { return owner_->has_occurrence_observations(); }
    std::vector<std::size_t> update() const {
        const auto u=owner_->update();return {u.chain_reused?1U:0U,u.reused_observations,u.assembled_observations,u.execution_steps};
    }
    EvidenceDetails details(std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        const auto& d = owner_->declaration(); const auto& system = owner_->observation_system();
        bound(d.variables.size(), maximum); bound(d.parameters.size(), maximum);
        bound(owner_->chain_coordinates().size(), maximum); bound(system.constraints.size(), maximum);
        EvidenceDetails out; out.parameters = d.parameters; out.free_parameters = owner_->free_parameters();
        for (const auto& v : d.variables) out.variables.push_back({v.occurrence, v.row, v.column});
        for (const auto& c : owner_->chain_coordinates()) out.chain_coordinates.push_back({c.grade, c.row, c.column});
        for (const auto& c : system.constraints) {
            out.observation_coordinates.push_back({c.sample, c.cell, c.axis});
            out.typed_coordinates.push_back({c.sample,static_cast<std::size_t>(c.domain),c.cell,c.axis});
        }
        out.basis = matrix_bytes(d.basis); out.particular = vector_bytes(owner_->particular()); out.kernel = matrix_bytes(owner_->kernel());
        out.chain = matrix_bytes(owner_->chain_constraints()); out.design = matrix_bytes(system.design);
        out.augmented = matrix_bytes(system.augmented); out.target = vector_bytes(system.target);
        out.has_parent = owner_->parent_relation().has_value();
        if (out.has_parent) { out.parent_shift = vector_bytes(owner_->parent_relation()->shift); out.parent_map = matrix_bytes(owner_->parent_relation()->map); }
        return out;
    }
    EvidenceObservationInfo observation(std::size_t index, std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        if (index >= owner_->observations().size()) throw std::out_of_range("original evidence observation index");
        const auto& o = owner_->observations()[index]; bound(o.observed.size(), maximum);bound(o.observed_coordinates.size(),maximum);
        EvidenceObservationInfo out; out.key = o.key; out.target_use = o.target_use;
        out.input = bounded(inference::encode_accession_field_result(o.input), maximum_);
        out.target_asset = bounded(inference::encode_relational_input_use_asset(o.target_asset), maximum_);
        out.target = tensor_bytes(o.target);
        for (const auto c : o.observed) out.observed.push_back({c.grade.value, c.index});
        if(o.port) out.port=OccurrenceTargetBridge::adopt(*o.port,bounds_,maximum_);
        out.observed_coordinates=o.observed_coordinates;
        return out;
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        std::size_t count{};
        for (const auto& o : owner_->observations()) {
            if (count > maximum || o.input.inputs().size() >= maximum-count)
                throw std::length_error("evidence source dependency bound");
            count += o.input.inputs().size()+1;
        }
        std::vector<std::string> out;
        for (const auto& o : owner_->observations()) {
            for (const auto& input : o.input.inputs()) out.push_back(bounded(storage::encode_native_series(input.accession().asset().history()), maximum_));
            out.push_back(bounded(storage::encode_native_series(o.target_asset.history()), maximum_));
        }
        return out;
    }
    std::shared_ptr<FieldProgramBridge> program() const { return FieldProgramBridge::adopt(owner_->program(), bounds_, maximum_); }
    std::shared_ptr<FieldEvidenceBridge> parent() const {
        const auto parent = owner_->parent(); return parent ? adopt(*parent, bounds_, maximum_) : nullptr;
    }
    std::shared_ptr<FieldEvidenceBridge> append(const std::vector<EvidenceObservation>& suffix) const {
        using namespace evidence_bridge_detail;
        std::vector<learning::RelationalFieldTrainingObservation> observations;
        bound(suffix.size(), bounds_[0]);
        for (const auto& o : suffix) {
            if (!o.input || !o.target_asset) throw std::invalid_argument("actual input and measured target asset owners required");
            learning::RelationalFieldTrainingObservation value{o.key, o.input->native_owner(), o.target_asset->native_owner(),
                o.target_use, interface_bridge_detail::field(o.target, bounds_, maximum_), {}};
            for (const auto& c : o.observed) {
                if (c.size() != 2) throw std::invalid_argument("whole observed stalk grade and index required");
                value.observed.push_back(cell(c[0], c[1]));
            }
            if(o.port)value.port=o.port->native_owner();
            value.observed_coordinates=o.observed_coordinates;
            observations.push_back(std::move(value));
        }
        return adopt(owner_->append(std::move(observations), limits(bounds_)), bounds_, maximum_);
    }
    std::shared_ptr<AffineResultBridge> execute(std::shared_ptr<AccessionResultBridge> input) const;
    std::shared_ptr<EvidenceResidualBridge> residual(std::size_t observation) const;
    std::shared_ptr<EvidenceMapBridge> occurrence_map(std::size_t grade, std::size_t occurrence) const;
private:
    friend class AffineResultBridge;
    FieldEvidenceBridge(rfc::learning::PreparedRelationalFieldEvidence owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::PreparedRelationalFieldEvidence>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedRelationalFieldEvidence> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AffineEvaluationBridge;
class AffineResultBridge {
public:
    static std::shared_ptr<AffineResultBridge> adopt(rfc::learning::RelationalAffineProgramResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AffineResultBridge>(new AffineResultBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AffineResultBridge> restore(const std::string& image, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace evidence_bridge_detail;
        return adopt(learning::decode_relational_affine_result(bytes(image), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_relational_affine_result(*owner_), maximum_); }
    std::shared_ptr<FieldEvidenceBridge> evidence() const { return FieldEvidenceBridge::adopt(owner_->evidence(), bounds_, maximum_); }
    std::shared_ptr<AccessionResultBridge> input() const { return AccessionResultBridge::adopt(owner_->input(), bounds_, maximum_); }
    std::vector<ProgramFieldInfo> fields(std::size_t maximum) const {
        auto metadata = input()->result()->fields(maximum);
        for (auto& item : metadata) item.nonzeros = owner_->field(item.node).particular.values().nonzeros();
        return metadata;
    }
    AffineFieldInfo field(std::uint64_t node, bool occurrences) const {
        return evidence_bridge_detail::field_info(occurrences ? owner_->occurrence_field(node) : owner_->field(node));
    }
    bool identified(std::uint64_t node, std::size_t coordinate) const { return owner_->identified(node, coordinate); }
    std::vector<AffineSelected> select(std::uint64_t node, const std::vector<std::vector<std::size_t>>& cells, std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        const auto program = owner_->evidence().program(); const auto& field = owner_->field(node);
        const auto selected_grade = learning::relational_evidence_detail::node(program, node).output_grade;
        const auto offsets = program.fibers().fibers().stalks().offsets(selected_grade);
        numeric::ExactActionGuard guard(limits(bounds_));
        std::size_t used{}; std::vector<AffineSelected> out; std::set<std::size_t> selected_cells;
        for (const auto& c : cells) {
            if (c.size() != 2 || c[0] != selected_grade.value || offsets.empty() || c[1] >= offsets.size()-1) throw std::out_of_range("original affine stalk selection");
            if (!selected_cells.insert(c[1]).second) throw std::invalid_argument("distinct original affine stalk selection required");
            const auto original = cell(c[0], c[1]); const auto count = offsets[c[1]+1]-offsets[c[1]];
            if (used > maximum || count > maximum-used) throw std::length_error("affine stalk coordinate bound");
            used += count; AffineSelected value;
            const auto& stalks = program.declaration().stalks;
            const auto found = std::find_if(stalks.begin(), stalks.end(), [&](const auto& s) { return s.cell == original; });
            if (found == stalks.end()) throw std::out_of_range("actual original affine stalk basis");
            value.particular.stalk = stalk_info(*found); numeric::RationalVector selected;
            numeric::ExactMatrix directions(count, field.directions.cols());
            for (std::size_t i = 0; i < count; ++i) {
                guard.step(); const auto row = offsets[c[1]]+i; const auto q = field.particular.state(row).value();
                selected.push_back(q); value.particular.values.push_back(q.get_str(16)); value.identified.push_back(owner_->identified(node, row));
            }
            for (const auto& [ij,q] : field.directions.entries()) {
                guard.step(); if (ij.first >= offsets[c[1]] && ij.first < offsets[c[1]+1]) directions.set(ij.first-offsets[c[1]], ij.second, q);
            }
            value.particular.field = tensor_bytes(inference::field_interface_detail::tensor(selected, guard));
            value.directions = matrix_bytes(directions); out.push_back(std::move(value));
        }
        return out;
    }
    std::shared_ptr<AffineEvaluationBridge> evaluate(const std::vector<std::string>& free) const;
private:
    friend class AffineEvaluationBridge;
    AffineResultBridge(rfc::learning::RelationalAffineProgramResult owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalAffineProgramResult>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalAffineProgramResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AffineResultBridge> FieldEvidenceBridge::execute(std::shared_ptr<AccessionResultBridge> input) const {
    if (!input) throw std::invalid_argument("actual independent held input owner required");
    return AffineResultBridge::adopt(owner_->execute(input->native_owner(), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
class AffineEvaluationBridge {
public:
    static std::shared_ptr<AffineEvaluationBridge> adopt(rfc::learning::RelationalAffineEvaluation owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AffineEvaluationBridge>(new AffineEvaluationBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AffineEvaluationBridge> restore(const std::string& image, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace evidence_bridge_detail;
        auto owner = learning::decode_relational_affine_evaluation(bytes(image), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<AffineEvaluationBridge>(new AffineEvaluationBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_relational_affine_evaluation(*owner_), maximum_); }
    std::shared_ptr<AffineResultBridge> family() const { return AffineResultBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<AccessionResultBridge> result() const { return AccessionResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::string free_values() const { return evidence_bridge_detail::vector_bytes(owner_->free_values()); }
private:
    friend class AffineResultBridge;
    AffineEvaluationBridge(rfc::learning::RelationalAffineEvaluation owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalAffineEvaluation>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalAffineEvaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AffineEvaluationBridge> AffineResultBridge::evaluate(const std::vector<std::string>& values) const {
    using namespace evidence_bridge_detail;
    numeric::ExactActionGuard guard(limits(bounds_)); guard.dimension(values.size()); numeric::RationalVector free;
    for (const auto& value : values) {
        numeric::Rational q; if (q.set_str(value,16) != 0 || q.get_den() == 0) throw std::invalid_argument("exact complete free values required");
        q.canonicalize(); guard.value(q); free.push_back(std::move(q));
    }
    return std::shared_ptr<AffineEvaluationBridge>(new AffineEvaluationBridge(owner_->evaluate(std::move(free), limits(bounds_)), bounds_, maximum_));
}
class EvidenceResidualBridge {
public:
    std::shared_ptr<OccurrenceTargetBridge> target_port() const {
        const auto p=owner_.target_port();return p?OccurrenceTargetBridge::adopt(*p,bounds_,maximum_):nullptr;
    }
    AffineFieldInfo field() const { return evidence_bridge_detail::field_info(owner_.field()); }
    std::shared_ptr<AffineResultBridge> response() const { return AffineResultBridge::adopt(owner_.response(), bounds_, maximum_); }
    std::size_t observation() const { return owner_.observation(); }
private:
    friend class FieldEvidenceBridge;
    EvidenceResidualBridge(rfc::learning::RelationalFieldResidual owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::move(owner)), bounds_(std::move(bounds)), maximum_(maximum) {}
    rfc::learning::RelationalFieldResidual owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<EvidenceResidualBridge> FieldEvidenceBridge::residual(std::size_t observation) const {
    return std::shared_ptr<EvidenceResidualBridge>(new EvidenceResidualBridge(owner_->residual(observation, program_bridge_detail::limits(bounds_)), bounds_, maximum_));
}
class EvidenceMapBridge {
public:
    AffineMapInfo info(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_.directions().size(), maximum);
        AffineMapInfo out; out.grade = owner_.grade().value; out.occurrence = owner_.occurrence(); out.particular = program_bridge_detail::matrix_bytes(owner_.particular());
        for (const auto& m : owner_.directions()) out.directions.push_back(program_bridge_detail::matrix_bytes(m));
        return out;
    }
    std::shared_ptr<FieldEvidenceBridge> evidence() const { return FieldEvidenceBridge::adopt(owner_.evidence(), bounds_, maximum_); }
private:
    friend class FieldEvidenceBridge;
    EvidenceMapBridge(rfc::learning::RelationalAffineOccurrenceMap owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::move(owner)), bounds_(std::move(bounds)), maximum_(maximum) {}
    rfc::learning::RelationalAffineOccurrenceMap owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<EvidenceMapBridge> FieldEvidenceBridge::occurrence_map(std::size_t grade, std::size_t occurrence) const {
    auto map = owner_->coefficient_independent_map(program_bridge_detail::cell(grade,0).grade, occurrence, program_bridge_detail::limits(bounds_));
    return std::shared_ptr<EvidenceMapBridge>(new EvidenceMapBridge(std::move(map), bounds_, maximum_));
}
} // namespace rexgraph::python
