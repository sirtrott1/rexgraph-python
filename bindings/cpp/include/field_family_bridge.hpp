#pragma once

#include "field_evidence_bridge.hpp"
#include <rfc/learning/relational_field_family_codec.hpp>

namespace rexgraph::python {
struct FamilyView {
    std::uint64_t frame{};
    ProgramMatrix accession, reconstruction;
};
struct FamilyInterface {
    std::uint64_t key{}, revision{}, cutoff{};
    std::size_t left{}, right{};
    std::string common;
    std::vector<std::vector<std::size_t>> left_cells, right_cells, left_occurrences, right_occurrences;
    std::vector<std::vector<FamilyView>> left_views, right_views;
};
struct FamilyDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<FamilyInterface> interfaces;
};
struct FamilyIdentity {
    std::uint64_t key{}, revision{};
    std::size_t members{}, interfaces{}, free_parameters{}, rank{}, retained_entries{};
    bool consistent{};
    std::vector<std::size_t> limits;
};
struct FamilyProof {
    std::string particular, kernel, coordinate_origin, coordinate_kernel, residual, equations, contradiction;
    std::vector<std::size_t> parameter_offsets, member_free_offsets, free_columns;
    std::vector<std::vector<std::size_t>> equation_coordinates;
    std::vector<unsigned> statuses;
};
struct FamilyWitness {
    std::size_t interface{}, grade{}, occurrence{}, output_dimension{};
    std::string common_input, left_input, right_input;
    std::vector<std::vector<std::size_t>> response_shapes, equation_coordinates;
};
struct FamilyCoordinates { std::string shift, map; };

namespace family_bridge_detail {
using namespace rfc;
using namespace evidence_bridge_detail;
inline algebra::RetainedSourceMap source_map(const std::vector<std::vector<std::size_t>>& cells,
        const std::vector<std::vector<std::size_t>>& occurrences, numeric::ExactActionGuard& guard) {
    algebra::RetainedSourceMap out;
    for (const auto& row : cells) {
        guard.dimension(row.size()); std::vector<std::uint32_t> items;
        for (const auto index : row) { guard.step(); items.push_back(cell(0, index).index); }
        out.cells.push_back(std::move(items));
    }
    out.occurrences = occurrences;
    return out;
}
inline std::vector<std::vector<learning::TransportContextView>> views(
        const std::vector<std::vector<FamilyView>>& all, numeric::ExactActionGuard& guard) {
    guard.dimension(all.size()); std::vector<std::vector<learning::TransportContextView>> out;
    for (const auto& row : all) {
        guard.dimension(row.size()); std::vector<learning::TransportContextView> items;
        for (const auto& v : row) items.push_back({v.frame, matrix(v.accession, guard), matrix(v.reconstruction, guard)});
        out.push_back(std::move(items));
    }
    return out;
}
inline learning::RelationalFieldFamilyDeclaration declaration(const FamilyDeclaration& d,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
    numeric::ExactActionGuard guard(limits(bounds)); guard.dimension(d.interfaces.size());
    learning::RelationalFieldFamilyDeclaration out{d.key, d.revision, {}};
    storage::NativeSeriesLimits policy; policy.max_archive_bytes = maximum; policy.temporal = reader(bounds, maximum);
    for (const auto& i : d.interfaces) {
        const auto history = storage::decode_native_series(bytes(i.common), policy);
        out.interfaces.push_back({i.key, i.revision, i.left, i.right, history.at(i.cutoff),
            source_map(i.left_cells, i.left_occurrences, guard), source_map(i.right_cells, i.right_occurrences, guard),
            views(i.left_views, guard), views(i.right_views, guard)});
    }
    return out;
}
} // namespace family_bridge_detail

class JointResultBridge;
class JointEvaluationBridge;
class JointMapBridge;
class FieldFamilyBridge {
public:
    const rfc::learning::PreparedRelationalFieldFamily& native_owner() const & { return *owner_; }
    const rfc::learning::PreparedRelationalFieldFamily& native_owner() const && = delete;
    FieldFamilyBridge(const FamilyDeclaration& d, const std::vector<std::shared_ptr<FieldEvidenceBridge>>& members,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace family_bridge_detail;
        bound(members.size(), bounds.at(0)); std::vector<learning::PreparedRelationalFieldEvidence> actual;
        for (const auto& member : members) {
            if (!member) throw std::invalid_argument("actual field evidence member owner required");
            actual.push_back(member->native_owner());
        }
        owner_ = std::make_shared<const learning::PreparedRelationalFieldFamily>(declaration(d, bounds, maximum), std::move(actual), limits(bounds));
    }
    static std::shared_ptr<FieldFamilyBridge> adopt(rfc::learning::PreparedRelationalFieldFamily owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<FieldFamilyBridge>(new FieldFamilyBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<FieldFamilyBridge> restore(const std::string& image,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace family_bridge_detail;
        return adopt(learning::decode_relational_field_family(bytes(image), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const {
        return program_bridge_detail::text(rfc::learning::encode_relational_field_family(*owner_, maximum_));
    }
    FamilyIdentity identity() const {
        const auto& d = owner_->declaration(); const auto& g = owner_->gluing(); const auto l = owner_->limits();
        return {d.key, d.revision, owner_->members().size(), d.interfaces.size(), g.coordinate_kernel().cols(),
            g.rank(), owner_->retained_entries(), g.consistent(), {l.dimension, l.entries, l.steps, l.integer_bits}};
    }
    FamilyProof proof(std::size_t maximum) const {
        using namespace family_bridge_detail;
        const auto& g = owner_->gluing(); bound(g.equation_coordinates().size(), maximum);
        bound(g.statuses().size(), maximum); bound(g.parameter_offsets().size(), maximum);
        bound(g.member_free_offsets().size(), maximum); bound(g.free_columns().size(), maximum);
        FamilyProof out{vector_bytes(g.particular()), matrix_bytes(g.kernel()), vector_bytes(g.coordinate_origin()),
            matrix_bytes(g.coordinate_kernel()), vector_bytes(g.residual_origin()), matrix_bytes(g.equations()),
            vector_bytes(g.contradiction()), g.parameter_offsets(), g.member_free_offsets(), g.free_columns(), {}, {}};
        for (const auto& c : g.equation_coordinates()) out.equation_coordinates.push_back({c.interface, c.grade, c.occurrence, c.row, c.column});
        for (const auto s : g.statuses()) out.statuses.push_back(static_cast<unsigned>(s));
        return out;
    }
    FamilyWitness witness(std::size_t index, std::size_t maximum) const {
        using namespace family_bridge_detail;
        const auto& w = owner_->gluing().witnesses().at(index);
        bound(w.input.response_shapes.size(), maximum); bound(w.input.equation_coordinates.size(), maximum);
        FamilyWitness out{w.interface, w.grade, w.occurrence, w.output_dimension, matrix_bytes(w.input.common_input),
            matrix_bytes(w.input.origin_input), matrix_bytes(w.input.target_input), {}, {}};
        for (const auto [a,b] : w.input.response_shapes) out.response_shapes.push_back({a,b});
        for (const auto& c : w.input.equation_coordinates) out.equation_coordinates.push_back({c[0],c[1],c[2]});
        return out;
    }
    std::size_t witnesses() const { return owner_->gluing().witnesses().size(); }
    FamilyCoordinates coordinates(std::size_t member) const {
        const auto& c = owner_->member_coordinates(member);
        return {evidence_bridge_detail::vector_bytes(c.shift), program_bridge_detail::matrix_bytes(c.map)};
    }
    std::shared_ptr<FieldEvidenceBridge> member(std::size_t index) const {
        return FieldEvidenceBridge::adopt(owner_->members().at(index), bounds_, maximum_);
    }
    bool same_owner(std::shared_ptr<FieldFamilyBridge> other) const { return other && owner_->same_owner(*other->owner_); }
    std::shared_ptr<JointResultBridge> execute(std::size_t member, std::shared_ptr<AccessionResultBridge> input) const;
    std::shared_ptr<JointMapBridge> occurrence_map(std::size_t member, std::size_t grade, std::size_t occurrence) const;
private:
    FieldFamilyBridge(rfc::learning::PreparedRelationalFieldFamily owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::PreparedRelationalFieldFamily>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {
        const auto l = owner_->limits(); bounds_ = {l.dimension, l.entries, l.steps, l.integer_bits};
    }
    std::shared_ptr<const rfc::learning::PreparedRelationalFieldFamily> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};

class JointResultBridge {
public:
    static std::shared_ptr<JointResultBridge> adopt(rfc::learning::RelationalJointFieldResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<JointResultBridge>(new JointResultBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<JointResultBridge> restore(const std::string& image, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace family_bridge_detail;
        return adopt(learning::decode_relational_joint_result(bytes(image), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::text(rfc::learning::encode_relational_joint_result(*owner_, maximum_)); }
    std::shared_ptr<FieldFamilyBridge> family() const { return FieldFamilyBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<AffineResultBridge> original() const { return AffineResultBridge::adopt(owner_->original(), bounds_, maximum_); }
    std::shared_ptr<AccessionResultBridge> input() const { return AccessionResultBridge::adopt(owner_->input(), bounds_, maximum_); }
    std::size_t member() const { return owner_->member(); }
    std::vector<ProgramFieldInfo> fields(std::size_t maximum) const {
        auto out = input()->result()->fields(maximum);
        for (auto& item : out) item.nonzeros = owner_->field(item.node).particular.values().nonzeros();
        return out;
    }
    AffineFieldInfo field(std::uint64_t node, bool occurrences) const {
        return evidence_bridge_detail::field_info(occurrences ? owner_->occurrence_field(node) : owner_->field(node));
    }
    std::vector<AffineSelected> select(std::uint64_t node, const std::vector<std::vector<std::size_t>>& cells, std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        const auto family_pin = owner_->family();
        const auto program = family_pin.members().at(owner_->member()).program(); const auto& field = owner_->field(node);
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
    bool identified(std::uint64_t node, std::size_t coordinate) const { return owner_->identified(node, coordinate); }
    std::shared_ptr<JointEvaluationBridge> evaluate(const std::vector<std::string>& values) const;
private:
    JointResultBridge(rfc::learning::RelationalJointFieldResult owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalJointFieldResult>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {
        const auto l = owner_->family().limits(); bounds_ = {l.dimension, l.entries, l.steps, l.integer_bits};
    }
    std::shared_ptr<const rfc::learning::RelationalJointFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<JointResultBridge> FieldFamilyBridge::execute(std::size_t member, std::shared_ptr<AccessionResultBridge> input) const {
    if (!input) throw std::invalid_argument("actual independent accession input owner required");
    return JointResultBridge::adopt(owner_->execute(member, input->native_owner(), program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
class JointEvaluationBridge {
public:
    static std::shared_ptr<JointEvaluationBridge> adopt(rfc::learning::RelationalJointFieldEvaluation owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<JointEvaluationBridge>(new JointEvaluationBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<JointEvaluationBridge> restore(const std::string& image, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace family_bridge_detail;
        return adopt(learning::decode_relational_joint_evaluation(bytes(image), reader(bounds, maximum), limits(bounds)), bounds, maximum);
    }
    std::string archive() const { return program_bridge_detail::text(rfc::learning::encode_relational_joint_evaluation(*owner_, maximum_)); }
    std::shared_ptr<JointResultBridge> family() const { return JointResultBridge::adopt(owner_->family(), bounds_, maximum_); }
    std::shared_ptr<AccessionResultBridge> result() const { return AccessionResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::string free_values() const { return evidence_bridge_detail::vector_bytes(owner_->free_values()); }
private:
    JointEvaluationBridge(rfc::learning::RelationalJointFieldEvaluation owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalJointFieldEvaluation>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {
        const auto l = owner_->family().family().limits(); bounds_ = {l.dimension, l.entries, l.steps, l.integer_bits};
    }
    std::shared_ptr<const rfc::learning::RelationalJointFieldEvaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<JointEvaluationBridge> JointResultBridge::evaluate(const std::vector<std::string>& values) const {
    using namespace family_bridge_detail;
    numeric::ExactActionGuard guard(limits(bounds_)); guard.dimension(values.size()); numeric::RationalVector free;
    for (const auto& value : values) {
        numeric::Rational q; if (q.set_str(value, 16) != 0 || q.get_den() == 0) throw std::invalid_argument("exact complete free values required");
        q.canonicalize(); guard.value(q); free.push_back(std::move(q));
    }
    return JointEvaluationBridge::adopt(owner_->evaluate(std::move(free), limits(bounds_)), bounds_, maximum_);
}
class JointMapBridge {
public:
    AffineMapInfo info(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_.directions().size(), maximum);
        AffineMapInfo out; out.grade = owner_.grade().value; out.occurrence = owner_.occurrence();
        out.particular = program_bridge_detail::matrix_bytes(owner_.particular());
        for (const auto& m : owner_.directions()) out.directions.push_back(program_bridge_detail::matrix_bytes(m));
        return out;
    }
    std::shared_ptr<FieldFamilyBridge> family() const { return FieldFamilyBridge::adopt(owner_.family(), bounds_, maximum_); }
    std::size_t member() const { return owner_.member(); }
private:
    friend class FieldFamilyBridge;
    JointMapBridge(rfc::learning::RelationalJointOccurrenceMap owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::move(owner)), bounds_(std::move(bounds)), maximum_(maximum) {}
    rfc::learning::RelationalJointOccurrenceMap owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<JointMapBridge> FieldFamilyBridge::occurrence_map(std::size_t member, std::size_t grade, std::size_t occurrence) const {
    auto map = owner_->coefficient_independent_map(member, program_bridge_detail::cell(grade, 0).grade, occurrence, program_bridge_detail::limits(bounds_));
    return std::shared_ptr<JointMapBridge>(new JointMapBridge(std::move(map), bounds_, maximum_));
}
} // namespace rexgraph::python
