#pragma once

#include <rfc/inference/relational_field_program_codec.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <rfc/storage/encoded_tensor.hpp>
#include <memory>
#include <set>
#include <string>

namespace rexgraph::python {
struct ProgramEntry { std::size_t row{}, column{}; std::string value; };
struct ProgramMatrix {
    std::size_t rows{}, columns{};
    std::vector<ProgramEntry> entries;
};
struct ProgramStalk {
    std::size_t grade{}, cell{};
    std::uint64_t basis{};
    std::vector<std::uint64_t> directions;
    ProgramMatrix metric;
};
struct ProgramOccurrence {
    std::size_t grade{}, slot{};
    unsigned role{};
    ProgramMatrix map;
};
struct ProgramNode {
    std::uint64_t key{}, input{};
    unsigned kind{};
    std::size_t grade{};
    bool has_input{};
    ProgramMatrix map;
};
struct ProgramInput { std::uint64_t node{}, use{}; std::string field; };
struct ProgramSourceInfo {
    std::uint64_t name_space{}, sequence{};
    std::size_t semantic_scale{};
    std::string endpoint, archive;
    std::vector<std::vector<std::string>> coordinates;
};
struct ProgramStalkInfo {
    std::size_t grade{}, cell{};
    std::uint64_t basis{};
    std::vector<std::uint64_t> directions;
    std::string metric;
};
struct ProgramOccurrenceInfo {
    std::size_t grade{}, slot{}, upper{}, lower{}, dimension{};
    std::uint64_t key{};
    unsigned role{};
    bool active{};
    std::string coefficient, map;
};
struct ProgramFieldInfo {
    std::uint64_t node{};
    std::size_t grade{}, dimension{}, supported{}, nonzeros{}, occurrence_dimension{};
    bool has_occurrences{};
};
struct ProgramSelected {
    ProgramStalkInfo stalk;
    std::string field;
    std::vector<std::string> values;
};

namespace program_bridge_detail {
using namespace rfc;
inline std::span<const std::byte> bytes(const std::string& value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}
inline std::string text(std::span<const std::byte> value) {
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
inline std::string bounded(std::span<const std::byte> value, std::size_t maximum) {
    if (value.size() > maximum) throw std::length_error("field program archive byte limit");
    return text(value);
}
inline numeric::ExactActionLimits limits(const std::vector<std::size_t>& bounds) {
    if (bounds.size() != 4 || std::any_of(bounds.begin(), bounds.end(), [](auto n) { return !n; }))
        throw std::invalid_argument("four positive field program action bounds required");
    return {bounds[0], bounds[1], bounds[2], bounds[3]};
}
inline storage::TemporalReadLimits reader(const std::vector<std::size_t>& bounds, std::size_t maximum) {
    (void)limits(bounds);
    if (!maximum) throw std::invalid_argument("positive field program archive byte bound required");
    storage::TemporalReadLimits policy;
    policy.max_file_bytes = maximum; policy.max_records = bounds[1];
    policy.max_dimension = std::max(bounds[0], bounds[1]); policy.max_logical_size = policy.max_dimension;
    policy.max_integer_bytes = bounds[3] / 8u + (bounds[3] % 8u != 0);
    return policy;
}
inline core::CellId cell(std::size_t grade, std::size_t index) {
    if (grade > std::numeric_limits<std::uint16_t>::max() || index > std::numeric_limits<std::uint32_t>::max())
        throw std::out_of_range("field program original cell index");
    return {core::Grade{static_cast<std::uint16_t>(grade)}, static_cast<std::uint32_t>(index)};
}
inline numeric::ExactMatrix matrix(const ProgramMatrix& value, numeric::ExactActionGuard& guard) {
    guard.dimension(value.rows); guard.dimension(value.columns); guard.entries(value.entries.size());
    numeric::ExactMatrix out(value.rows, value.columns);
    std::set<std::pair<std::size_t, std::size_t>> used;
    for (const auto& entry : value.entries) {
        guard.step(); numeric::Rational q;
        if (q.set_str(entry.value, 16) != 0 || q.get_den() == 0 ||
                !used.emplace(entry.row, entry.column).second)
            throw std::invalid_argument("distinct exact field program map coordinates required");
        q.canonicalize(); guard.value(q); out.set(entry.row, entry.column, q);
    }
    return out;
}
inline std::string matrix_bytes(const numeric::ExactMatrix& value) {
    storage::temporal_codec_detail::Writer writer; storage::temporal_codec_detail::matrix(writer, value);
    return text(writer.bytes());
}
inline std::string tensor_bytes(const numeric::SparseRationalTensor& value) {
    storage::temporal_codec_detail::Writer writer; storage::temporal_codec_detail::tensor(writer, value);
    return text(writer.bytes());
}
inline ProgramSourceInfo source_info(const storage::NativeSeriesEndpoint& publication) {
    const auto& endpoint = publication.endpoint();
    ProgramSourceInfo info; info.name_space = publication.source_namespace(); info.sequence = publication.sequence();
    info.semantic_scale = endpoint.source.semantic_scale().value;
    const auto source = storage::encode_rfcx(endpoint.source).bytes;
    const auto channels = storage::encode_temporal_endpoint(endpoint);
    storage::temporal_codec_detail::Writer writer;
    writer.u64(source.size()); writer.raw(source); writer.u64(channels.size()); writer.raw(channels);
    info.endpoint = text(writer.bytes());
    info.archive = text(storage::encode_native_series(storage::NativeTemporalSeries(endpoint, info.name_space)));
    for (std::size_t k = 0; k < endpoint.source.grade_count(); ++k) {
        const auto& arena = endpoint.source.grade(core::Grade{static_cast<std::uint16_t>(k)});
        std::vector<std::string> keys;
        if (k == 0) for (const auto key : arena.identities) keys.push_back(std::to_string(key));
        else for (const auto key : arena.relation_identities) keys.push_back(std::to_string(key.value));
        info.coordinates.push_back(std::move(keys));
    }
    return info;
}
inline ProgramSourceInfo source_info(const inference::OwnedRelationalFieldProgram& program) {
    return source_info(program.publication());
}
inline ProgramStalkInfo stalk_info(const inference::StalkBasisDeclaration& value) {
    return {value.cell.grade.value, value.cell.index, value.basis, value.directions, matrix_bytes(value.metric)};
}
} // namespace program_bridge_detail

class FieldProgramResultBridge;
class FieldProgramBridge {
public:
    FieldProgramBridge(const std::string& source, std::uint64_t key, std::uint64_t revision,
            const std::vector<ProgramStalk>& stalks, const std::vector<ProgramOccurrence>& occurrences,
            const std::vector<ProgramNode>& nodes, bool differential,
            const std::vector<std::size_t>& bounds, std::size_t maximum)
        : bounds_(bounds), maximum_(maximum) {
        using namespace program_bridge_detail;
        const auto policy = reader(bounds, maximum); storage::NativeSeriesLimits admission;
        admission.max_archive_bytes = maximum; admission.temporal = policy;
        const auto publication = storage::decode_native_series(bytes(source), admission).tail();
        numeric::ExactActionGuard guard(limits(bounds));
        inference::RelationalFieldProgramDeclaration declaration;
        declaration.key = key; declaration.revision = revision; declaration.differential = differential;
        for (const auto& value : stalks)
            declaration.stalks.push_back({cell(value.grade, value.cell), value.basis, value.directions, matrix(value.metric, guard)});
        for (const auto& value : occurrences) {
            if (value.role > 1) throw std::invalid_argument("field program map role");
            declaration.occurrences.push_back({cell(value.grade, 0).grade, value.slot,
                value.role == 0 ? storage::TemporalTensorRole::RealizedIncidenceMap : storage::TemporalTensorRole::CoefficientIndependentMap,
                matrix(value.map, guard)});
        }
        for (const auto& value : nodes) {
            if (value.kind > 2) throw std::invalid_argument("field program node kind");
            declaration.nodes.push_back({value.key, static_cast<inference::FieldProgramNodeKind>(value.kind),
                cell(value.grade, 0).grade, value.has_input ? std::optional<std::uint64_t>{value.input} : std::nullopt,
                matrix(value.map, guard)});
        }
        owner_ = std::make_shared<const inference::OwnedRelationalFieldProgram>(publication, std::move(declaration), limits(bounds));
    }
    static std::shared_ptr<FieldProgramBridge> restore(const std::string& archive,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace program_bridge_detail;
        auto owner = inference::decode_relational_field_program(bytes(archive), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<FieldProgramBridge>(new FieldProgramBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<FieldProgramBridge> adopt(rfc::inference::OwnedRelationalFieldProgram owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<FieldProgramBridge>(new FieldProgramBridge(std::move(owner), bounds, maximum));
    }
    const rfc::inference::OwnedRelationalFieldProgram& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalFieldProgram& native_owner() const && = delete;
    std::string archive() const { return program_bridge_detail::bounded(rfc::inference::encode_relational_field_program(*owner_), maximum_); }
    ProgramSourceInfo source() const { return program_bridge_detail::source_info(*owner_); }
    std::uint64_t key() const { return owner_->declaration().key; }
    std::uint64_t revision() const { return owner_->declaration().revision; }
    bool differential() const { return owner_->declaration().differential; }
    std::vector<ProgramNode> nodes(std::size_t maximum) const {
        if (owner_->declaration().nodes.size() > maximum) throw std::length_error("field program node metadata bound");
        std::vector<ProgramNode> out;
        for (const auto& node : owner_->declaration().nodes)
            out.push_back({node.key, node.input.value_or(0), static_cast<unsigned>(node.kind), node.output_grade.value,
                          node.input.has_value(), {}});
        return out;
    }
    std::vector<ProgramStalkInfo> stalks(std::size_t maximum) const {
        if (owner_->declaration().stalks.size() > maximum) throw std::length_error("field program stalk metadata bound");
        std::vector<ProgramStalkInfo> out;
        for (const auto& value : owner_->declaration().stalks) out.push_back(program_bridge_detail::stalk_info(value));
        return out;
    }
    std::vector<ProgramOccurrenceInfo> occurrences(std::size_t maximum) const {
        if (owner_->declaration().occurrences.size() > maximum) throw std::length_error("field program occurrence metadata bound");
        const auto publication = owner_->publication(); const auto& source = publication.endpoint().source;
        std::vector<ProgramOccurrenceInfo> out;
        for (const auto& value : owner_->declaration().occurrences) {
            const auto& arena = source.grade(value.grade);
            out.push_back({value.grade.value, value.occurrence, arena.upper_index[value.occurrence], arena.lower_index[value.occurrence],
                value.map.rows(), arena.occurrence_identities[value.occurrence].value,
                value.role == rfc::storage::TemporalTensorRole::RealizedIncidenceMap ? 0u : 1u,
                arena.aspects.get(source.signature().existence_aspect(), value.occurrence),
                source.incidence_coefficient(value.grade, value.occurrence).get_str(16), program_bridge_detail::matrix_bytes(value.map)});
        }
        return out;
    }
    std::shared_ptr<FieldProgramResultBridge> execute(const std::vector<ProgramInput>& inputs) const;
private:
    friend class FieldProgramResultBridge;
    FieldProgramBridge(rfc::inference::OwnedRelationalFieldProgram owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalFieldProgram>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalFieldProgram> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};

class FieldProgramResultBridge {
public:
    static std::shared_ptr<FieldProgramResultBridge> restore(const std::string& archive,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace program_bridge_detail;
        auto owner = rfc::inference::decode_relational_field_result(bytes(archive), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<FieldProgramResultBridge>(new FieldProgramResultBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<FieldProgramResultBridge> adopt(rfc::inference::OwnedRelationalFieldResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<FieldProgramResultBridge>(new FieldProgramResultBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::inference::encode_relational_field_result(*owner_), maximum_); }
    std::shared_ptr<FieldProgramBridge> program() const {
        return std::shared_ptr<FieldProgramBridge>(new FieldProgramBridge(owner_->program(), bounds_, maximum_));
    }
    std::vector<ProgramInput> inputs(std::size_t maximum) const {
        if (owner_->inputs().size() > maximum) throw std::length_error("field program input metadata bound");
        std::vector<ProgramInput> out;
        for (const auto& value : owner_->inputs()) out.push_back({value.node, value.use, program_bridge_detail::tensor_bytes(value.field)});
        return out;
    }
    std::vector<ProgramFieldInfo> fields(std::size_t maximum) const {
        const auto program = owner_->program(); std::vector<ProgramFieldInfo> out;
        if (program.declaration().nodes.size() > maximum) throw std::length_error("field program node metadata bound");
        for (const auto& node : program.declaration().nodes) {
            const auto& value = owner_->field(node.key);
            const bool occurrences = node.kind == rfc::inference::FieldProgramNodeKind::Boundary;
            out.push_back({node.key, node.output_grade.value, value.domain().logical_size(), value.support().count(),
                value.values().nonzeros(), occurrences ? owner_->occurrence_field(node.key).domain().logical_size() : 0, occurrences});
        }
        return out;
    }
    std::string field(std::uint64_t node, bool occurrences) const {
        const auto encoded = program_bridge_detail::tensor_bytes(occurrences ? owner_->occurrence_field(node) : owner_->field(node));
        return program_bridge_detail::bounded(program_bridge_detail::bytes(encoded), maximum_);
    }
    std::vector<ProgramSelected> select(std::uint64_t node, const std::vector<std::vector<std::size_t>>& cells,
            std::size_t coordinate_limit) const {
        using namespace program_bridge_detail;
        std::vector<rfc::core::CellId> requested;
        for (const auto& value : cells) {
            if (value.size() != 2) throw std::invalid_argument("field program cell requires grade and original index");
            requested.push_back(cell(value[0], value[1]));
        }
        const auto selected = owner_->select(node, std::move(requested), coordinate_limit, limits(bounds_));
        const auto program = owner_->program(); std::vector<ProgramSelected> out;
        for (const auto& value : selected) {
            const auto& declarations = program.declaration().stalks;
            const auto found = std::find_if(declarations.begin(), declarations.end(), [&](const auto& d) { return d.cell == value.cell; });
            ProgramSelected row{stalk_info(*found), tensor_bytes(value.field), {}};
            for (const auto& q : value.field.values().dense()) row.values.push_back(q.get_str(16));
            out.push_back(std::move(row));
        }
        return out;
    }
private:
    friend class FieldProgramBridge;
    FieldProgramResultBridge(rfc::inference::OwnedRelationalFieldResult owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalFieldResult>(std::move(owner))), bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};

inline std::shared_ptr<FieldProgramResultBridge> FieldProgramBridge::execute(const std::vector<ProgramInput>& inputs) const {
    using namespace program_bridge_detail;
    std::vector<rfc::inference::RelationalFieldInput> declarations;
    const auto policy = reader(bounds_, maximum_);
    rfc::numeric::ExactActionGuard guard(limits(bounds_)); std::size_t retained = inputs.size(), image_bytes{};
    guard.entries(retained);
    for (const auto& value : inputs) {
        guard.step(); image_bytes = rfc::machine::checked_size_add(image_bytes, value.field.size());
        if (image_bytes > maximum_) throw std::length_error("field program input archive byte limit");
        rfc::storage::EncodedRationalTensor field(bytes(value.field), policy);
        retained = rfc::machine::checked_size_add(retained, field.stored_nonzeros()); guard.entries(retained);
        guard.dimension(field.domain().logical_size());
        declarations.push_back({value.node, value.use, field.materialize()});
    }
    auto result = owner_->execute(std::move(declarations), limits(bounds_));
    return std::shared_ptr<FieldProgramResultBridge>(new FieldProgramResultBridge(std::move(result), bounds_, maximum_));
}
} // namespace rexgraph::python
