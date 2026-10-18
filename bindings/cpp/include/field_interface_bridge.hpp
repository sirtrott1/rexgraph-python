#pragma once
#include "field_program_bridge.hpp"
#include <rfc/inference/relational_field_interface_codec.hpp>

namespace rexgraph::python {
struct InterfaceUse {
    std::uint64_t key{};
    std::vector<std::vector<std::uint64_t>> cells, occurrences;
    std::vector<std::vector<std::size_t>> spans;
};
struct InterfaceAssetInfo {
    std::uint64_t key{}, channel{};
    std::string encoding;
    std::size_t byte_count{}, use_count{};
    ProgramSourceInfo source;
};
struct InterfaceAccession {
    std::uint64_t key{}, node{};
    std::size_t grade{};
    std::vector<ProgramStalk> stalks;
    std::vector<std::vector<std::size_t>> cells, occurrences;
    ProgramMatrix accession, reconstruction;
    bool isometric{};
};
struct InterfaceInputInfo {
    std::uint64_t key{}, node{}, use{};
    std::size_t grade{};
    bool isometric{};
    std::string asset, field, represented, accession, reconstruction, source_metric, target_metric;
    std::vector<ProgramStalkInfo> stalks;
    std::vector<std::vector<std::size_t>> cells, occurrences;
};
struct InterfaceNodeCell { std::uint64_t node{}; std::size_t grade{}, cell{}; };
struct InterfacePatch {
    std::uint64_t key{}, branch{};
    std::vector<InterfaceNodeCell> cells;
};
struct InterfaceView {
    std::uint64_t key{}, basis{};
    std::vector<std::uint64_t> directions;
    ProgramMatrix accession, reconstruction, physical;
    bool has_physical{};
};
struct InterfaceReadingInfo {
    std::uint64_t key{}, basis{}, interface_key{}, branch{};
    unsigned status{};
    std::vector<std::uint64_t> directions;
    std::string visible, reconstructed, metric, accession, reconstruction, physical;
    bool has_physical{};
};

namespace interface_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline numeric::SparseRationalTensor field(const std::string& image,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
    return storage::EncodedRationalTensor(bytes(image), reader(bounds, maximum)).materialize();
}
inline InterfaceUse use_info(const inference::RelationalSpanUse& use) {
    InterfaceUse out; out.key = use.key;
    for (const auto cell : use.cells) out.cells.push_back({cell.grade.value, cell.key});
    for (const auto occurrence : use.occurrences) out.occurrences.push_back({occurrence.upper_grade.value, occurrence.key.value});
    for (const auto span : use.spans) out.spans.push_back({span.begin, span.end});
    return out;
}
inline InterfaceInputInfo input_info(const inference::RelationalAccessionInput& input, std::size_t maximum) {
    const auto accession = input.accession(); const auto& d = accession.declaration();
    InterfaceInputInfo out; out.key = d.key; out.node = d.node; out.use = input.use();
    out.grade = d.source_grade.value; out.isometric = d.isometric;
    out.asset = bounded(inference::encode_relational_input_use_asset(accession.asset()), maximum);
    out.field = tensor_bytes(input.source_field()); out.represented = tensor_bytes(input.represented());
    out.accession = matrix_bytes(d.accession); out.reconstruction = matrix_bytes(d.reconstruction);
    out.source_metric = matrix_bytes(accession.source_metric()); out.target_metric = matrix_bytes(accession.target_metric());
    for (const auto& stalk : d.source_stalks) out.stalks.push_back(stalk_info(stalk));
    for (const auto& row : d.instances.cells) out.cells.emplace_back(row.begin(), row.end());
    out.occurrences = d.instances.occurrences;
    return out;
}
inline inference::RelationalInputAccessionDeclaration accession_declaration(
        const InterfaceAccession& value, numeric::ExactActionGuard& guard) {
    inference::RelationalInputAccessionDeclaration out;
    out.key = value.key; out.node = value.node; out.source_grade = cell(value.grade, 0).grade;
    out.isometric = value.isometric;
    for (const auto& s : value.stalks)
        out.source_stalks.push_back({cell(s.grade, s.cell), s.basis, s.directions, matrix(s.metric, guard)});
    for (const auto& row : value.cells) {
        std::vector<std::uint32_t> targets;
        for (const auto index : row) targets.push_back(cell(0, index).index);
        out.instances.cells.push_back(std::move(targets));
    }
    out.instances.occurrences = value.occurrences;
    out.accession = matrix(value.accession, guard); out.reconstruction = matrix(value.reconstruction, guard);
    return out;
}
inline inference::RelationalFieldViewDeclaration view(const InterfaceView& value, numeric::ExactActionGuard& guard) {
    inference::RelationalFieldViewDeclaration out;
    out.key = value.key; out.basis = value.basis; out.directions = value.directions;
    out.accession = matrix(value.accession, guard); out.reconstruction = matrix(value.reconstruction, guard);
    if (value.has_physical) out.physical = matrix(value.physical, guard);
    return out;
}
inline InterfaceReadingInfo reading_info(const inference::RelationalFieldViewReading& reading) {
    const auto& d = reading.declaration(); const auto interface = reading.interface();
    InterfaceReadingInfo out; out.key = d.key; out.basis = d.basis;
    out.interface_key = interface.declaration().key; out.branch = interface.declaration().branch;
    out.status = static_cast<unsigned>(reading.status()); out.directions = d.directions;
    out.visible = tensor_bytes(reading.visible()); out.reconstructed = tensor_bytes(reading.reconstructed());
    out.metric = matrix_bytes(reading.metric()); out.accession = matrix_bytes(d.accession);
    out.reconstruction = matrix_bytes(d.reconstruction); out.has_physical = d.physical.has_value();
    if (d.physical) out.physical = matrix_bytes(*d.physical);
    return out;
}
} // namespace interface_bridge_detail

class InputUseAssetBridge {
public:
    static std::shared_ptr<InputUseAssetBridge> adopt(rfc::inference::OwnedRelationalInputUseAsset owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<InputUseAssetBridge>(new InputUseAssetBridge(std::move(owner), bounds, maximum));
    }
    InputUseAssetBridge(const std::string& source, std::uint64_t key, std::uint64_t channel,
            const std::string& encoding, const std::string& original, const std::vector<InterfaceUse>& uses,
            const std::vector<std::size_t>& bounds, std::size_t maximum)
        : bounds_(bounds), maximum_(maximum) {
        using namespace interface_bridge_detail;
        storage::NativeSeriesLimits admission; admission.max_archive_bytes = maximum;
        admission.temporal = reader(bounds, maximum);
        auto history = storage::decode_native_series(bytes(source), admission);
        inference::RelationalInputUseDeclaration d;
        d.key = key; d.channel = channel; d.encoding = encoding;
        const auto raw = bytes(original); d.bytes.assign(raw.begin(), raw.end());
        for (const auto& value : uses) {
            inference::RelationalSpanUse use; use.key = value.key;
            for (const auto& c : value.cells) {
                if (c.size() != 2) throw std::invalid_argument("original input cell needs grade and stable identity");
                use.cells.push_back({cell(c[0], 0).grade, c[1]});
            }
            for (const auto& p : value.occurrences) {
                if (p.size() != 2) throw std::invalid_argument("original input flag needs grade and stable identity");
                use.occurrences.push_back({cell(p[0], 0).grade, core::OccurrenceKey{p[1]}});
            }
            for (const auto& span : value.spans) {
                if (span.size() != 2) throw std::invalid_argument("ordered byte span needs begin and end");
                use.spans.push_back({span[0], span[1]});
            }
            d.uses.push_back(std::move(use));
        }
        owner_ = std::make_shared<const inference::OwnedRelationalInputUseAsset>(std::move(history), std::move(d), limits(bounds));
    }
    static std::shared_ptr<InputUseAssetBridge> restore(const std::string& archive,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace interface_bridge_detail;
        auto owner = inference::decode_relational_input_use_asset(bytes(archive), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<InputUseAssetBridge>(new InputUseAssetBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_input_use_asset(*owner_), maximum_);
    }
    InterfaceAssetInfo info() const {
        const auto& d = owner_->declaration();
        return {d.key, d.channel, d.encoding, d.bytes.size(), d.uses.size(), program_bridge_detail::source_info(owner_->publication())};
    }
    std::string source_archive() const {
        return program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->history()), maximum_);
    }
    std::vector<InterfaceUse> uses(std::size_t maximum) const {
        if (owner_->declaration().uses.size() > maximum) throw std::length_error("input asset use metadata bound");
        std::vector<InterfaceUse> out;
        for (const auto& use : owner_->declaration().uses) out.push_back(interface_bridge_detail::use_info(use));
        return out;
    }
    std::vector<std::string> recover(std::uint64_t use, std::size_t maximum) const {
        const auto intervals = owner_->recover(use, maximum, program_bridge_detail::limits(bounds_));
        std::vector<std::string> out;
        for (const auto& span : intervals) out.push_back(program_bridge_detail::text(span));
        return out;
    }
    bool same_owner(const InputUseAssetBridge& other) const { return owner_->same_owner(*other.owner_); }
    const rfc::inference::OwnedRelationalInputUseAsset& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalInputUseAsset& native_owner() const && = delete;
private:
    InputUseAssetBridge(rfc::inference::OwnedRelationalInputUseAsset owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalInputUseAsset>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalInputUseAsset> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};

class AccessionInputBridge;
class InputAccessionBridge {
public:
    const rfc::inference::OwnedRelationalInputAccession& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalInputAccession& native_owner() const && = delete;
    static std::shared_ptr<InputAccessionBridge> adopt(rfc::inference::OwnedRelationalInputAccession value,
            std::vector<std::size_t> bounds,std::size_t maximum) {
        return std::shared_ptr<InputAccessionBridge>(new InputAccessionBridge(std::move(value),std::move(bounds),maximum));
    }
    InputAccessionBridge(std::shared_ptr<InputUseAssetBridge> asset, std::shared_ptr<FieldProgramBridge> program,
            const InterfaceAccession& value, const std::vector<std::size_t>& bounds, std::size_t maximum)
        : bounds_(bounds), maximum_(maximum) {
        using namespace interface_bridge_detail;
        if (!asset || !program) throw std::invalid_argument("actual input asset and program owners required");
        numeric::ExactActionGuard guard(limits(bounds)); auto d = accession_declaration(value, guard);
        owner_ = std::make_shared<const inference::OwnedRelationalInputAccession>(asset->native_owner(), program->native_owner(), std::move(d), limits(bounds));
    }
    std::shared_ptr<AccessionInputBridge> admit(std::uint64_t use, const std::string& field) const;
private:
    InputAccessionBridge(rfc::inference::OwnedRelationalInputAccession value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::inference::OwnedRelationalInputAccession>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalInputAccession> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AccessionInputBridge {
public:
    static std::shared_ptr<AccessionInputBridge> adopt(rfc::inference::RelationalAccessionInput owner,
            const std::vector<std::size_t>&, std::size_t maximum) {
        return std::shared_ptr<AccessionInputBridge>(new AccessionInputBridge(std::move(owner), maximum));
    }
    const rfc::inference::RelationalAccessionInput& native_owner() const & { return *owner_; }
    const rfc::inference::RelationalAccessionInput& native_owner() const && = delete;
    InterfaceInputInfo info() const { return interface_bridge_detail::input_info(*owner_, maximum_); }
private:
    friend class InputAccessionBridge;
    AccessionInputBridge(rfc::inference::RelationalAccessionInput owner, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::RelationalAccessionInput>(std::move(owner))), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::RelationalAccessionInput> owner_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AccessionInputBridge> InputAccessionBridge::admit(std::uint64_t use, const std::string& field) const {
    auto input = owner_->admit(use, interface_bridge_detail::field(field, bounds_, maximum_), program_bridge_detail::limits(bounds_));
    return std::shared_ptr<AccessionInputBridge>(new AccessionInputBridge(std::move(input), maximum_));
}

class FieldPatchBridge;
class AccessionResultBridge {
public:
    static std::shared_ptr<AccessionResultBridge> adopt(rfc::inference::OwnedAccessionFieldResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AccessionResultBridge>(new AccessionResultBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AccessionResultBridge> execute(std::shared_ptr<FieldProgramBridge> program,
            const std::vector<std::shared_ptr<AccessionInputBridge>>& inputs,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!program) throw std::invalid_argument("actual accession program owner required");
        std::vector<rfc::inference::RelationalAccessionInput> fields;
        for (const auto& input : inputs) {
            if (!input) throw std::invalid_argument("actual admitted input owner required");
            fields.push_back(input->native_owner());
        }
        auto result = rfc::inference::execute_relational_inputs(program->native_owner(), std::move(fields), program_bridge_detail::limits(bounds));
        return std::shared_ptr<AccessionResultBridge>(new AccessionResultBridge(std::move(result), bounds, maximum));
    }
    static std::shared_ptr<AccessionResultBridge> restore(const std::string& archive,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace interface_bridge_detail;
        auto result = inference::decode_accession_field_result(bytes(archive), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<AccessionResultBridge>(new AccessionResultBridge(std::move(result), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_accession_field_result(*owner_), maximum_);
    }
    std::shared_ptr<FieldProgramResultBridge> result() const { return FieldProgramResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    std::vector<InterfaceInputInfo> inputs(std::size_t maximum) const {
        if (owner_->inputs().size() > maximum) throw std::length_error("accession input metadata bound");
        std::vector<InterfaceInputInfo> out;
        for (const auto& input : owner_->inputs()) out.push_back(interface_bridge_detail::input_info(input, maximum_));
        return out;
    }
    std::shared_ptr<FieldPatchBridge> interface(const InterfacePatch& patch) const;
    const rfc::inference::OwnedAccessionFieldResult& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedAccessionFieldResult& native_owner() const && = delete;
private:
    friend class FieldSectionBridge;
    AccessionResultBridge(rfc::inference::OwnedAccessionFieldResult owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedAccessionFieldResult>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedAccessionFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};

class FieldReadingBridge;
class FieldPatchBridge {
public:
    InterfacePatch info() const {
        const auto& d = owner_->declaration(); InterfacePatch out; out.key = d.key; out.branch = d.branch;
        for (const auto& c : d.cells) out.cells.push_back({c.node, c.cell.grade.value, c.cell.index});
        return out;
    }
    std::string metric() const { return program_bridge_detail::matrix_bytes(owner_->metric()); }
    std::string original() const { return program_bridge_detail::tensor_bytes(owner_->original()); }
    bool same_owner(const FieldPatchBridge& other) const { return owner_->same_owner(*other.owner_); }
    std::shared_ptr<FieldReadingBridge> observe(const InterfaceView& value) const;
    std::shared_ptr<FieldReadingBridge> admit(const InterfaceView& value, const std::string& field, unsigned status) const;
private:
    friend class AccessionResultBridge;
    friend class FieldReadingBridge;
    friend class FieldSectionBridge;
    FieldPatchBridge(rfc::inference::RelationalFieldInterface owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::RelationalFieldInterface>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::RelationalFieldInterface> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<FieldPatchBridge> AccessionResultBridge::interface(const InterfacePatch& patch) const {
    rfc::inference::RelationalFieldInterfaceDeclaration d; d.key = patch.key; d.branch = patch.branch;
    for (const auto& c : patch.cells) d.cells.push_back({c.node, program_bridge_detail::cell(c.grade, c.cell)});
    auto owner = rfc::inference::RelationalFieldInterface(*owner_, std::move(d), program_bridge_detail::limits(bounds_));
    return std::shared_ptr<FieldPatchBridge>(new FieldPatchBridge(std::move(owner), bounds_, maximum_));
}
class FieldReadingBridge {
public:
    InterfaceReadingInfo info() const { return interface_bridge_detail::reading_info(*owner_); }
    std::shared_ptr<FieldPatchBridge> interface() const {
        return std::shared_ptr<FieldPatchBridge>(new FieldPatchBridge(owner_->interface(), bounds_, maximum_));
    }
    const rfc::inference::RelationalFieldViewReading& native_owner() const & { return *owner_; }
    const rfc::inference::RelationalFieldViewReading& native_owner() const && = delete;
private:
    friend class FieldPatchBridge;
    friend class FieldSectionBridge;
    FieldReadingBridge(rfc::inference::RelationalFieldViewReading owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::RelationalFieldViewReading>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::RelationalFieldViewReading> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<FieldReadingBridge> FieldPatchBridge::observe(const InterfaceView& value) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
    auto reading = owner_->observe(interface_bridge_detail::view(value, guard), program_bridge_detail::limits(bounds_));
    return std::shared_ptr<FieldReadingBridge>(new FieldReadingBridge(std::move(reading), bounds_, maximum_));
}
inline std::shared_ptr<FieldReadingBridge> FieldPatchBridge::admit(const InterfaceView& value, const std::string& field, unsigned status) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
    auto reading = owner_->admit(interface_bridge_detail::view(value, guard), interface_bridge_detail::field(field, bounds_, maximum_),
        static_cast<rfc::inference::FieldReadingStatus>(status), program_bridge_detail::limits(bounds_));
    return std::shared_ptr<FieldReadingBridge>(new FieldReadingBridge(std::move(reading), bounds_, maximum_));
}

class FieldSectionBridge {
public:
    static std::shared_ptr<FieldSectionBridge> assemble(const std::vector<std::shared_ptr<FieldReadingBridge>>& readings,
            bool contributions, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        std::vector<rfc::inference::RelationalFieldViewReading> native;
        for (const auto& reading : readings) {
            if (!reading) throw std::invalid_argument("actual field reading owner required");
            native.push_back(reading->native_owner());
        }
        auto section = contributions ? rfc::inference::sum_relational_contributions(std::move(native), program_bridge_detail::limits(bounds))
            : rfc::inference::glue_relational_readings(std::move(native), program_bridge_detail::limits(bounds));
        return std::shared_ptr<FieldSectionBridge>(new FieldSectionBridge(std::move(section), bounds, maximum));
    }
    static std::shared_ptr<FieldSectionBridge> restore(const std::string& archive,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace interface_bridge_detail;
        auto section = inference::decode_relational_field_section(bytes(archive), reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<FieldSectionBridge>(new FieldSectionBridge(std::move(section), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_field_section(*owner_), maximum_);
    }
    unsigned kind() const { return static_cast<unsigned>(owner_->kind()); }
    std::vector<std::shared_ptr<FieldReadingBridge>> readings(std::size_t maximum) const {
        if (owner_->readings().size() > maximum) throw std::length_error("section reading metadata bound");
        std::vector<std::shared_ptr<FieldReadingBridge>> out;
        for (const auto& value : owner_->readings()) out.push_back(std::shared_ptr<FieldReadingBridge>(new FieldReadingBridge(value, bounds_, maximum_)));
        return out;
    }
    std::shared_ptr<AccessionResultBridge> parent() const {
        return std::shared_ptr<AccessionResultBridge>(new AccessionResultBridge(owner_->readings().front().interface().parent(), bounds_, maximum_));
    }
    std::vector<ProgramSelected> select(const std::vector<std::vector<std::size_t>>& cells, std::size_t coordinate_limit) const {
        using namespace interface_bridge_detail;
        const auto interface = owner_->readings().front().interface(); const auto program = interface.parent().result().program();
        const auto& values = owner_->section().values(); std::set<core::CellId> seen; std::size_t total{};
        std::vector<ProgramSelected> out;
        for (const auto& c : cells) {
            if (c.size() != 2) throw std::invalid_argument("section cell requires grade and original index");
            const auto id = cell(c[0], c[1]); const auto found = values.find(id);
            if (!seen.insert(id).second || found == values.end()) throw std::invalid_argument("section needs distinct supported original cells");
            total = machine::checked_size_add(total, found->second.size());
            if (total > coordinate_limit) throw std::length_error("section selected stalk coordinate limit");
            const auto& basis = inference::field_interface_detail::stalk(program, id);
            numeric::ExactActionGuard guard(limits(bounds_));
            ProgramSelected value{stalk_info(basis), tensor_bytes(inference::field_interface_detail::tensor(found->second, guard)), {}};
            for (const auto& q : found->second) value.values.push_back(q.get_str(16));
            out.push_back(std::move(value));
        }
        return out;
    }
private:
    FieldSectionBridge(rfc::inference::OwnedRelationalFieldSection owner, std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalFieldSection>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalFieldSection> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
} // namespace rexgraph::python
