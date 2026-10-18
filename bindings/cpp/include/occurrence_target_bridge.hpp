#pragma once
#include "field_interface_bridge.hpp"
#include <rfc/learning/relational_affine_program_codec.hpp>

namespace rexgraph::python {
struct OccurrenceTargetDeclaration {
    std::uint64_t key{}, revision{}, node{}, frame{};
    std::vector<std::vector<std::uint64_t>> coordinates;
    ProgramMatrix accession, reconstruction;
};
struct OccurrenceTargetInfo {
    std::uint64_t key{}, revision{}, node{}, frame{};
    std::vector<std::vector<std::uint64_t>> coordinates;
    std::string accession, reconstruction, selection, action, coordinate_metric, represented_metric;
    std::size_t retained_entries{};
};
class OccurrenceTargetBridge {
public:
    const rfc::learning::OwnedRelationalOccurrenceTargetPort& native_owner() const & { return *owner_; }
    const rfc::learning::OwnedRelationalOccurrenceTargetPort& native_owner() const && = delete;
    OccurrenceTargetBridge(std::shared_ptr<FieldProgramBridge> program, const OccurrenceTargetDeclaration& value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace program_bridge_detail;
        if (!program || value.coordinates.size() > bounds.at(0)) throw std::invalid_argument("bounded actual target program required");
        rfc::numeric::ExactActionGuard guard(limits(bounds));
        rfc::learning::RelationalOccurrenceTargetDeclaration d;
        d.key=value.key; d.revision=value.revision; d.node=value.node; d.frame=value.frame;
        for (const auto& c : value.coordinates) {
            if(c.size()!=5) throw std::invalid_argument("target key, original grade/flag, basis and direction required");
            d.coordinates.push_back({c[0], {cell(c[1],0).grade, {c[2]}}, c[3], c[4]});
        }
        d.accession=matrix(value.accession,guard); d.reconstruction=matrix(value.reconstruction,guard);
        owner_=std::make_shared<const rfc::learning::OwnedRelationalOccurrenceTargetPort>(program->native_owner(),std::move(d),limits(bounds));
    }
    static std::shared_ptr<OccurrenceTargetBridge> adopt(rfc::learning::OwnedRelationalOccurrenceTargetPort port,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<OccurrenceTargetBridge>(new OccurrenceTargetBridge(std::move(port),bounds,maximum));
    }
    static std::shared_ptr<OccurrenceTargetBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace program_bridge_detail;
        return adopt(rfc::learning::decode_relational_occurrence_target_port(bytes(image),reader(bounds,maximum),limits(bounds)),bounds,maximum);
    }
    std::shared_ptr<OccurrenceTargetBridge> rebind(std::shared_ptr<FieldProgramBridge> program) const {
        using namespace program_bridge_detail;
        if(!program || rfc::inference::encode_relational_field_program(program->native_owner()) !=
                       rfc::inference::encode_relational_field_program(owner_->program()))
            throw std::invalid_argument("explicit target rebind requires identical complete program bytes");
        return adopt(rfc::learning::OwnedRelationalOccurrenceTargetPort(program->native_owner(),owner_->declaration(),limits(bounds_)),bounds_,maximum_);
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::learning::encode_relational_occurrence_target_port(*owner_),maximum_); }
    std::shared_ptr<FieldProgramBridge> program() const { return FieldProgramBridge::adopt(owner_->program(),bounds_,maximum_); }
    bool same_owner(const std::shared_ptr<OccurrenceTargetBridge>& other) const { return other && owner_->same_owner(*other->owner_); }
    OccurrenceTargetInfo info(std::size_t maximum) const {
        using namespace program_bridge_detail;
        const auto& d=owner_->declaration();
        if(d.coordinates.size()>maximum) throw std::length_error("explicit target metadata bound exceeded");
        OccurrenceTargetInfo out; out.key=d.key;out.revision=d.revision;out.node=d.node;out.frame=d.frame;
        for(const auto& c:d.coordinates) out.coordinates.push_back({c.key,c.occurrence.upper_grade.value,c.occurrence.key.value,c.basis,c.direction});
        out.accession=matrix_bytes(d.accession);out.reconstruction=matrix_bytes(d.reconstruction);
        out.selection=matrix_bytes(owner_->selection());out.action=matrix_bytes(owner_->action());
        out.coordinate_metric=matrix_bytes(owner_->coordinate_metric());out.represented_metric=matrix_bytes(owner_->represented_metric());
        out.retained_entries=owner_->retained_entries();return out;
    }
private:
    OccurrenceTargetBridge(rfc::learning::OwnedRelationalOccurrenceTargetPort port,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::OwnedRelationalOccurrenceTargetPort>(std::move(port))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::OwnedRelationalOccurrenceTargetPort> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
}
