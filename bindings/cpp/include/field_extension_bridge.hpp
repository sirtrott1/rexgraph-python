#pragma once

#include "field_evidence_admission_bridge.hpp"
#include <rfc/learning/relational_available_extension_codec.hpp>

namespace rexgraph::python {
struct ExtensionStalk {
    std::size_t grade{},cell{};
    std::uint64_t source_basis{},target_basis{};
    ProgramMatrix inclusion,reconstruction;
};
struct ExtensionDeclaration {
    std::uint64_t key{},revision{},free_carrier{},free_basis{};
    std::vector<std::uint64_t> free_directions;
    std::vector<std::vector<std::size_t>> cells,occurrences;
    std::vector<ExtensionStalk> stalks;
    bool isometric{};
};
struct ExtensionInfo {
    std::uint64_t key{},revision{},free_carrier{},free_basis{};
    unsigned status{};
    bool has_parent{},isometric{};
    std::vector<std::uint64_t> free_directions;
    std::vector<std::vector<std::size_t>> cells,occurrences;
    std::vector<std::vector<std::string>> grade_maps;
};
namespace extension_bridge_detail {
using namespace rfc;using namespace program_bridge_detail;
inline inference::RelationalFieldExtensionDeclaration declaration(const ExtensionDeclaration& d,const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds));inference::RelationalFieldExtensionDeclaration out;
    out.key=d.key;out.revision=d.revision;out.free_carrier=d.free_carrier;out.free_basis=d.free_basis;
    out.free_directions=d.free_directions;out.isometric=d.isometric;guard.dimension(d.stalks.size());guard.dimension(d.free_directions.size());
    for(const auto& row:d.cells) {
        guard.dimension(row.size());std::vector<std::uint32_t> cells;
        for(const auto i:row) {guard.step();cells.push_back(cell(0,i).index);}out.source_map.cells.push_back(std::move(cells));
    }
    out.source_map.occurrences=d.occurrences;
    for(const auto& s:d.stalks)out.stalks.push_back({cell(s.grade,s.cell),s.source_basis,s.target_basis,matrix(s.inclusion,guard),matrix(s.reconstruction,guard)});
    return out;
}
inline learning::AvailableRelationalExtensionParent parent(std::shared_ptr<AvailableProgramBridge> fixed,std::shared_ptr<AvailableEvaluationBridge> selected) {
    if(bool(fixed)==bool(selected))throw std::invalid_argument("one actual fixed or selected extension parent required");
    if(fixed)return fixed->native_owner();
    return selected->native_owner();
}
inline learning::RelationalExtensionStatus status(unsigned value) {
    if(value>1)throw std::invalid_argument("extension requires observed or prospective relationship status");
    return static_cast<learning::RelationalExtensionStatus>(value);
}
inline std::string completion(const numeric::RationalVector& values,const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds));return tensor_bytes(inference::field_interface_detail::tensor(values,guard));
}
} // namespace extension_bridge_detail

class ExtensionFamilyBridge;
class ExtensionEvaluationBridge;
class ExtensionModelBridge {
public:
    ExtensionModelBridge(std::shared_ptr<AvailableProgramBridge> fixed,std::shared_ptr<AvailableEvaluationBridge> selected,
            std::shared_ptr<AvailableProgramBridge> target,const ExtensionDeclaration& d,unsigned status,std::uint64_t origin,
            std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :bounds_(bounds),maximum_(maximum) {
        using namespace extension_bridge_detail;
        if(!target)throw std::invalid_argument("independently admitted actual target required");
        owner_=std::make_shared<const learning::PreparedAvailableRelationalExtension>(extension_bridge_detail::parent(fixed,selected),target->native_owner(),
            declaration(d,bounds),extension_bridge_detail::status(status),origin,available_evidence_bridge_detail::authority(context),limits(bounds));
    }
    static std::shared_ptr<ExtensionModelBridge> adopt(rfc::learning::PreparedAvailableRelationalExtension value,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<ExtensionModelBridge>(new ExtensionModelBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<ExtensionModelBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace extension_bridge_detail;
        return adopt(learning::decode_available_relational_extension(bytes(image),available_evidence_bridge_detail::authority(context),reader(bounds,maximum),limits(bounds)),bounds,maximum);
    }
    static std::string root_payload(std::shared_ptr<AvailableProgramBridge> fixed,std::shared_ptr<AvailableEvaluationBridge> selected,
            std::shared_ptr<AvailableProgramBridge> target,const ExtensionDeclaration& d,unsigned kind,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace extension_bridge_detail;
        if(!target)throw std::invalid_argument("actual extension target required");
        return bounded(learning::relational_extension_payload(extension_bridge_detail::parent(fixed,selected),target->native_owner(),declaration(d,bounds),status(kind)),maximum);
    }
    std::string extension_payload(std::shared_ptr<AvailableProgramBridge> target,const ExtensionDeclaration& d,unsigned kind) const {
        using namespace extension_bridge_detail;
        if(!target)throw std::invalid_argument("actual next target required");
        return bounded(owner_->extension_payload(target->native_owner(),declaration(d,bounds_),status(kind)),maximum_);
    }
    std::shared_ptr<ExtensionModelBridge> extend(std::shared_ptr<AvailableProgramBridge> target,const ExtensionDeclaration& d,unsigned kind,
            std::uint64_t origin,std::shared_ptr<AvailabilityContextBridge> context) const {
        using namespace extension_bridge_detail;
        if(!target)throw std::invalid_argument("actual next target required");
        return adopt(owner_->extend(target->native_owner(),declaration(d,bounds_),status(kind),origin,available_evidence_bridge_detail::authority(context),limits(bounds_)),bounds_,maximum_);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_extension(*owner_),maximum_);}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(available_evidence_bridge_detail::authority(context));}
    std::shared_ptr<FieldProgramBridge> source() const {return FieldProgramBridge::adopt(owner_->extension().source(),bounds_,maximum_);}
    std::shared_ptr<AvailableProgramBridge> target() const {return AvailableProgramBridge::adopt(owner_->target(),bounds_,maximum_);}
    std::shared_ptr<ExtensionModelBridge> parent() const {const auto value=owner_->parent();return value?adopt(*value,bounds_,maximum_):nullptr;}
    std::shared_ptr<AvailableProgramBridge> fixed_parent() const {
        const auto root=owner_->root();if(!root)return nullptr;
        const auto p=std::get_if<rfc::inference::PreparedAvailableRelationalProgram>(&*root);
        return p?AvailableProgramBridge::adopt(*p,bounds_,maximum_):nullptr;
    }
    std::shared_ptr<AvailableEvaluationBridge> selected_parent() const {
        const auto root=owner_->root();if(!root)return nullptr;
        const auto p=std::get_if<rfc::learning::AvailableRelationalAffineEvaluation>(&*root);
        return p?AvailableEvaluationBridge::adopt(*p,bounds_,maximum_):nullptr;
    }
    ExtensionInfo info() const {
        using namespace program_bridge_detail;const auto extension=owner_->extension();const auto& d=extension.declaration();ExtensionInfo out;
        out.key=d.key;out.revision=d.revision;out.free_carrier=d.free_carrier;out.free_basis=d.free_basis;out.free_directions=d.free_directions;
        out.isometric=d.isometric;out.status=static_cast<unsigned>(owner_->status());out.has_parent=owner_->parent().has_value();
        for(const auto& cells:d.source_map.cells)out.cells.emplace_back(cells.begin(),cells.end());
        out.occurrences=d.source_map.occurrences;
        const auto target=extension.target();const auto count=target.fibers().fibers().complex().grade_count();
        for(std::size_t k=0;k<count;++k) {
            const auto grade=cell(k,0).grade;out.grade_maps.push_back({matrix_bytes(extension.inclusion(grade)),matrix_bytes(extension.reconstruction(grade)),matrix_bytes(extension.kernel(grade))});
            for(const auto& image:out.grade_maps.back())if(image.size()>maximum_)throw std::length_error("extension matrix byte admission");
        }
        return out;
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        using namespace program_bridge_detail;std::vector<std::string> out;
        if(const auto previous=parent())out=previous->dependency_archives(maximum);
        else if(const auto fixed=fixed_parent())out=fixed->dependency_archives(maximum);
        else out=selected_parent()->dependency_archives(maximum);
        auto next=target()->dependency_archives(maximum);out.insert(out.end(),next.begin(),next.end());
        out.push_back(bounded(rfc::storage::encode_native_series(owner_->origin().publication().history()),maximum_));
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
    std::shared_ptr<InputAccessionBridge> transport(std::shared_ptr<InputAccessionBridge> original,std::uint64_t key,
            std::shared_ptr<AvailabilityContextBridge> context) const {
        if(!original)throw std::invalid_argument("actual original accession required");
        admit(context);
        const auto extension=owner_->extension();const auto& old=original->native_owner();
        auto d=extension.transport_accession(old,key,program_bridge_detail::limits(bounds_));
        return InputAccessionBridge::adopt(rfc::inference::OwnedRelationalInputAccession(old.asset(),extension.target(),std::move(d),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
    }
    std::shared_ptr<ExtensionFamilyBridge> execute(const std::vector<AdmissionInput>& rows,std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    ExtensionModelBridge(rfc::learning::PreparedAvailableRelationalExtension value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::PreparedAvailableRelationalExtension>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedAvailableRelationalExtension> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};

class ExtensionFamilyBridge {
public:
    static std::shared_ptr<ExtensionFamilyBridge> adopt(rfc::learning::AvailableRelationalExtensionFamily value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<ExtensionFamilyBridge>(new ExtensionFamilyBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<ExtensionFamilyBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace extension_bridge_detail;
        return adopt(learning::decode_available_relational_extension_family(bytes(image),available_evidence_bridge_detail::authority(context),reader(bounds,maximum),limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_extension_family(*owner_),maximum_);}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(available_evidence_bridge_detail::authority(context));}
    std::shared_ptr<ExtensionModelBridge> model() const {return ExtensionModelBridge::adopt(owner_->model(),bounds_,maximum_);}
    std::shared_ptr<AccessionResultBridge> input() const {return AccessionResultBridge::adopt(owner_->family().input(),bounds_,maximum_);}
    AffineFieldInfo field(std::uint64_t node,bool occurrences) const {
        const auto family=owner_->family();
        const auto encode=[](const auto& value) {return evidence_bridge_detail::field_info({value.particular,value.directions});};
        if(!occurrences)return encode(family.field(node));
        for(const auto& field:family.fields())if(field.node==node && field.occurrences)return encode(*field.occurrences);
        throw std::out_of_range("extension node has no occurrence field");
    }
    std::vector<ProgramFieldInfo> fields(std::size_t maximum) const {
        const auto family=owner_->family();const auto& fields=family.fields();evidence_bridge_detail::bound(fields.size(),maximum);
        std::vector<ProgramFieldInfo> out;
        for(const auto& value:fields)out.push_back({value.node,value.grade.value,value.value.particular.domain().logical_size(),value.value.particular.support().count(),value.value.particular.values().nonzeros(),
            value.occurrences?value.occurrences->particular.domain().logical_size():0,value.occurrences.has_value()});
        return out;
    }
    std::vector<AffineSelected> select(std::uint64_t node,const std::vector<std::vector<std::size_t>>& cells,std::size_t maximum) const {
        using namespace evidence_bridge_detail;
        const auto family=owner_->family();const auto program=family.extension().target();const auto& field=family.field(node);
        const auto grade=rfc::learning::relational_evidence_detail::node(program,node).output_grade;
        const auto offsets=program.fibers().fibers().stalks().offsets(grade);
        rfc::numeric::ExactActionGuard guard(limits(bounds_));std::size_t used{};
        std::vector<AffineSelected> out;std::set<std::size_t> seen;
        for(const auto& c:cells){
            if(c.size()!=2 || c[0]!=grade.value || offsets.empty() || c[1]>=offsets.size()-1)throw std::out_of_range("original extension stalk selection");
            if(!seen.insert(c[1]).second)throw std::invalid_argument("distinct extension stalk selection required");
            const auto original=cell(c[0],c[1]);const auto count=offsets[c[1]+1]-offsets[c[1]];
            if(used>maximum || count>maximum-used)throw std::length_error("extension stalk coordinate bound");
            used+=count;AffineSelected value;const auto& stalks=program.declaration().stalks;
            const auto found=std::find_if(stalks.begin(),stalks.end(),[&](const auto& s){return s.cell==original;});
            if(found==stalks.end())throw std::out_of_range("actual extension stalk basis");
            value.particular.stalk=stalk_info(*found);rfc::numeric::RationalVector selected;
            rfc::numeric::ExactMatrix directions(count,field.directions.cols());
            for(std::size_t i=0;i<count;++i){guard.step();const auto q=field.particular.state(offsets[c[1]]+i).value();
                selected.push_back(q);value.particular.values.push_back(q.get_str(16));value.identified.push_back(true);}
            for(const auto& [ij,q]:field.directions.entries()){guard.step();if(ij.first>=offsets[c[1]] && ij.first<offsets[c[1]+1]){
                directions.set(ij.first-offsets[c[1]],ij.second,q);value.identified[ij.first-offsets[c[1]]]=false;}}
            value.particular.field=tensor_bytes(rfc::inference::field_interface_detail::tensor(selected,guard));
            value.directions=matrix_bytes(directions);out.push_back(std::move(value));
        }
        return out;
    }
    std::string completion_payload(const std::vector<std::string>& values) const {
        using namespace extension_bridge_detail;
        return bounded(learning::relational_extension_completion_payload(*owner_,available_evidence_bridge_detail::free_values(values,bounds_)),maximum_);
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out=model()->dependency_archives(maximum);
        auto origins=admission_bridge_detail::dependencies(owner_->obligations(),maximum_);out.insert(out.end(),origins.begin(),origins.end());
        for(const auto& row:owner_->inputs())out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(row.input.asset.history()),maximum_));
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
    std::shared_ptr<ExtensionEvaluationBridge> evaluate(const std::vector<std::string>& values,std::uint64_t origin,std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    ExtensionFamilyBridge(rfc::learning::AvailableRelationalExtensionFamily value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::AvailableRelationalExtensionFamily>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalExtensionFamily> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};

class ExtensionEvaluationBridge {
public:
    const rfc::learning::AvailableRelationalExtensionEvaluation& native_owner() const & {return *owner_;}
    const rfc::learning::AvailableRelationalExtensionEvaluation& native_owner() const && = delete;
    static std::shared_ptr<ExtensionEvaluationBridge> adopt(rfc::learning::AvailableRelationalExtensionEvaluation value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<ExtensionEvaluationBridge>(new ExtensionEvaluationBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<ExtensionEvaluationBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace extension_bridge_detail;
        return adopt(learning::decode_available_relational_extension_evaluation(bytes(image),available_evidence_bridge_detail::authority(context),reader(bounds,maximum),limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_extension_evaluation(*owner_),maximum_);}
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(available_evidence_bridge_detail::authority(context));}
    std::shared_ptr<ExtensionFamilyBridge> family() const {return ExtensionFamilyBridge::adopt(owner_->family(),bounds_,maximum_);}
    std::shared_ptr<FieldProgramResultBridge> result() const {return FieldProgramResultBridge::adopt(owner_->result().result(),bounds_,maximum_);}
    std::string free_values() const {const auto result=owner_->result();return extension_bridge_detail::completion(result.free_values(),bounds_);}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out=family()->dependency_archives(maximum);out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->origin().publication().history()),maximum_));
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
private:
    ExtensionEvaluationBridge(rfc::learning::AvailableRelationalExtensionEvaluation value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::AvailableRelationalExtensionEvaluation>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalExtensionEvaluation> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
inline std::shared_ptr<ExtensionFamilyBridge> ExtensionModelBridge::execute(const std::vector<AdmissionInput>& rows,std::shared_ptr<AvailabilityContextBridge> context) const {
    using namespace extension_bridge_detail;const auto frozen=rows;const auto authority=context;
    owner_->admit(available_evidence_bridge_detail::authority(authority));
    return ExtensionFamilyBridge::adopt(owner_->execute_inputs(available_evidence_bridge_detail::submissions(frozen,available_evidence_bridge_detail::authority(authority),bounds_,maximum_),
        available_evidence_bridge_detail::authority(authority),limits(bounds_)),bounds_,maximum_);
}
inline std::shared_ptr<ExtensionEvaluationBridge> ExtensionFamilyBridge::evaluate(const std::vector<std::string>& values,std::uint64_t origin,std::shared_ptr<AvailabilityContextBridge> context) const {
    return ExtensionEvaluationBridge::adopt(owner_->evaluate(available_evidence_bridge_detail::free_values(values,bounds_),origin,
        available_evidence_bridge_detail::authority(context),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
} // namespace rexgraph::python
