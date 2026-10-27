#pragma once
#include "finite_field_bridge.hpp"
#include "task_outcome_bridge.hpp"
#include <rfc/learning/relational_assignment_evidence_codec.hpp>

namespace rexgraph::python {
struct AssignmentDeclaration {
    std::uint64_t key{}, revision{}, origin{}, parameter_carrier{};
    std::vector<std::uint64_t> parameters;
};
struct AssignmentCandidate {
    std::uint64_t key{};
    std::vector<std::uint64_t> assignment;
    TaskBinding binding;
    std::shared_ptr<AvailableEvidenceBridge> evidence;
    std::vector<std::string> shift;
    ProgramMatrix injection;
};
struct AssignmentInput {
    std::uint64_t candidate{};
    std::shared_ptr<AvailableResultBridge> input;
};
struct AssignmentObservation {
    std::uint64_t key{}, origin{};
    TaskBinding binding;
    std::shared_ptr<AvailableResultBridge> measured;
    std::vector<AssignmentInput> inputs;
};
struct AssignmentCoordinates {
    std::string shift, map;
    std::size_t original_dimension{}, free_dimension{};
};
struct AssignmentSummary {
    AssignmentDeclaration declaration;
    std::uint64_t revision{};
    std::size_t candidates{}, observations{}, remainder{}, retained_entries{}, identified_candidate{};
    unsigned status{};
    bool identified{}, has_parent{};
};
struct AssignmentBranchInfo {
    bool rejected{};
    std::size_t solves{};
    AssignmentCoordinates coordinates;
};
struct AssignmentMapInfo {
    std::string particular;
    std::vector<std::string> directions;
};
struct AssignmentBatchInfo {
    std::vector<std::string> dependencies;
    std::size_t origin_count{};
};
namespace assignment_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
using Model=learning::PreparedRelationalAssignmentEvidence;
using Response=learning::RelationalAssignmentResponse;
inline learning::RelationalAssignmentDeclaration declaration(const AssignmentDeclaration& d) {
    return {d.key,d.revision,d.origin,d.parameter_carrier,d.parameters};
}
inline std::vector<learning::RelationalAssignmentCandidate> candidates(const std::vector<AssignmentCandidate>& values,
        const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds));std::vector<learning::RelationalAssignmentCandidate> out;
    for(const auto& value:values) {
        if(!value.evidence) throw std::invalid_argument("actual assignment evidence root required");
        guard.dimension(value.shift.size());numeric::RationalVector shift;
        for(const auto& q:value.shift) {
            numeric::Rational parsed;
            if(parsed.set_str(q,16)!=0 || parsed.get_den()==0) throw std::invalid_argument("exact assignment shift required");
            parsed.canonicalize();guard.value(parsed);shift.push_back(std::move(parsed));
        }
        out.push_back({value.key,value.assignment,task_bridge_detail::binding(value.binding,bounds),
            value.evidence->native_owner(),std::move(shift),matrix(value.injection,guard)});
    }
    return out;
}
inline learning::RelationalAssignmentObservation observation(const AssignmentObservation& value,const std::vector<std::size_t>& bounds) {
    if(!value.measured) throw std::invalid_argument("actual returned measurement owner required");
    std::vector<learning::RelationalAssignmentInput> inputs;
    for(const auto& row:value.inputs) {
        if(!row.input) throw std::invalid_argument("actual candidate input owner required");
        inputs.push_back({row.candidate,row.input->native_owner()});
    }
    return {value.key,value.origin,task_bridge_detail::binding(value.binding,bounds),value.measured->native_owner(),std::move(inputs)};
}
inline AssignmentCoordinates coordinates(const learning::RelationalAssignmentCoordinates& c,std::size_t maximum) {
    auto shift=evidence_bridge_detail::vector_bytes(c.shift),map=matrix_bytes(c.map);
    if(shift.size()>maximum || map.size()>maximum) throw std::length_error("assignment coordinate byte bound");
    return {std::move(shift),std::move(map),c.map.rows(),c.map.cols()};
}
inline TaskCertificate certificate(const learning::RelationalTaskPostconditionCertificate& p,std::size_t maximum) {
    TaskCertificate out{static_cast<unsigned>(p.status),p.rank,tensor_bytes(p.residual.particular),
        matrix_bytes(p.residual.directions),evidence_bridge_detail::vector_bytes(p.feasible_shift),
        matrix_bytes(p.feasible_kernel),evidence_bridge_detail::vector_bytes(p.contradiction)};
    for(const auto* b:{&out.residual,&out.directions,&out.feasible_shift,&out.feasible_kernel,&out.contradiction})
        if(b->size()>maximum) throw std::length_error("assignment certificate byte bound");
    return out;
}
inline ProgramMatrix matrix_info(const numeric::ExactMatrix& m,std::size_t& used,std::size_t maximum) {
    used=machine::checked_size_add(used,m.nonzeros());evidence_bridge_detail::bound(used,maximum);
    ProgramMatrix out;out.rows=m.rows();out.columns=m.cols();
    for(const auto& [ij,q]:m.entries()) out.entries.push_back({ij.first,ij.second,q.get_str(16)});
    return out;
}
inline TaskBinding binding_info(const learning::RelationalWorkerBindingDeclaration& b,std::size_t& used,std::size_t maximum) {
    TaskBinding out;out.key=b.key;out.revision=b.revision;out.worker=b.worker;out.machine=b.machine;
    out.operation=b.operation;out.origin=b.origin;out.interface.key=b.interface.key;out.interface.branch=b.interface.branch;
    used=machine::checked_size_add(used,machine::checked_size_add(b.interface.cells.size(),b.view.directions.size()));
    evidence_bridge_detail::bound(used,maximum);
    for(const auto& c:b.interface.cells) out.interface.cells.push_back({c.node,c.cell.grade.value,c.cell.index});
    out.view.key=b.view.key;out.view.basis=b.view.basis;out.view.directions=b.view.directions;
    out.view.accession=matrix_info(b.view.accession,used,maximum);out.view.reconstruction=matrix_info(b.view.reconstruction,used,maximum);
    out.view.has_physical=b.view.physical.has_value();
    if(b.view.physical) out.view.physical=matrix_info(*b.view.physical,used,maximum);
    return out;
}
inline void append_dependencies(std::vector<std::string>& target,std::vector<std::string> values,std::size_t maximum) {
    evidence_bridge_detail::bound(machine::checked_size_add(target.size(),values.size()),maximum);
    target.insert(target.end(),std::make_move_iterator(values.begin()),std::make_move_iterator(values.end()));
}
inline std::vector<std::string> input_dependencies(const inference::AvailableRelationalFieldResult& input,
        const std::vector<std::size_t>& bounds,std::size_t maximum,std::size_t records) {
    auto out=AvailableResultBridge::adopt(input,bounds,maximum)->dependency_archives(records);
    const auto value=input.result();std::vector<std::string> sources;
    sources.push_back(FieldProgramBridge::adopt(value.result().program(),bounds,maximum)->source().archive);
    for(const auto& use:value.inputs())sources.push_back(InputUseAssetBridge::adopt(use.accession().asset(),bounds,maximum)->source_archive());
    append_dependencies(out,std::move(sources),records);return out;
}
} // namespace assignment_bridge_detail

class AssignmentResponseBridge;
class AssignmentSolveBridge;
class AssignmentModelBridge {
public:
    AssignmentModelBridge(const AssignmentDeclaration& d,std::shared_ptr<FiniteSourceBridge> finite,
            const std::vector<AssignmentCandidate>& values,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check)
        :bounds_(bounds),maximum_(maximum) {
        using namespace assignment_bridge_detail;
        if(!finite) throw std::invalid_argument("actual assignment finite source required");
        owner_=std::make_shared<const Model>(declaration(d),finite->native_owner(),candidates(values,bounds),
            available_evidence_bridge_detail::authority(context),limits(bounds),check.callback());
    }
    AssignmentModelBridge(assignment_bridge_detail::Model owner,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const assignment_bridge_detail::Model>(std::move(owner))),bounds_(std::move(bounds)),maximum_(maximum) {}
    const assignment_bridge_detail::Model& native_owner() const & {return *owner_;}
    const assignment_bridge_detail::Model& native_owner() const && = delete;
    static std::shared_ptr<AssignmentModelBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace assignment_bridge_detail;
        return std::make_shared<AssignmentModelBridge>(learning::decode_relational_assignment_evidence(bytes(image),
            available_evidence_bridge_detail::authority(context),reader(bounds,maximum),limits(bounds),check.callback()),bounds,maximum);
    }
    static std::string root_payload(const AssignmentDeclaration& d,std::shared_ptr<FiniteSourceBridge> finite,
            const std::vector<AssignmentCandidate>& values,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace assignment_bridge_detail;
        if(!finite) throw std::invalid_argument("actual assignment finite source required");
        return bounded(learning::relational_assignment_root_payload(declaration(d),finite->native_owner(),candidates(values,bounds),limits(bounds)),maximum);
    }
    static std::string measurement_payload(const AssignmentObservation& value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return program_bridge_detail::bounded(rfc::learning::relational_assignment_measurement_payload(
            assignment_bridge_detail::observation(value,bounds)),maximum);
    }
    static std::vector<std::shared_ptr<ValueOriginBridge>> input_origins(std::shared_ptr<AvailableResultBridge> input,
            std::shared_ptr<AvailabilityContextBridge> context,std::size_t records,std::size_t maximum,GraphCheck check) {
        if(!input)throw std::invalid_argument("actual original assignment input required");
        const auto& value=input->native_owner();value.admit(available_evidence_bridge_detail::authority(context),check.callback());
        evidence_bridge_detail::bound(value.obligations().size(),records);
        std::vector<std::shared_ptr<ValueOriginBridge>> out;
        for(const auto& origin:value.obligations())out.push_back(ValueOriginBridge::adopt(origin,maximum));return out;
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        owner_->admit(available_evidence_bridge_detail::authority(context),check.callback());
    }
    bool same_owner(const AssignmentModelBridge& other) const {return owner_->same_owner(*other.owner_);}
    AssignmentSummary summary(std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);const auto& d=owner_->declaration();evidence_bridge_detail::bound(d.parameters.size(),maximum);
        const auto& c=context->native_owner();const auto identified=owner_->identified_candidate(c);
        return {{d.key,d.revision,d.origin,d.parameter_carrier,d.parameters},owner_->revision(),owner_->candidates().size(),
            owner_->observations().size(),owner_->remainder().size(),owner_->retained_entries(),identified.value_or(0),
            static_cast<unsigned>(owner_->status(c)),identified.has_value(),owner_->parent().has_value()};
    }
    std::shared_ptr<AssignmentModelBridge> parent(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto p=owner_->parent();return p?std::make_shared<AssignmentModelBridge>(*p,bounds_,maximum_):nullptr;
    }
    std::string archive(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        return program_bridge_detail::text(rfc::learning::encode_relational_assignment_evidence(*owner_,
            available_evidence_bridge_detail::authority(context),maximum_,check.callback()));
    }
    std::size_t stream(TaskSink sink,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        if(!sink.write) throw std::invalid_argument("assignment packet sink required");
        std::size_t count{};rfc::learning::stream_relational_assignment_evidence(*owner_,
            available_evidence_bridge_detail::authority(context),[&](std::span<const std::byte> part) {
                if(!sink.write(sink.context,reinterpret_cast<const char*>(part.data()),part.size())) throw std::runtime_error("assignment packet sink failed");
                count=rfc::machine::checked_size_add(count,part.size());
            },maximum_,check.callback());return count;
    }
    std::shared_ptr<FiniteSourceBridge> finite_source(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return FiniteSourceBridge::adopt(owner_->finite_source(),bounds_,maximum_);
    }
    AssignmentCandidate candidate(std::size_t index,std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);const auto& c=owner_->candidates().at(index);std::size_t used=c.assignment.size()+c.shift.size();
        evidence_bridge_detail::bound(used,maximum);AssignmentCandidate out;out.key=c.key;out.assignment=c.assignment;
        out.binding=assignment_bridge_detail::binding_info(c.binding,used,maximum);
        out.evidence=AvailableEvidenceBridge::adopt(c.evidence,bounds_,maximum_);
        for(const auto& q:c.shift) out.shift.push_back(q.get_str(16));
        out.injection=assignment_bridge_detail::matrix_info(c.injection,used,maximum);return out;
    }
    AssignmentBranchInfo branch(std::size_t index,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto& b=owner_->branch(index,context->native_owner());
        return {b.rejected,b.solves.size(),assignment_bridge_detail::coordinates(b.coordinates,maximum_)};
    }
    AssignmentObservation observation(std::size_t index,std::shared_ptr<AvailabilityContextBridge> context,
            std::size_t maximum,GraphCheck check) const {
        admit(context,check);const auto& value=owner_->observations().at(index);std::size_t used=value.inputs.size();
        evidence_bridge_detail::bound(used,maximum);AssignmentObservation out;out.key=value.key;out.origin=value.origin;
        out.binding=assignment_bridge_detail::binding_info(value.binding,used,maximum);
        out.measured=AvailableResultBridge::adopt(value.measured,bounds_,maximum_);
        for(const auto& row:value.inputs)out.inputs.push_back({row.candidate,AvailableResultBridge::adopt(row.input,bounds_,maximum_)});
        return out;
    }
    std::vector<std::vector<std::uint64_t>> remainder(std::size_t begin,std::size_t count,
            std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);const auto& rows=owner_->remainder();
        if(begin>rows.size()||count>rows.size()-begin) throw std::out_of_range("assignment remainder slice");
        std::size_t used=count;evidence_bridge_detail::bound(used,maximum);
        for(std::size_t i=begin;i<begin+count;++i){used=rfc::machine::checked_size_add(used,rows[i].size());evidence_bridge_detail::bound(used,maximum);}
        return {rows.begin()+static_cast<std::ptrdiff_t>(begin),rows.begin()+static_cast<std::ptrdiff_t>(begin+count)};
    }
    std::vector<std::shared_ptr<ValueOriginBridge>> origins(std::shared_ptr<AvailabilityContextBridge> context,
            std::size_t maximum,GraphCheck check) const {
        admit(context,check);const auto values=owner_->dependency_origins();evidence_bridge_detail::bound(values.size(),maximum);
        std::vector<std::shared_ptr<ValueOriginBridge>> out;for(const auto& origin:values) out.push_back(ValueOriginBridge::adopt(origin,maximum_));return out;
    }
    std::vector<std::string> dependencies(std::shared_ptr<AvailabilityContextBridge> context,std::size_t records,GraphCheck check) const {
        using namespace assignment_bridge_detail;
        admit(context,check);const auto required=owner_->dependency_origins();evidence_bridge_detail::bound(required.size(),records);
        auto out=admission_bridge_detail::dependencies(required,maximum_);
        append_dependencies(out,{InputUseAssetBridge::adopt(owner_->finite_source().asset(),bounds_,maximum_)->source_archive()},records);
        for(const auto& c:owner_->candidates()) {
            append_dependencies(out,AvailableEvidenceBridge::adopt(c.evidence,bounds_,maximum_)->dependency_archives(records),records);
            append_dependencies(out,{FieldProgramBridge::adopt(c.evidence.program().program(),bounds_,maximum_)->source().archive},records);
        }
        for(const auto& o:owner_->observations()) {
            append_dependencies(out,input_dependencies(o.measured,bounds_,maximum_,records),records);
            for(const auto& input:o.inputs)append_dependencies(out,input_dependencies(input.input,bounds_,maximum_,records),records);
        }
        return out;
    }
    AssignmentBatchInfo observation_dependencies(const std::vector<AssignmentObservation>& values,
            std::shared_ptr<AvailabilityContextBridge> context,std::size_t records,GraphCheck check) const {
        using namespace assignment_bridge_detail;
        admit(context,check);AssignmentBatchInfo out;const auto authority=available_evidence_bridge_detail::authority(context);
        for(const auto& row:values) {
            const auto o=assignment_bridge_detail::observation(row,bounds_);o.measured.admit(authority,check.callback());
            const auto raw=o.measured.result();
            const auto maps=learning::assignment_evidence_detail::admit_binding(o.binding,raw.result().program(),authority);
            const auto target=authority.require(o.origin,inference::RelationalDependencyRole::TargetValue,
                learning::relational_assignment_measurement_payload(o));
            out.origin_count=rfc::machine::checked_size_add(out.origin_count,2);
            append_dependencies(out.dependencies,admission_bridge_detail::dependencies(
                std::vector<rfc::inference::OwnedRelationalValueOrigin>{maps,target},maximum_),records);
            append_dependencies(out.dependencies,input_dependencies(o.measured,bounds_,maximum_,records),records);
            for(const auto& input:o.inputs) {
                input.input.admit(authority,check.callback());
                out.origin_count=rfc::machine::checked_size_add(out.origin_count,input.input.obligations().size());
                append_dependencies(out.dependencies,input_dependencies(input.input,bounds_,maximum_,records),records);
            }
            out.origin_count=rfc::machine::checked_size_add(out.origin_count,o.measured.obligations().size());
        }
        return out;
    }
    std::shared_ptr<AssignmentModelBridge> append(const std::vector<AssignmentObservation>& values,
            std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        std::vector<rfc::learning::RelationalAssignmentObservation> observations;
        for(const auto& v:values) observations.push_back(assignment_bridge_detail::observation(v,bounds_));
        return std::make_shared<AssignmentModelBridge>(owner_->append(std::move(observations),
            available_evidence_bridge_detail::authority(context),check.callback()),bounds_,maximum_);
    }
    std::shared_ptr<AvailableResultBridge> restore_input(std::size_t candidate,const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        return AvailableResultBridge::adopt(rfc::learning::restore_relational_assignment_input(*owner_,candidate,
            program_bridge_detail::bytes(image),available_evidence_bridge_detail::authority(context),
            program_bridge_detail::reader(bounds_,maximum_),check.callback()),bounds_,maximum_);
    }
    std::shared_ptr<AssignmentSolveBridge> solve(std::size_t candidate,std::size_t index,
            std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const;
    std::shared_ptr<AssignmentResponseBridge> execute(std::size_t candidate,std::shared_ptr<AvailableResultBridge> input,
            std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const;
private:
    std::shared_ptr<const assignment_bridge_detail::Model> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};

class AssignmentSolveBridge {
public:
    AssignmentSolveBridge(assignment_bridge_detail::Model model,std::shared_ptr<const rfc::learning::RelationalAssignmentSolve> solve,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :model_(std::move(model)),owner_(std::move(solve)),bounds_(std::move(bounds)),maximum_(maximum) {}
    bool same_owner(const AssignmentSolveBridge& other) const {return owner_==other.owner_;}
    std::uint64_t observation(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {admit(context,check);return owner_->observation;}
    TaskCertificate certificate(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return assignment_bridge_detail::certificate(owner_->certificate,maximum_);
    }
    AssignmentCoordinates coordinates(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return assignment_bridge_detail::coordinates(owner_->child,maximum_);
    }
    std::shared_ptr<AvailableAffineBridge> original(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return AvailableAffineBridge::adopt(owner_->response,bounds_,maximum_);
    }
private:
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {model_.admit(available_evidence_bridge_detail::authority(context),check.callback());}
    assignment_bridge_detail::Model model_;
    std::shared_ptr<const rfc::learning::RelationalAssignmentSolve> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
class AssignmentResponseBridge {
public:
    AssignmentResponseBridge(assignment_bridge_detail::Response owner,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum) {}
    static std::shared_ptr<AssignmentResponseBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace assignment_bridge_detail;
        return std::make_shared<AssignmentResponseBridge>(learning::decode_relational_assignment_response(bytes(image),
            available_evidence_bridge_detail::authority(context),reader(bounds,maximum),limits(bounds),check.callback()),bounds,maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {owner_.admit(available_evidence_bridge_detail::authority(context),check.callback());}
    std::shared_ptr<AssignmentModelBridge> model(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return std::make_shared<AssignmentModelBridge>(owner_.model(),bounds_,maximum_);
    }
    std::size_t candidate() const {return owner_.candidate();}
    std::vector<std::shared_ptr<ValueOriginBridge>> origins(std::shared_ptr<AvailabilityContextBridge> context,
            std::size_t maximum,GraphCheck check) const {
        admit(context,check);auto values=owner_.model().dependency_origins();std::map<std::uint64_t,rfc::inference::OwnedRelationalValueOrigin> unique;
        for(const auto& origin:values)unique.emplace(origin.key(),origin);
        const auto original=owner_.original();
        const auto append=[&](const auto& origins) {
            for(const auto& origin:origins) {
                const auto [at,inserted]=unique.emplace(origin.key(),origin);
                if(!inserted&&!at->second.same_owner(origin))throw std::invalid_argument("different actual assignment response origins share a key");
            }
        };
        const auto evidence=original.evidence();const auto input=original.input();
        append(evidence.obligations());append(input.obligations());
        evidence_bridge_detail::bound(unique.size(),maximum);std::vector<std::shared_ptr<ValueOriginBridge>> out;
        for(const auto& [key,origin]:unique){(void)key;out.push_back(ValueOriginBridge::adopt(origin,maximum_));}return out;
    }
    std::vector<std::string> dependencies(std::shared_ptr<AvailabilityContextBridge> context,std::size_t records,GraphCheck check) const {
        admit(context,check);AssignmentModelBridge model(owner_.model(),bounds_,maximum_);
        auto out=model.dependencies(context,records,check);
        assignment_bridge_detail::append_dependencies(out,AvailableAffineBridge::adopt(owner_.original(),bounds_,maximum_)->dependency_archives(records),records);
        return out;
    }
    AssignmentCoordinates coordinates(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return assignment_bridge_detail::coordinates(owner_.coordinates(),maximum_);
    }
    std::shared_ptr<AvailableAffineBridge> original(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return AvailableAffineBridge::adopt(owner_.original(),bounds_,maximum_);
    }
    AffineFieldInfo field(std::uint64_t node,bool occurrences,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto& authority=context->native_owner();
        auto out=evidence_bridge_detail::field_info(occurrences?owner_.occurrence_field(node,authority):owner_.field(node,authority));
        if(out.particular.size()>maximum_||out.directions.size()>maximum_) throw std::length_error("conditioned field byte bound");return out;
    }
    AssignmentMapInfo map(std::size_t grade,std::size_t flag,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto value=owner_.coefficient_independent_map(program_bridge_detail::cell(grade,0).grade,flag,context->native_owner());
        AssignmentMapInfo out;out.particular=program_bridge_detail::matrix_bytes(value.particular);
        if(out.particular.size()>maximum_) throw std::length_error("conditioned map byte bound");
        for(const auto& d:value.directions) {auto b=program_bridge_detail::matrix_bytes(d);if(b.size()>maximum_)throw std::length_error("conditioned map byte bound");out.directions.push_back(std::move(b));}return out;
    }
    TaskCertificate postcondition(std::shared_ptr<TaskOutcomeBridge> goal,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);if(!goal)throw std::invalid_argument("actual assignment goal required");
        return assignment_bridge_detail::certificate(owner_.postcondition(goal->native_owner(),context->native_owner()),maximum_);
    }
    std::shared_ptr<TaskOutcomeBridge> restore_goal(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        return TaskOutcomeBridge::adopt(rfc::learning::restore_relational_assignment_goal(owner_,program_bridge_detail::bytes(image),
            available_evidence_bridge_detail::authority(context),program_bridge_detail::reader(bounds_,maximum_),check.callback()),bounds_,maximum_);
    }
    std::string archive(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        return program_bridge_detail::text(rfc::learning::encode_relational_assignment_response(owner_,
            available_evidence_bridge_detail::authority(context),maximum_,check.callback()));
    }
    std::size_t stream(TaskSink sink,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        if(!sink.write)throw std::invalid_argument("assignment packet sink required");
        std::size_t count{};rfc::learning::stream_relational_assignment_response(owner_,available_evidence_bridge_detail::authority(context),
            [&](std::span<const std::byte> part) {
                if(!sink.write(sink.context,reinterpret_cast<const char*>(part.data()),part.size()))throw std::runtime_error("assignment packet sink failed");
                count=rfc::machine::checked_size_add(count,part.size());
            },maximum_,check.callback());return count;
    }
private:
    assignment_bridge_detail::Response owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
inline std::shared_ptr<AssignmentSolveBridge> AssignmentModelBridge::solve(std::size_t candidate,std::size_t index,
        std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
    admit(context,check);return std::make_shared<AssignmentSolveBridge>(*owner_,owner_->branch(candidate,context->native_owner()).solves.at(index),bounds_,maximum_);
}
inline std::shared_ptr<AssignmentResponseBridge> AssignmentModelBridge::execute(std::size_t candidate,std::shared_ptr<AvailableResultBridge> input,
        std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
    if(!input)throw std::invalid_argument("actual assignment input owner required");
    return std::make_shared<AssignmentResponseBridge>(owner_->execute(candidate,input->native_owner(),
        available_evidence_bridge_detail::authority(context),check.callback()),bounds_,maximum_);
}
} // namespace rexgraph::python
