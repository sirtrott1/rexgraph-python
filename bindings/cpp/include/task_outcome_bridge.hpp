#pragma once

#include "field_evidence_admission_bridge.hpp"
#include "known_tensor_graph_bridge.hpp"
#include <rfc/learning/relational_task_outcome_codec.hpp>

namespace rexgraph::python {
struct TaskDeclaration {
    std::uint64_t key{}, revision{}, use{}, origin{}, basis{};
    std::vector<std::uint64_t> directions;
    ProgramMatrix metric;
    std::string target;
};
struct TaskBinding {
    std::uint64_t key{}, revision{}, worker{}, machine{}, operation{}, origin{};
    InterfacePatch interface;
    InterfaceView view;
};
struct TaskCertificate {
    unsigned status{};
    std::size_t rank{};
    std::string residual, directions, feasible_shift, feasible_kernel, contradiction;
};
struct TaskSink {
    int (*write)(void*, const char*, std::size_t) = nullptr;
    void* context{};
};
struct TaskDetails {
    std::uint64_t interface_key{}, branch{}, view{};
    std::vector<std::uint64_t> directions;
    std::vector<InterfaceNodeCell> cells;
    std::string target, metric, accession, reconstruction;
};
namespace task_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
using Owner = learning::OwnedRelationalTaskOutcome;
inline learning::RelationalTaskResponse response(const std::shared_ptr<AvailableResultBridge>& field,
        const std::shared_ptr<AvailableAffineBridge>& family) {
    if (bool(field) == bool(family)) throw std::invalid_argument("one actual available execution or affine owner required");
    if (field) return field->native_owner();
    return family->native_owner();
}
inline learning::RelationalTaskPostconditionDeclaration task(const TaskDeclaration& d,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
    numeric::ExactActionGuard guard(limits(bounds));
    return {d.key,d.revision,d.use,d.origin,d.basis,d.directions,matrix(d.metric,guard),
            interface_bridge_detail::field(d.target,bounds,maximum)};
}
inline learning::RelationalWorkerBindingDeclaration binding(const TaskBinding& d,
        const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds));
    inference::RelationalFieldInterfaceDeclaration patch{d.interface.key,d.interface.branch,{}};
    for (const auto& c:d.interface.cells) patch.cells.push_back({c.node,cell(c.grade,c.cell)});
    return {d.key,d.revision,d.worker,d.machine,d.operation,d.origin,std::move(patch),
            interface_bridge_detail::view(d.view,guard)};
}
inline std::string vector_bytes(const numeric::RationalVector& v) {
    storage::temporal_codec_detail::Writer w;
    learning::relational_affine_codec_detail::vector(w,v);
    return text(w.bytes());
}
}
class TaskOutcomeBridge {
public:
    TaskOutcomeBridge(const TaskDeclaration& task, std::shared_ptr<InputUseAssetBridge> asset,
            const TaskBinding& binding, std::shared_ptr<AvailableResultBridge> field,
            std::shared_ptr<AvailableAffineBridge> family, std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds, std::size_t maximum, GraphCheck check)
        :bounds_(bounds),maximum_(maximum) {
        using namespace task_bridge_detail;
        if (!asset) throw std::invalid_argument("actual original task asset required");
        const auto value=response(field,family);const auto authority=available_evidence_bridge_detail::authority(context);
        learning::task_outcome_detail::admit(value,authority,check.callback());
        owner_=std::make_shared<const Owner>(task_bridge_detail::task(task,bounds,maximum),asset->native_owner(),
            task_bridge_detail::binding(binding,bounds),value,authority,limits(bounds),check.callback());
    }
    static std::shared_ptr<TaskOutcomeBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,
            std::size_t maximum,GraphCheck check) {
        using namespace task_bridge_detail;
        const auto authority=available_evidence_bridge_detail::authority(context);
        auto value=learning::decode_relational_task_outcome(bytes(image),authority,reader(bounds,maximum),limits(bounds),check.callback());
        return std::shared_ptr<TaskOutcomeBridge>(new TaskOutcomeBridge(std::move(value),bounds,maximum));
    }
    static std::string goal_payload(const TaskDeclaration& task,std::shared_ptr<InputUseAssetBridge> asset,
            std::shared_ptr<AvailableResultBridge> field,std::shared_ptr<AvailableAffineBridge> family,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace task_bridge_detail;
        if (!asset) throw std::invalid_argument("actual original task asset required");
        return bounded(learning::relational_task_goal_payload(task_bridge_detail::task(task,bounds,maximum),
            asset->native_owner(),response(field,family)),maximum);
    }
    static std::string binding_payload(const TaskBinding& binding,std::shared_ptr<AvailableResultBridge> field,
            std::shared_ptr<AvailableAffineBridge> family,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace task_bridge_detail;
        return bounded(learning::relational_worker_binding_payload(task_bridge_detail::binding(binding,bounds),response(field,family)),maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        owner_->admit(available_evidence_bridge_detail::authority(context),check.callback());
    }
    unsigned status(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return static_cast<unsigned>(owner_->status(context->native_owner()));
    }
    bool executed() const {return owner_->executed_response();}
    std::string archive(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        return program_bridge_detail::text(rfc::learning::encode_relational_task_outcome(*owner_,
            available_evidence_bridge_detail::authority(context),maximum_,check.callback()));
    }
    std::size_t stream(TaskSink sink,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        if (!sink.write) throw std::invalid_argument("synchronous task packet sink required");
        std::size_t count{};
        rfc::learning::stream_relational_task_outcome(*owner_,available_evidence_bridge_detail::authority(context),
            [&](std::span<const std::byte> chunk) {
                if (!sink.write(sink.context,reinterpret_cast<const char*>(chunk.data()),chunk.size()))
                    throw std::runtime_error("task packet sink failed");
                count=rfc::machine::checked_size_add(count,chunk.size());
            },maximum_,check.callback());
        return count;
    }
    TaskCertificate certificate(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        using namespace task_bridge_detail;
        admit(context,check);const auto p=owner_->certificate(context->native_owner());
        TaskCertificate out{static_cast<unsigned>(p.status),p.rank,tensor_bytes(p.residual.particular),
            matrix_bytes(p.residual.directions),vector_bytes(p.feasible_shift),matrix_bytes(p.feasible_kernel),vector_bytes(p.contradiction)};
        for (const auto* image:{&out.residual,&out.directions,&out.feasible_shift,&out.feasible_kernel,&out.contradiction})
            if (image->size()>maximum_) throw std::length_error("complete task certificate byte bound");
        return out;
    }
    std::vector<std::uint64_t> identity(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto& t=owner_->task();const auto& b=owner_->binding();
        return {t.key,t.revision,t.use,t.origin,t.basis,b.key,b.revision,b.worker,b.machine,b.operation,b.origin};
    }
    std::shared_ptr<InputUseAssetBridge> asset(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return InputUseAssetBridge::adopt(owner_->contract_asset(),bounds_,maximum_);
    }
    std::shared_ptr<AvailableResultBridge> field(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto value=owner_->response();
        const auto* result=std::get_if<rfc::inference::AvailableRelationalFieldResult>(&value);
        return result?AvailableResultBridge::adopt(*result,bounds_,maximum_):nullptr;
    }
    std::shared_ptr<AvailableAffineBridge> family(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto value=owner_->response();
        const auto* result=std::get_if<rfc::learning::AvailableRelationalAffineResult>(&value);
        return result?AvailableAffineBridge::adopt(*result,bounds_,maximum_):nullptr;
    }
    ProgramSourceInfo source(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return program_bridge_detail::source_info(rfc::learning::task_outcome_detail::program(owner_->response()));
    }
    TaskDetails details(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        using namespace program_bridge_detail;
        admit(context,check);const auto& t=owner_->task();const auto& b=owner_->binding();
        TaskDetails out{b.interface.key,b.interface.branch,b.view.key,t.directions,{},tensor_bytes(t.target),
            matrix_bytes(t.metric),matrix_bytes(b.view.accession),matrix_bytes(b.view.reconstruction)};
        for (const auto& c:b.interface.cells) out.cells.push_back({c.node,c.cell.grade.value,c.cell.index});
        for (const auto* image:{&out.target,&out.metric,&out.accession,&out.reconstruction})
            if (image->size()>maximum_) throw std::length_error("complete task definition byte bound");
        return out;
    }
    std::vector<std::string> dependencies(std::shared_ptr<AvailabilityContextBridge> context,
            std::size_t maximum,GraphCheck check) const {
        admit(context,check);std::vector<std::string> out;
        const auto fixed=field(context,check);
        if (fixed) out=fixed->dependency_archives(maximum);
        else out=family(context,check)->dependency_archives(maximum);
        out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->contract_asset().history()),maximum_));
        const auto input=rfc::learning::task_outcome_detail::input(owner_->response()).result();
        for (const auto& value:input.inputs()) {
            evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(),1),maximum);
            out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(value.accession().asset().history()),maximum_));
        }
        const auto original=origins(context,maximum,check);
        for (const auto& value:original) {
            evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(),1),maximum);
            out.push_back(value->source_archive());
        }
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
    std::vector<std::shared_ptr<ValueOriginBridge>> origins(std::shared_ptr<AvailabilityContextBridge> context,
            std::size_t maximum,GraphCheck check) const {
        using namespace rfc;
        admit(context,check);const auto response=owner_->response();
        std::vector<inference::OwnedRelationalValueOrigin> original;
        std::visit([&](const auto& value) {
            if constexpr(std::is_same_v<std::decay_t<decltype(value)>,learning::AvailableRelationalAffineResult>) {
                const auto e=value.evidence();original=e.obligations();
                const auto in=value.input();original.insert(original.end(),in.obligations().begin(),in.obligations().end());
            } else original=value.obligations();
        },response);
        const auto& t=owner_->task();const auto& b=owner_->binding();const auto& authority=context->native_owner();
        original.push_back(authority.require(t.origin,inference::RelationalDependencyRole::Prospective,
            learning::relational_task_goal_payload(t,owner_->contract_asset(),response)));
        original.push_back(authority.require(b.origin,inference::RelationalDependencyRole::Maps,
            learning::relational_worker_binding_payload(b,response)));
        evidence_bridge_detail::bound(original.size(),maximum);
        std::vector<std::shared_ptr<ValueOriginBridge>> out;
        for (const auto& origin:original) out.push_back(ValueOriginBridge::adopt(origin,maximum_));
        return out;
    }
private:
    TaskOutcomeBridge(rfc::learning::OwnedRelationalTaskOutcome value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::OwnedRelationalTaskOutcome>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum){}
    std::shared_ptr<const rfc::learning::OwnedRelationalTaskOutcome> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
}
