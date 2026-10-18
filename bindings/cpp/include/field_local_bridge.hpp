#pragma once

#include "field_extension_bridge.hpp"
#include <rfc/learning/relational_local_solve_work.hpp>
#include <atomic>

namespace rexgraph::python {
struct LocalDeclaration {
    std::uint64_t key{}, revision{}, coordinate_carrier{}, coordinate_basis{};
    std::size_t grade{};
    unsigned kind{};
    std::string lambda;
    bool has_frame{};
    ProgramMatrix frame;
    std::vector<std::size_t> interior;
};
struct LocalReadout { std::vector<std::vector<std::size_t>> patch; InterfaceView view; };
struct LocalInfo {
    std::uint64_t key{}, revision{}, node{}, coordinate_carrier{}, coordinate_basis{};
    std::size_t grade{};
    unsigned kind{};
    std::string lambda;
    std::vector<std::size_t> interior, interface, layout;
    std::vector<std::vector<std::size_t>> patch;
    std::uint64_t readout_key{}, readout_basis{};
    std::vector<std::uint64_t> readout_directions;
    std::vector<std::vector<std::string>> matrices;
};
namespace local_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline inference::RelationalFieldSolveDeclaration declaration(const LocalDeclaration& d, const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds));
    inference::RelationalFieldSolveDeclaration out;
    out.key=d.key; out.revision=d.revision; out.coordinate_carrier=d.coordinate_carrier; out.coordinate_basis=d.coordinate_basis;
    out.grade=cell(d.grade,0).grade; out.kind=static_cast<inference::RelationalFieldSolveKind>(d.kind);
    out.lambda=available_evidence_bridge_detail::free_values({d.lambda},bounds).at(0);
    if(d.has_frame) out.harmonic_frame=matrix(d.frame,guard);
    out.interior=d.interior; guard.dimension(out.interior.size()); return out;
}
inline learning::RelationalSolveReadoutDeclaration readout(const LocalReadout& d, const std::vector<std::size_t>& bounds) {
    numeric::ExactActionGuard guard(limits(bounds)); learning::RelationalSolveReadoutDeclaration out;
    guard.dimension(d.patch.size());
    for(const auto& row:d.patch) { guard.dimension(row.size()); std::vector<std::uint32_t> cells;
        for(const auto i:row) { guard.step(); cells.push_back(cell(0,i).index); } out.patch.push_back(std::move(cells)); }
    out.view=interface_bridge_detail::view(d.view,guard); return out;
}
inline learning::RelationalSolveOrigins origins(const std::vector<std::uint64_t>& values) {
    if(values.size()!=4) throw std::invalid_argument("four local origin identities required");
    return {values[0],values[1],values[2],values[3]};
}
inline learning::AvailableRelationalSolveRhs rhs(std::shared_ptr<AvailableResultBridge> fixed,
        std::shared_ptr<AvailableEvaluationBridge> selected,std::shared_ptr<ExtensionEvaluationBridge> extended) {
    if(unsigned(bool(fixed))+unsigned(bool(selected))+unsigned(bool(extended))!=1u) throw std::invalid_argument("one actual fixed or selected RHS required");
    if(fixed) return fixed->native_owner();
    if(selected) return selected->native_owner();
    return extended->native_owner();
}
} // namespace local_bridge_detail

class LocalResultBridge {
public:
    static std::shared_ptr<LocalResultBridge> adopt(rfc::learning::AvailableRelationalLocalResult value,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<LocalResultBridge>(new LocalResultBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<LocalResultBridge> restore(const std::string& packet,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace local_bridge_detail;
        return adopt(learning::decode_available_relational_local_result(bytes(packet),available_evidence_bridge_detail::authority(context),
            reader(bounds,maximum),limits(bounds)),bounds,maximum);
    }
    static std::vector<std::string> payloads(std::shared_ptr<AvailableResultBridge> fixed,std::shared_ptr<AvailableEvaluationBridge> selected,std::shared_ptr<ExtensionEvaluationBridge> extended,
            std::uint64_t node,const LocalDeclaration& d,const LocalReadout& read,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace local_bridge_detail; const auto input=rhs(fixed,selected,extended); const auto native=declaration(d,bounds);
        return {bounded(learning::relational_local_problem_payload(input,node,native),maximum),
            bounded(learning::relational_local_readout_payload(input,node,native,readout(read,bounds)),maximum)};
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::learning::encode_available_relational_local_result(*owner_),maximum_);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const { owner_->admit(available_evidence_bridge_detail::authority(context)); }
    std::string field(const std::string& name) const {
        using namespace program_bridge_detail; const auto vector_bytes=[&](const rfc::numeric::RationalVector& v) { rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_)); return tensor_bytes(rfc::inference::field_interface_detail::tensor(v,guard)); };
        const auto result=owner_->result(); std::string out;
        if(name=="global") out=tensor_bytes(result.field());
        else if(name=="patch") out=tensor_bytes(owner_->patch_field());
        else if(name=="visible") out=tensor_bytes(owner_->visible());
        else if(name=="reconstructed") out=tensor_bytes(owner_->reconstructed());
        else if(name=="augmented_rhs") out=vector_bytes(result.augmented_rhs());
        else if(name=="augmented_response") out=vector_bytes(result.reconstructed());
        else if(name=="interface_rhs") out=vector_bytes(result.interface_rhs());
        else if(name=="interface_response") out=vector_bytes(result.interface_response());
        else if(name=="residual") out=vector_bytes(result.residual());
        else throw std::invalid_argument("unknown local whole field");
        if(out.size()>maximum_) throw std::length_error("local field byte admission");
        return out;
    }
    LocalInfo info() const {
        using namespace program_bridge_detail; const auto parent=owner_->parent(); const auto problem=parent.problem();
        const auto& d=parent.declaration(); const auto& op=problem.operators(); const auto& l=op.layout;
        LocalInfo out; out.key=d.key;out.revision=d.revision;out.node=parent.node();out.grade=d.grade.value;
        out.coordinate_carrier=d.coordinate_carrier;out.coordinate_basis=d.coordinate_basis;out.kind=static_cast<unsigned>(d.kind);
        out.lambda=d.lambda.get_str();out.interior=problem.partition().interior();out.interface=problem.partition().interface();
        out.layout={l.a,l.b,l.c,l.d,l.f,l.g,l.e,l.t,l.x,l.size};
        for(const auto& cells:parent.readout().patch) out.patch.emplace_back(cells.begin(),cells.end());
        out.readout_key=parent.readout().view.key;out.readout_basis=parent.readout().view.basis;
        out.readout_directions=parent.readout().view.directions;
        const auto add=[&](const std::string& name,const rfc::numeric::ExactMatrix& m) {
            auto value=matrix_bytes(m); if(value.size()>maximum_) throw std::length_error("local matrix byte admission");
            out.matrices.push_back({name,std::move(value)}); };
        add("lower_boundary",op.lower);add("upper_boundary",op.upper);add("metric",op.metric);
        add("below_metric",op.below_metric);add("above_metric",op.above_metric);add("harmonic_frame",op.frame);
        add("augmented_operator",op.augmented);add("schur",problem.partition().schur());
        add("accession",parent.readout().view.accession);add("reconstruction",parent.readout().view.reconstruction);
        add("readout_metric",parent.readout_metric());if(parent.readout().view.physical) add("physical",*parent.readout().view.physical);
        return out;
    }
    std::vector<ProgramSelected> select(std::uint64_t node,const std::vector<std::vector<std::size_t>>& cells,std::size_t maximum) const {
        using namespace rfc;using namespace program_bridge_detail;
        const auto parent=owner_->parent();if(node!=parent.node()) throw std::invalid_argument("local response requires its declared RHS node");
        const auto program=parent.problem().program();const auto grade=parent.declaration().grade;
        const auto offsets=program.fibers().fibers().stalks().offsets(grade);
        numeric::ExactActionGuard guard(limits(bounds_));std::set<std::uint32_t> seen;std::size_t count{};
        for(const auto& row:cells) {
            if(row.size()!=2) throw std::invalid_argument("local stalk requires grade and original cell index");
            const auto original=cell(row[0],row[1]);
            if(original.grade!=grade || original.index+std::size_t{1}>=offsets.size() || !seen.insert(original.index).second)
                throw std::invalid_argument("local stalks must be distinct in response grade");
            count=machine::checked_size_add(count,offsets[original.index+1u]-offsets[original.index]);
        }
        if(count>maximum) throw std::length_error("bounded local stalk export");
        guard.entries(count);
        const auto response=owner_->result();std::vector<ProgramSelected> out;
        for(const auto& row:cells) {
            const auto original=cell(row[0],row[1]);const auto& basis=inference::field_interface_detail::stalk(program,original);
            numeric::RationalVector values;ProgramSelected value;value.stalk=stalk_info(basis);
            for(auto i=offsets[original.index];i<offsets[original.index+1u];++i) {
                guard.step();const auto q=response.field().values().coefficient(i);guard.value(q);values.push_back(q);value.values.push_back(q.get_str(16));
            }
            value.field=tensor_bytes(inference::field_interface_detail::tensor(values,guard));out.push_back(std::move(value));
        }
        return out;
    }
    std::shared_ptr<AvailableResultBridge> fixed_rhs() const {
        const auto parent=owner_->parent();const auto& rhs=parent.rhs();
        const auto p=std::get_if<rfc::inference::AvailableRelationalFieldResult>(&rhs);
        return p ? AvailableResultBridge::adopt(*p,bounds_,maximum_) : nullptr;
    }
    std::shared_ptr<AvailableEvaluationBridge> selected_rhs() const {
        const auto parent=owner_->parent();const auto& rhs=parent.rhs();
        const auto p=std::get_if<rfc::learning::AvailableRelationalAffineEvaluation>(&rhs);
        return p ? AvailableEvaluationBridge::adopt(*p,bounds_,maximum_) : nullptr;
    }
    std::shared_ptr<ExtensionEvaluationBridge> extension_rhs() const {
        const auto parent=owner_->parent();const auto rhs=parent.rhs();
        const auto p=std::get_if<rfc::learning::AvailableRelationalExtensionEvaluation>(&rhs);
        return p?ExtensionEvaluationBridge::adopt(*p,bounds_,maximum_):nullptr;
    }
    ProgramSourceInfo source() const {
        const auto parent=owner_->parent(); const auto actual=rfc::learning::available_local_detail::rhs_program(parent.rhs());
        return FieldProgramBridge::adopt(actual,bounds_,maximum_)->source();
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto fixed=fixed_rhs();auto selected=selected_rhs();auto extended=extension_rhs();
        auto out=fixed?fixed->dependency_archives(maximum):selected?selected->dependency_archives(maximum):extended->dependency_archives(maximum);
        const auto parent=owner_->parent(); evidence_bridge_detail::bound(rfc::machine::checked_size_add(out.size(),parent.obligations().size()),maximum);
        for(const auto& origin:parent.obligations()) out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(origin.publication().history()),maximum_));
        return out;
    }
private:
    LocalResultBridge(rfc::learning::AvailableRelationalLocalResult value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::AvailableRelationalLocalResult>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::AvailableRelationalLocalResult> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};

class LocalWorkBridge {
    struct PhaseState {
        std::mutex mutex;std::condition_variable changed;int current{-1},pause{-1};bool released{};
        void enter(rfc::learning::RelationalLocalWorkPhase value) {
            std::unique_lock lock(mutex);current=static_cast<int>(value);changed.notify_all();
            if(current==pause) changed.wait(lock,[&]{return released;});
        }
        void resume() { std::lock_guard lock(mutex);released=true;changed.notify_all(); }
    };
public:
    static std::shared_ptr<LocalWorkBridge> start(std::shared_ptr<AvailableResultBridge> fixed,std::shared_ptr<AvailableEvaluationBridge> selected,std::shared_ptr<ExtensionEvaluationBridge> extended,
            std::uint64_t node,const LocalDeclaration& d,const LocalReadout& read,const std::vector<std::uint64_t>& keys,
            std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum,
            const std::vector<std::size_t>& work,int pause) {
        using namespace local_bridge_detail;
        auto out=std::shared_ptr<LocalWorkBridge>(new LocalWorkBridge(bounds,maximum,work,pause));const auto state=out->phase_;
        out->submit(learning::relational_local_solve_job(out->key_,out->demand_,rhs(fixed,selected,extended),node,declaration(d,bounds),
            readout(read,bounds),origins(keys),available_evidence_bridge_detail::authority(context),limits(bounds),{},
            [state](auto phase){state->enter(phase);}));return out;
    }
    static std::shared_ptr<LocalWorkBridge> replay(const std::string& packet,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,const std::vector<std::size_t>& work,int pause) {
        using namespace local_bridge_detail;auto out=std::shared_ptr<LocalWorkBridge>(new LocalWorkBridge(bounds,maximum,work,pause));const auto state=out->phase_;
        out->submit(learning::relational_local_replay_job(out->key_,out->demand_,std::vector<std::byte>(bytes(packet).begin(),bytes(packet).end()),available_evidence_bridge_detail::authority(context),
            reader(bounds,maximum),limits(bounds),{},[state](auto p){state->enter(p);}));return out;
    }
    ~LocalWorkBridge() { try { cancel(); } catch(...) {} }
    void resume() const { phase_->resume(); }
    int phase() const { std::lock_guard lock(phase_->mutex);return phase_->current; }
    void cancel() const { if(run_) { run_->cancel();phase_->resume();join(); } }
    std::shared_ptr<LocalResultBridge> wait(std::shared_ptr<AvailabilityContextBridge> context) const {
        join();const auto result=run_->completion(key_).result<rfc::learning::AvailableRelationalLocalResult>();
        result->admit(available_evidence_bridge_detail::authority(context));return LocalResultBridge::adopt(*result,bounds_,maximum_);
    }
    std::vector<std::uint64_t> receipt() const {
        const auto r=run_->completion(key_).receipt();return {r.key.id,r.key.generation,static_cast<std::uint64_t>(r.status),
            r.queue_ns,r.execution_ns,static_cast<std::uint64_t>(joined_.load())};
    }
private:
    LocalWorkBridge(std::vector<std::size_t> bounds,std::size_t maximum,const std::vector<std::size_t>& work,int pause)
        :bounds_(std::move(bounds)),maximum_(maximum),phase_(std::make_shared<PhaseState>()) {
        if(work.size()!=5 || pause < -1 || pause>2) throw std::invalid_argument("local work declaration");
        key_={work[0],work[1]};demand_={work[2],work[3],work[4]};phase_->pause=pause;
        resources_=std::make_unique<rfc::runtime::CpuResourceCoordinator>(rfc::runtime::detect_hardware(),
            rfc::runtime::CpuResourcePolicy{demand_.workers,demand_.buffer_bytes,1000000});
    }
    void submit(rfc::runtime::WorkJob job) {
        graph_=std::make_unique<rfc::runtime::WorkGraph>(std::vector<rfc::runtime::WorkJob>{std::move(job)},
            rfc::runtime::WorkGraphLimits{1,0,1,demand_.result_bytes,1});
        run_=std::make_unique<rfc::runtime::WorkRun>(graph_->submit(*resources_));
    }
    void join() const { std::lock_guard lock(join_mutex_);if(!joined_) { run_->wait();joined_=true; } }
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
    std::shared_ptr<PhaseState> phase_;rfc::runtime::WorkKey key_;rfc::runtime::WorkDemand demand_;
    std::unique_ptr<rfc::runtime::CpuResourceCoordinator> resources_;
    std::unique_ptr<rfc::runtime::WorkGraph> graph_;std::unique_ptr<rfc::runtime::WorkRun> run_;
    mutable std::mutex join_mutex_;mutable std::atomic<bool> joined_{};
};
} // namespace rexgraph::python
