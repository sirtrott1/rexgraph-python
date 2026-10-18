#include "bridge.hpp"
#include "rfc/inference/native_field_execution.hpp"
#include "rfc/learning/integer_network.hpp"
#include "rfc/learning/native_exact_trainer.hpp"
#include "rfc/learning/integer_perceptron.hpp"
#include "rfc/runtime/cpu_tiles.hpp"
#include "rfc/core/builder.hpp"
#include "rfc/algebra/morphism.hpp"
#include "rfc/storage/native_series_codec.hpp"
#include "rfc/storage/native_series_prefix.hpp"
#include <atomic>
#include <algorithm>
#include <charconv>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace rexgraph::python {
namespace {
using namespace rfc;
storage::NativeAccessionRequest tensor_request(storage::NativeAccessionKind kind, std::uint64_t key) {
    storage::NativeAccessionRequest request; request.kind=kind; request.subject=key; return request;
}
std::string bytes_string(const std::vector<std::byte>& value) {
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::vector<std::string> rationals(const numeric::RationalVector& values) {
    std::vector<std::string> result; result.reserve(values.size());
    for(const auto& value:values) result.push_back(value.get_str(16));
    return result;
}
runtime::ExecutionHostPolicy policy_for(const std::vector<std::size_t>& limits) {
    if(limits.size()!=5) throw std::invalid_argument("execution policy requires five resource limits");
    runtime::ExecutionHostPolicy policy; policy.limits={limits[0],limits[1],limits[2],limits[3],limits[4]}; return policy;
}
numeric::Rational coefficient(const std::string& value) {
    numeric::Rational out;
    if(out.set_str(value,16)!=0 || out.get_den()==0) throw std::invalid_argument("invalid exact coefficient");
    out.canonicalize(); return out;
}
std::int64_t integer_parameter(const numeric::Rational& value) {
    if(value.get_den()!=1)throw std::invalid_argument("integer parameters require integral coefficients");
    const auto text=value.get_num().get_str();
    std::int64_t result{};
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),result);
    if(parsed.ec==std::errc::result_out_of_range)throw std::overflow_error("integer parameter exceeds int64");
    if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())throw std::invalid_argument("invalid integer parameter");
    return result;
}
void failure(ExecutionResult& out,const runtime::ExecutionCancelled& error) {
    out.error=1; out.error_message=error.what();
}
void failure(ExecutionResult& out,const runtime::ExecutionLimitExceeded& error) {
    out.error=2; out.error_message=error.what(); out.error_resource=error.resource();
    out.error_required=error.required(); out.error_limit=error.limit();
}
std::vector<learning::ExactMetricChannel> metric_channels(const std::vector<unsigned>& kinds) {
    std::vector<learning::ExactMetricChannel> out;
    for(const auto kind:kinds) out.push_back({static_cast<learning::ExactMetricChannelKind>(kind),numeric::Rational{0}});
    return out;
}
template<class Requirements>
field::PreparedExactExecutionTower::BatchWorkspace& select_batch_workspace(
    const inference::PreparedNativeExactTower& tower, core::Grade grade, std::size_t samples,
    std::unique_ptr<field::PreparedExactExecutionTower::BatchWorkspace>& workspace,
    const runtime::ExecutionLease& lease, Requirements requirements) {
    using inference::native_field_detail::add;
    lease.check_cancelled();
    const bool grow = !workspace || workspace->grade() != grade || samples > workspace->capacity();
    const auto retained = workspace ? workspace->retained_slots() : 0;
    const auto candidate = grow ? tower.tower().batch_workspace_slots(grade, samples) : 0;
    const auto required = requirements(add(retained, candidate));
    runtime::admit_execution(required, lease.policy());
    if (required.scalar_work > tower.publication().limits().max_action_work)
        throw std::length_error("native batch accession work budget");
    if (grow) {
        auto next = std::make_unique<field::PreparedExactExecutionTower::BatchWorkspace>(
            tower.tower().make_batch_workspace(grade, samples));
        lease.check_cancelled();
        workspace = std::move(next);
    }
    return *workspace;
}
}
struct Source::State {
    storage::NativeTemporalSeries history;
    std::unique_ptr<storage::NativeAccessionDirectory> directory;
    mutable std::mutex mutex;
    SourceInfo manifest;
    explicit State(storage::NativeTemporalSeries value):history(std::move(value)) {
        const auto endpoint=history.tail().pin();
        directory=std::make_unique<storage::NativeAccessionDirectory>(*endpoint, history.source_namespace());
        manifest.name_space=history.source_namespace(); manifest.sequence=endpoint->sequence;
        manifest.first_sequence=history.first_sequence();
        manifest.archive=bytes_string(storage::encode_native_series(history));
        const auto source_bytes=storage::encode_rfcx(endpoint->source).bytes;
        const auto channel_bytes=storage::encode_temporal_endpoint(*endpoint);
        storage::temporal_codec_detail::Writer writer;
        writer.u64(source_bytes.size()); writer.raw(source_bytes);
        writer.u64(channel_bytes.size()); writer.raw(channel_bytes);
        manifest.endpoint=bytes_string(writer.bytes());
        for(std::size_t k=0;k<endpoint->source.grade_count();++k) {
            const auto& arena=endpoint->source.grade(core::Grade{static_cast<std::uint16_t>(k)});
            std::vector<std::string> keys; keys.reserve(arena.cell_count());
            if(k==0) for(const auto key:arena.identities) keys.push_back(std::to_string(key));
            else for(const auto key:arena.relation_identities) keys.push_back(std::to_string(key.value));
            manifest.coordinates.push_back(std::move(keys));
        }
        for(const auto& carrier:endpoint->carriers) {
            manifest.carrier_keys.push_back(carrier.key); manifest.carrier_grades.push_back(carrier.grade.value);
            manifest.carrier_kinds.push_back(static_cast<unsigned>(carrier.kind));
        }
        for(const auto& tensor:endpoint->tensors) {
            manifest.tensor_keys.push_back(tensor.key); manifest.tensor_carriers.push_back(tensor.output.value_or(0));
            manifest.tensor_has_output.push_back(tensor.output.has_value()?1u:0u);
            manifest.tensor_roles.push_back(static_cast<unsigned>(tensor.role));
        }
    }
};
struct Field::State {
    storage::PinnedNativeAccession pin;
    explicit State(storage::PinnedNativeAccession value):pin(std::move(value)) {}
};
struct Plan::State {
    inference::PreparedNativeExactTower tower;
    runtime::ExecutionHostPolicy policy;
    mutable std::mutex mutex;
    mutable std::unique_ptr<field::PreparedExactExecutionTower::BatchWorkspace> workspace;
    State(inference::PreparedNativeExactTower value, runtime::ExecutionHostPolicy host)
        :tower(std::move(value)),policy(std::move(host)) {}
};
struct Cancellation::State {
    std::shared_ptr<std::atomic_bool> flag=std::make_shared<std::atomic_bool>(false);
};
Cancellation::Cancellation():state_(std::make_shared<State>()) {}
void Cancellation::cancel(){state_->flag->store(true,std::memory_order_relaxed);}
bool Cancellation::cancelled() const{return state_->flag->load(std::memory_order_relaxed);}
std::shared_ptr<const std::atomic_bool> Cancellation::flag() const{return state_->flag;}
Source::Source(std::shared_ptr<State> state):state_(std::move(state)) {}
std::shared_ptr<Source> Source::load(const char* data,std::size_t size,std::size_t byte_limit,
    std::size_t max_dimension,std::size_t max_records,std::size_t max_logical_size) {
    storage::NativeSeriesLimits limits; limits.max_archive_bytes=byte_limit;
    limits.temporal.max_dimension=max_dimension;limits.temporal.max_records=max_records;
    limits.temporal.max_logical_size=max_logical_size;
    auto history=storage::decode_native_series(
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(data),size),limits);
    return std::shared_ptr<Source>(new Source(std::make_shared<State>(std::move(history))));
}
SourceInfo Source::info() const{return state_->manifest;}
std::shared_ptr<const storage::TemporalEndpoint> Source::endpoint() const {
    return state_->history.tail().pin();
}
storage::NativeSeriesEndpoint Source::native_endpoint() const {
    return state_->history.tail();
}
std::size_t Source::retained_revisions() const {
    return machine::checked_size_add(machine::checked_size(state_->history.last_sequence()-state_->history.first_sequence()),1);
}
SourceCarrier Source::carrier(std::uint64_t key) const {
    const auto endpoint=state_->history.tail().pin();
    const auto carriers=storage::temporal_endpoint_detail::carriers(*endpoint);
    const auto& c=*carriers.at(key);
    return {c.key,c.coordinate_system,static_cast<unsigned>(c.kind),c.grade.value,c.domain.logical_size()};
}
std::shared_ptr<Source> Source::build(const SourceDeclaration& d,std::size_t max_bytes) {
    core::RelationSignature signature;
    std::vector<composite::AspectId> role_aspects;
    for(const auto& name:d.role_aspects)
        role_aspects.push_back(signature.aspects().add(name,composite::AspectRole::RelationRole));
    for(const auto& t:d.types)signature.add_type(t.name,core::BoundaryPolicy::DeclaredExact,t.min_arity,t.max_arity);
    core::RelationalComplexBuilder builder(std::move(signature),core::SemanticScale{d.semantic_scale});
    builder.declare_grade(core::Grade{d.max_grade});
    for(const auto identity:d.witnesses)builder.declare_witness_identity(identity);
    std::vector<std::unordered_map<std::uint64_t,std::uint32_t>> keys(static_cast<std::size_t>(d.max_grade)+1);
    unsigned previous=0;
    for(const auto& r:d.relations) {
        if(!r.grade || r.grade>d.max_grade || r.grade<previous)throw std::invalid_argument("source relations require ascending declared grades");
        previous=r.grade;
        core::RelationId added;
        if(r.grade==1) {
            std::vector<core::BoundaryOccurrenceInput> occurrences;
            for(const auto& o:r.occurrences) {
                std::vector<core::AspectValue> aspects;
                for(const auto index:o.aspects)aspects.push_back({role_aspects.at(index),true});
                occurrences.emplace_back(o.lower,o.exists,o.negative,coefficient(o.magnitude),
                    std::move(aspects),core::OccurrenceKey{o.key});
            }
            added=builder.add_grade_one(core::RelationKey{r.key},r.type,occurrences);
        } else {
            std::vector<core::HigherIncidenceInput> occurrences;
            for(const auto& o:r.occurrences) {
                std::vector<core::AspectValue> aspects;
                for(const auto index:o.aspects)aspects.push_back({role_aspects.at(index),true});
                occurrences.emplace_back(keys.at(r.grade-1).at(o.lower),o.exists,o.negative,
                    coefficient(o.magnitude),std::move(aspects),core::OccurrenceKey{o.key});
            }
            added=builder.add_higher(core::Grade{r.grade},core::RelationKey{r.key},r.type,occurrences);
        }
        keys.at(r.grade).emplace(r.key,added.index);
    }
    storage::TemporalEndpoint endpoint(builder.freeze());
    for(const auto& c:d.carriers) {
        if(c.kind>1)throw std::invalid_argument("unsupported source construction carrier");
        endpoint.carriers.push_back({c.key,static_cast<storage::TemporalCarrierKind>(c.kind),core::Grade{c.grade},{},
            c.coordinates,numeric::IndexDomain::linear(c.size)});
    }
    const auto carriers=storage::temporal_endpoint_detail::carriers(endpoint);
    for(const auto& f:d.fields) {
        const auto& domain=carriers.at(f.carrier)->domain;
        if(f.indices.size()!=f.values.size())throw std::invalid_argument("native field sparse entry mismatch");
        std::vector<numeric::SparseRationalEntry> entries;
        for(std::size_t i=0;i<f.indices.size();++i)entries.push_back({f.indices[i],coefficient(f.values[i])});
        storage::TemporalTensor field;field.key=f.key;field.role=storage::TemporalTensorRole::Field;field.output=f.carrier;
        field.tensor={domain,f.full?numeric::StructuralSupport::full(domain.logical_size()):
            numeric::StructuralSupport::sparse(domain.logical_size(),f.support),numeric::SparseRationalVector(domain.logical_size(),std::move(entries))};
        endpoint.tensors.push_back(std::move(field));
    }
    for(const auto& m:d.metrics) {
        const auto n=carriers.at(m.carrier)->domain.logical_size();
        if(n!=m.diagonal.size())throw std::invalid_argument("native metric diagonal size mismatch");
        numeric::ExactMatrix matrix(n,n);
        for(std::size_t i=0;i<n;++i)matrix.set(i,i,coefficient(m.diagonal[i]));
        storage::TemporalTensor metric;metric.key=m.key;metric.role=storage::TemporalTensorRole::Metric;
        metric.input=metric.output=m.carrier;metric.tensor=storage::temporal_matrix_tensor(matrix);
        endpoint.tensors.push_back(std::move(metric));
    }
    storage::validate_temporal_endpoint(endpoint);
    auto result=std::shared_ptr<Source>(new Source(std::make_shared<State>(storage::NativeTemporalSeries(std::move(endpoint),d.name_space))));
    if(result->state_->manifest.archive.size()>max_bytes)throw std::length_error("constructed native archive exceeds byte limit");
    return result;
}
std::shared_ptr<Source> Source::at(std::uint64_t cutoff) const {
    const auto prefix=storage::NativeSeriesPrefix::capture(state_->history,cutoff);
    const auto first=prefix.at(prefix.first_sequence());
    storage::NativeTemporalSeries retained(first.endpoint(),prefix.source_namespace(),state_->history.limits());
    for(std::size_t i=0;i<prefix.segment_count();++i) retained.append_encoded(prefix.segment_bytes(i));
    return std::shared_ptr<Source>(new Source(std::make_shared<State>(std::move(retained))));
}
std::shared_ptr<Field> Source::field(std::uint64_t key) const {
    std::lock_guard guard(state_->mutex);
    return std::shared_ptr<Field>(new Field(std::make_shared<Field::State>(
        state_->directory->pin(tensor_request(storage::NativeAccessionKind::Field,key)))));
}
std::shared_ptr<Plan> Source::prepare(const std::vector<std::uint64_t>& carriers,
    const std::vector<std::uint64_t>& metrics,unsigned selection,bool global,
    std::size_t dimension_limit,const std::vector<std::size_t>& limits) const {
    auto policy=policy_for(limits);
    inference::NativeExactTowerRequest request; request.grade_carriers=carriers;
    request.metrics=static_cast<inference::NativeMetricSelection>(selection);
    request.profile=global?field::PreparedExactProfile::GlobalActions:field::PreparedExactProfile::FiniteActions;
    request.global_grade_dimension_limit=dimension_limit;
    std::lock_guard guard(state_->mutex);
    for(const auto key:metrics) request.metric_tensors.push_back(state_->directory->pin(tensor_request(storage::NativeAccessionKind::Tensor,key)));
    auto anchor=storage::NativeAccessionRequest::chain(0,core::Grade{0}); anchor.inline_value=true;
    const auto publication=state_->directory->pin(anchor);
    auto prepared=inference::PreparedNativeExactTower::prepare(publication,std::move(request),policy);
    return std::shared_ptr<Plan>(new Plan(std::make_shared<Plan::State>(std::move(prepared),std::move(policy))));
}
Field::Field(std::shared_ptr<State> state):state_(std::move(state)) {}
std::uint64_t Field::key() const {return state_->pin.tensor_declaration()->key;}
std::uint64_t Field::carrier() const {return state_->pin.tensor_declaration()->output.value();}
FieldInfo Field::info() const {
    const auto declaration=state_->pin.tensor_declaration(); const auto tensor=state_->pin.tensor();
    FieldInfo out; out.key=declaration->key; out.carrier=declaration->output.value();
    out.values=rationals(inference::native_field_detail::dense(*tensor));
    tensor->support().for_each_active([&](std::size_t index){out.support.push_back(index);});
    return out;
}
Plan::Plan(std::shared_ptr<State> state):state_(std::move(state)) {}
RationalBatchResult Plan::execute_batch(const std::vector<std::shared_ptr<Field>>& inputs,
    unsigned kind,std::uint16_t grade,const std::string& parameter,const std::shared_ptr<Cancellation>& cancellation) const {
    RationalBatchResult out;
    try {
        if(inputs.empty() || inputs.size()>state_->policy.limits.records)
            throw std::invalid_argument("native field batch sample count exceeds its record bound");
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        std::vector<storage::PinnedNativeAccession> pins; pins.reserve(inputs.size());
        for(const auto& input:inputs) {
            if(!input)throw std::invalid_argument("native field batch requires owning fields");
            pins.push_back(input->state_->pin);
        }
        const field::OperatorRequest operation{static_cast<field::OperatorKind>(kind),core::Grade{grade},coefficient(parameter)};
        std::lock_guard guard(state_->mutex);
        auto& workspace=select_batch_workspace(state_->tower,operation.grade,inputs.size(),state_->workspace,lease,
            [&](auto slots){return state_->tower.batch_requirements(operation,pins,slots);});
        const auto result=state_->tower.execute_batch(operation,pins,workspace,lease); const auto& report=result.provenance();
        auto& status=out.status; status.values=rationals(result.values());
        status.input_carrier=report.input_carrier; status.output_carrier=report.output_carrier;
        status.name_space=report.source_namespace; status.sequence=report.sequence; status.profile=report.profile;
        status.implementation=report.implementation; status.integer_bits=report.requirements.integer_bits;
        status.scalar_work=report.requirements.scalar_work;
        status.output_grade=static_cast<std::uint16_t>(operation.kind==field::OperatorKind::Boundary?grade-1u:grade);
        for(const auto& input:inputs)out.input_fields.push_back(input->key());
        out.cells=result.cells(); out.samples=result.samples(); out.workspace_capacity=workspace.capacity();
        out.workspace_slots=workspace.retained_slots();
        lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
ExecutionResult Plan::execute(const std::shared_ptr<Field>& input,unsigned kind,std::uint16_t grade,
    const std::string& coefficient,const std::shared_ptr<Cancellation>& cancellation) const {
    ExecutionResult out;
    try {
        if(!input) throw std::invalid_argument("native execution requires an owning field");
        numeric::Rational parameter;
        if(parameter.set_str(coefficient,16)!=0 || parameter.get_den()==0) throw std::invalid_argument("invalid exact operation parameter");
        parameter.canonicalize();
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled(); std::lock_guard guard(state_->mutex);
        const auto result=state_->tower.execute({static_cast<field::OperatorKind>(kind),core::Grade{grade},parameter},input->state_->pin,lease);
        const auto& report=result.provenance();
        out.values=rationals(result.values()); out.input_field=report.input_field;
        out.input_carrier=report.input_carrier; out.output_carrier=report.output_carrier;
        out.name_space=report.source_namespace; out.sequence=report.sequence;
        out.profile=report.profile; out.implementation=report.implementation;
        out.integer_bits=report.requirements.integer_bits; out.scalar_work=report.requirements.scalar_work;
        out.output_grade=static_cast<std::uint16_t>(kind==static_cast<unsigned>(field::OperatorKind::Boundary)?grade-1u:grade);
    } catch(const runtime::ExecutionCancelled& error) {
        out.error=1; out.error_message=error.what();
    } catch(const runtime::ExecutionLimitExceeded& error) {
        out.error=2; out.error_message=error.what(); out.error_resource=error.resource();
        out.error_required=error.required(); out.error_limit=error.limit();
    }
    return out;
}
struct IntegerInput::State {
    std::shared_ptr<Source::State> source;
    std::uint16_t grade;
    std::uint64_t carrier;
    std::vector<std::int64_t> values;
    State(std::shared_ptr<Source::State> s,std::uint16_t g,std::uint64_t c,std::vector<std::int64_t> v)
        :source(std::move(s)),grade(g),carrier(c),values(std::move(v)){}
};
std::shared_ptr<IntegerInput> Source::integer_input(std::uint16_t grade,std::uint64_t carrier,
    const std::vector<std::int64_t>& values) const {
    const auto endpoint=state_->history.tail().pin();
    const auto carriers=storage::temporal_endpoint_detail::carriers(*endpoint);
    if(!carriers.contains(carrier))throw std::invalid_argument("integer input carrier absent");
    const auto& selected=*carriers.at(carrier);
    if(selected.kind!=storage::TemporalCarrierKind::SourceGrade || selected.grade!=core::Grade{grade})
        throw std::invalid_argument("integer input requires its declared SourceGrade carrier");
    if(values.size()!=endpoint->source.grade(core::Grade{grade}).cell_count())
        throw std::invalid_argument("integer input does not match its retained grade");
    return std::shared_ptr<IntegerInput>(new IntegerInput(std::make_shared<IntegerInput::State>(state_,grade,carrier,values)));
}
IntegerInput::IntegerInput(std::shared_ptr<State> state):state_(std::move(state)){}
std::vector<std::int64_t> IntegerInput::values() const{return state_->values;}
std::uint64_t IntegerInput::carrier() const{return state_->carrier;}
std::uint16_t IntegerInput::grade() const{return state_->grade;}
struct IntegerBatch::State {
    std::shared_ptr<Source::State> source;
    std::uint16_t grade;
    std::uint64_t carrier;
    std::size_t cells,samples;
    std::vector<std::int64_t> values;
    State(std::shared_ptr<Source::State> s,std::uint16_t g,std::uint64_t c,std::size_t n,std::size_t b,
          std::vector<std::int64_t> v):source(std::move(s)),grade(g),carrier(c),cells(n),samples(b),values(std::move(v)){}
};
IntegerBatchBuild Source::integer_batch(std::uint16_t grade,std::uint64_t carrier,
    std::vector<std::int64_t> values,std::size_t samples,const std::vector<std::size_t>& limits) const {
    IntegerBatchBuild out;
    try {
        const auto endpoint=state_->history.tail().pin();
        const auto carriers=storage::temporal_endpoint_detail::carriers(*endpoint);
        if(!carriers.contains(carrier) || carriers.at(carrier)->kind!=storage::TemporalCarrierKind::SourceGrade ||
           carriers.at(carrier)->grade!=core::Grade{grade})throw std::invalid_argument("integer batch SourceGrade carrier mismatch");
        const auto cells=endpoint->source.grade(core::Grade{grade}).cell_count();
        if(!cells || !samples || values.size()!=learning::IntegerNetworkWorkspace::slots(cells,samples))
            throw std::invalid_argument("integer batch requires nonempty matching cells and samples");
        runtime::admit_execution({values.size(),values.size(),values.size(),values.size(),64},policy_for(limits));
        out.batch=std::shared_ptr<IntegerBatch>(new IntegerBatch(std::make_shared<IntegerBatch::State>(
            state_,grade,carrier,cells,samples,std::move(values))));
    } catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
IntegerBatch::IntegerBatch(std::shared_ptr<State> state):state_(std::move(state)){}
std::vector<std::int64_t> IntegerBatch::values() const{return state_->values;}
std::uint64_t IntegerBatch::carrier() const{return state_->carrier;}
std::uint16_t IntegerBatch::grade() const{return state_->grade;}
std::size_t IntegerBatch::cells() const{return state_->cells;}
std::size_t IntegerBatch::samples() const{return state_->samples;}
struct IntegerNetwork::State {
    std::shared_ptr<Source::State> source;
    std::shared_ptr<const storage::TemporalEndpoint> endpoint;
    core::Grade grade;
    std::uint64_t carrier;
    runtime::ExecutionHostPolicy policy;
    runtime::ExecutionRequirements requirements;
    learning::IntegerRelationalNetwork network;
    std::unique_ptr<learning::IntegerNetworkWorkspace> workspace;
    mutable std::mutex batch_mutex;
    State(std::shared_ptr<Source::State> s,std::uint16_t g,std::uint64_t c,
          const std::vector<IntegerLayer>& layers,const std::vector<std::size_t>& limits)
        :source(std::move(s)),endpoint(source->history.tail().pin()),grade(g),carrier(c),
         policy(policy_for(limits)),network(endpoint->source,grade) {
        if(layers.empty() || layers.size()>64)throw std::invalid_argument("integer network requires 1 to 64 layers");
        const auto carriers=storage::temporal_endpoint_detail::carriers(*endpoint);
        if(!carriers.contains(carrier) || carriers.at(carrier)->kind!=storage::TemporalCarrierKind::SourceGrade ||
           carriers.at(carrier)->grade!=grade)throw std::invalid_argument("integer network SourceGrade carrier mismatch");
        using inference::native_field_detail::add; using inference::native_field_detail::mul;
        std::size_t cells=0,occurrences=0,channels=0;
        for(std::size_t k=0;k<endpoint->source.grade_count();++k) {
            const auto& arena=endpoint->source.grade(core::Grade{static_cast<std::uint16_t>(k)});
            cells=add(cells,arena.cell_count()); occurrences=add(occurrences,arena.incidence_count());
        }
        for(const auto& layer:layers) {
            if(layer.channels.size()>64)throw std::invalid_argument("integer network channel count limit");
            channels=add(channels,add(layer.channels.size(),1));
        }
        requirements.records=add(cells,channels); requirements.sparse_entries=mul(cells,4);
        requirements.scalar_work=mul(mul(add(cells,occurrences),channels),8); requirements.integer_bits=64;
        runtime::admit_execution(requirements,policy);
        for(const auto& layer:layers) {
            std::vector<learning::OperatorChannelKind> kinds;
            for(const auto kind:layer.channels)kinds.push_back(static_cast<learning::OperatorChannelKind>(kind));
            network.add_layer(learning::CompiledIntegerRelationalLayer(endpoint->source,grade,
                std::move(kinds),layer.gains,layer.bias,static_cast<learning::IntegerActivation>(layer.activation)));
        }
    }
};
std::shared_ptr<IntegerNetwork> Source::integer_network(std::uint16_t grade,std::uint64_t carrier,
    const std::vector<IntegerLayer>& layers,const std::vector<std::size_t>& limits) const {
    return std::shared_ptr<IntegerNetwork>(new IntegerNetwork(std::make_shared<IntegerNetwork::State>(state_,grade,carrier,layers,limits)));
}
IntegerNetwork::IntegerNetwork(std::shared_ptr<State> state):state_(std::move(state)){}
ExecutionResult IntegerNetwork::forward(const std::shared_ptr<IntegerInput>& input,
    const std::shared_ptr<Cancellation>& cancellation) const {
    ExecutionResult out;
    try {
        if(!input || input->state_->source.get()!=state_->source.get() || input->grade()!=state_->grade.value ||
           input->carrier()!=state_->carrier)throw std::invalid_argument("integer network input publication/grade/carrier mismatch");
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled(); runtime::admit_execution(state_->requirements,lease.policy());
        const auto result=state_->network.forward(field::MachineLatticeField(state_->endpoint->source,state_->grade,input->state_->values));
        for(const auto value:result.values())out.values.push_back(numeric::Rational{std::to_string(value)}.get_str(16));
        out.name_space=state_->source->history.source_namespace(); out.sequence=state_->endpoint->sequence;
        out.input_carrier=out.output_carrier=state_->carrier; out.output_grade=state_->grade.value;
        out.profile="checked-int64-lattice-network-v1"; out.implementation="IntegerRelationalNetwork/checked-int64/serial-v1";
        out.integer_bits=64; out.scalar_work=state_->requirements.scalar_work; lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){failure(out,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out,error);}
    return out;
}
IntegerBatchResult IntegerNetwork::forward_batch(const std::shared_ptr<IntegerBatch>& input,
    const std::shared_ptr<Cancellation>& cancellation) const {
    IntegerBatchResult out;
    try {
        using inference::native_field_detail::add; using inference::native_field_detail::mul;
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        if(!input || input->state_->source.get()!=state_->source.get() || input->grade()!=state_->grade.value ||
           input->carrier()!=state_->carrier)throw std::invalid_argument("integer batch publication/grade/carrier mismatch");
        std::lock_guard lock(state_->batch_mutex);
        lease.check_cancelled();
        const auto b=input->samples();const auto n=input->cells();const auto count=mul(n,b);
        const auto& complex=state_->endpoint->source;
        const auto lower=state_->grade.value?complex.grade(core::Grade{static_cast<std::uint16_t>(state_->grade.value-1u)}).cell_count():0;
        const auto upper=state_->grade.value<complex.max_grade().value?
            complex.grade(core::Grade{static_cast<std::uint16_t>(state_->grade.value+1u)}).cell_count():0;
        const auto retained=state_->workspace?state_->workspace->retained_slots():0;
        const auto grow=!state_->workspace || b>state_->workspace->capacity();
        const auto candidate=grow?mul(add(mul(n,5),add(lower,upper)),b):0;
        auto required=state_->requirements;
        required.records=add(required.records,count);
        required.rational_slots=add(mul(count,2),add(retained,candidate));
        required.sparse_entries=add(required.sparse_entries,required.rational_slots);
        required.scalar_work=mul(required.scalar_work,b);
        runtime::admit_execution(required,lease.policy());
        if(grow)state_->workspace=std::make_unique<learning::IntegerNetworkWorkspace>(complex,state_->grade,b);
        auto values=state_->network.forward_batch(input->state_->values,b,*state_->workspace,[&lease]{lease.check_cancelled();});
        lease.check_cancelled();
        out.values=std::move(values);out.cells=n;out.samples=b;
        out.workspace_capacity=state_->workspace->capacity();out.workspace_slots=state_->workspace->retained_slots();
        auto& status=out.status;
        status.name_space=state_->source->history.source_namespace();status.sequence=state_->endpoint->sequence;
        status.input_carrier=status.output_carrier=state_->carrier;status.output_grade=state_->grade.value;
        status.profile="checked-int64-lattice-batch-v1";
        status.implementation="IntegerRelationalNetwork/checked-int64/reused-batch-v1";
        status.integer_bits=64;status.scalar_work=required.scalar_work;
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
struct Trainer::State {
    std::shared_ptr<Source::State> source;
    std::shared_ptr<Plan> plan;
    TrainingRequest request;
    learning::NativeExactTrainer initial,current;
    std::shared_ptr<learning::NativeExactTrainer::TrainingWorkspace> training_workspace;
    std::shared_ptr<Source> parameter_source;
    std::shared_ptr<Plan> parameter_plan;
    std::unique_ptr<learning::PreparedNativeMetricLayer> layer;
    TrainingInfo manifest;
    State(std::shared_ptr<Source::State> s,std::shared_ptr<Plan> p,TrainingRequest r,
          learning::NativeExactTrainer first,learning::NativeExactTrainer candidate,
          std::shared_ptr<learning::NativeExactTrainer::TrainingWorkspace> workspace)
        :source(std::move(s)),plan(std::move(p)),request(std::move(r)),initial(std::move(first)),current(std::move(candidate)),
         training_workspace(std::move(workspace)) {
        auto history=current.history(); manifest.step=history.last_sequence(); manifest.history_namespace=history.source_namespace();
        manifest.parameters=rationals(current.parameters()); const auto loss=current.last_loss();
        manifest.parameter_knowledge_time=current.parameter_knowledge_time();
        manifest.has_loss=loss.has_value(); if(loss)manifest.loss=loss->get_str(16);
        parameter_source=std::shared_ptr<Source>(new Source(std::make_shared<Source::State>(std::move(history))));
        const auto& tower=plan->state_->tower;
        std::vector<std::uint64_t> carriers;
        for(std::size_t k=0;k<tower.publication().endpoint()->source.grade_count();++k)
            carriers.push_back(tower.grade_carrier(core::Grade{static_cast<std::uint16_t>(k)}));
        const auto& bounds=plan->state_->policy.limits;
        parameter_plan=parameter_source->prepare(carriers,tower.metric_keys(),static_cast<unsigned>(tower.metric_selection()),
            false,64,{bounds.records,bounds.rational_slots,bounds.sparse_entries,bounds.scalar_work,bounds.integer_bits});
        const auto gains=parameter_source->field(request.gains),bias=parameter_source->field(request.bias);
        layer=std::make_unique<learning::PreparedNativeMetricLayer>(parameter_plan->state_->tower,core::Grade{request.grade},
            metric_channels(request.channels),gains->state_->pin,bias->state_->pin,
            learning::NativeMetricLayerBinding{request.model,request.channel_coordinates,request.bias_coordinates},plan->state_->policy);
    }
};
std::shared_ptr<Trainer> Source::trainer(const std::shared_ptr<Plan>& plan,const TrainingRequest& request,
    const std::string& archive,std::size_t cache_samples) const {
    if(cache_samples>1024)throw std::invalid_argument("training cache sample capacity limit");
    if(!plan || plan->state_->tower.profile()!=field::PreparedExactProfile::FiniteActions)
        throw std::invalid_argument("exact trainer requires the finite declared metric profile");
    std::vector<storage::PinnedNativeAccession> parameters;
    std::vector<learning::NativeExactTrainingSample> samples;
    {
        std::lock_guard guard(state_->mutex);
        parameters.push_back(state_->directory->pin(tensor_request(storage::NativeAccessionKind::Field,request.gains)));
        parameters.push_back(state_->directory->pin(tensor_request(storage::NativeAccessionKind::Field,request.bias)));
        for(const auto& sample:request.samples)samples.push_back({
            state_->directory->pin(tensor_request(storage::NativeAccessionKind::Field,sample.input)),
            state_->directory->pin(tensor_request(storage::NativeAccessionKind::Field,sample.target)),
            static_cast<learning::NativeSamplePartition>(sample.partition),sample.event_time,sample.knowledge_time,sample.label_available_time});
    }
    const learning::NativeExactTrainingContract contract{learning::NativeTrainingObjective::HalfMetricSquaredError,
        {request.model,request.channel_coordinates,request.bias_coordinates},request.history_namespace,
        request.metadata_carrier,request.metadata_tensor,request.time_coordinates,request.initial_parameter_knowledge_time,
        request.publication_cutoff,request.metadata_coordinates,request.metadata_capacity};
    std::lock_guard guard(plan->state_->mutex);
    auto initial=learning::NativeExactTrainer::prepare(plan->state_->tower,core::Grade{request.grade},metric_channels(request.channels),
        parameters.at(0),parameters.at(1),std::move(samples),contract,plan->state_->policy);
    std::shared_ptr<learning::NativeExactTrainer::TrainingWorkspace> workspace;
    if(cache_samples)workspace=std::make_shared<learning::NativeExactTrainer::TrainingWorkspace>(
        initial.make_training_workspace(std::min(cache_samples,request.samples.size()),runtime::ExecutionLease(plan->state_->policy)));
    auto current=initial;
    if(!archive.empty()) {
        const auto bytes=std::span<const std::byte>(reinterpret_cast<const std::byte*>(archive.data()),archive.size());
        const runtime::ExecutionLease lease(plan->state_->policy);
        current=workspace?learning::NativeExactTrainer::resume_prepared(initial,bytes,*workspace,lease)
                         :learning::NativeExactTrainer::resume(initial,bytes,lease);
    }
    return std::shared_ptr<Trainer>(new Trainer(std::make_shared<Trainer::State>(state_,plan,request,std::move(initial),std::move(current),std::move(workspace))));
}
Trainer::Trainer(std::shared_ptr<State> state):state_(std::move(state)){}
TrainingInfo Trainer::info() const {
    auto manifest=state_->manifest;
    manifest.archive=state_->parameter_source->info().archive;
    return manifest;
}
std::uint64_t Trainer::parameter_knowledge_time() const noexcept{return state_->manifest.parameter_knowledge_time;}
TrainingExecutionInfo Trainer::training_execution() const {
    std::lock_guard guard(state_->plan->state_->mutex);
    TrainingExecutionInfo out;
    const auto& workspace=state_->training_workspace;
    out.profile=workspace?"original-parameter-finite-channel-statistics-v1":"original-parameter-field-gradient-v1";
    if(workspace) {
        for(const auto& channel:workspace->response_channels())
            out.response_channels.emplace_back(learning::exact_metric_channel_name(channel.kind));
        out.response_channels.emplace_back("scalar_bias");
        out.parameter_coordinates=workspace->parameter_coordinates();
        out.capacity=workspace->capacity(); out.cached_samples=workspace->cached_samples();
        out.retained_slots=workspace->retained_slots(); out.hits=workspace->hits(); out.misses=workspace->misses();
        const auto& required=workspace->last_requirements();
        out.scalar_work=required.scalar_work; out.integer_bits=required.integer_bits; out.rational_slots=required.rational_slots;
    }
    return out;
}
TrainingUpdate Trainer::advance(const std::string& rate,std::uint64_t events,std::uint64_t knowledge,
    std::size_t steps,const std::shared_ptr<Cancellation>& cancellation) const {
    TrainingUpdate out;
    try {
        if(!steps || steps>1024-state_->manifest.step)throw std::invalid_argument("exact training requires 1 to 1024 retained steps");
        std::lock_guard guard(state_->plan->state_->mutex);
        const runtime::ExecutionLease lease(state_->plan->state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled(); auto candidate=state_->current; const auto selected=coefficient(rate);
        for(std::size_t i=0;i<steps;++i)candidate=state_->training_workspace
            ?candidate.advance_prepared(selected,events,knowledge,*state_->training_workspace,lease)
            :candidate.advance(selected,events,knowledge,lease);
        auto result=std::make_shared<State>(state_->source,state_->plan,state_->request,state_->initial,std::move(candidate),state_->training_workspace);
        lease.check_cancelled(); out.trainer=std::shared_ptr<Trainer>(new Trainer(std::move(result)));
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
ExecutionResult Trainer::forward(const std::shared_ptr<Field>& input,const std::shared_ptr<Cancellation>& cancellation) const {
    ExecutionResult out;
    try {
        if(!input)throw std::invalid_argument("exact model inference requires an owning field");
        const auto& original=state_->plan->state_->tower;
        original.validate_input(input->state_->pin,core::Grade{state_->request.grade});
        const auto retained=state_->parameter_source->field(input->key());
        std::lock_guard guard(state_->parameter_plan->state_->mutex);
        const runtime::ExecutionLease lease(state_->plan->state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        const auto result=state_->layer->forward(retained->state_->pin,lease); const auto& report=result.provenance();
        out.values=rationals(result.values()); out.input_field=input->key(); out.input_carrier=out.output_carrier=report.output_carrier;
        out.name_space=original.publication().source_namespace(); out.sequence=original.publication().sequence();
        out.output_grade=state_->request.grade; out.profile=report.profile;
        out.implementation="PreparedNativeMetricLayer/GMP/original-channels-v1";
        out.integer_bits=report.requirements.integer_bits; out.scalar_work=report.requirements.scalar_work;
    } catch(const runtime::ExecutionCancelled& error){failure(out,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out,error);}
    return out;
}
RationalBatchResult Trainer::forward_batch(const std::vector<std::shared_ptr<Field>>& inputs,
    const std::shared_ptr<Cancellation>& cancellation) const {
    RationalBatchResult out;
    try {
        if(inputs.empty() || inputs.size()>state_->plan->state_->policy.limits.records)
            throw std::invalid_argument("native metric batch sample count exceeds its record bound");
        const runtime::ExecutionLease lease(state_->plan->state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        const auto grade=core::Grade{state_->request.grade}; const auto& original=state_->plan->state_->tower;
        for(const auto& input:inputs) {
            if(!input)throw std::invalid_argument("native metric batch requires owning fields");
            original.validate_input(input->state_->pin,grade);
        }
        std::vector<storage::PinnedNativeAccession> pins; pins.reserve(inputs.size());
        for(const auto& input:inputs)pins.push_back(state_->parameter_source->field(input->key())->state_->pin);
        auto& plan=*state_->parameter_plan->state_;
        std::lock_guard guard(plan.mutex);
        auto& workspace=select_batch_workspace(plan.tower,grade,inputs.size(),plan.workspace,lease,
            [&](auto slots){return state_->layer->batch_requirements(pins,slots);});
        const auto result=state_->layer->forward_batch(pins,workspace,lease); const auto& report=result.provenance();
        auto& status=out.status; status.values=rationals(result.values());
        status.input_carrier=status.output_carrier=report.output_carrier; status.output_grade=grade.value;
        status.name_space=original.publication().source_namespace(); status.sequence=original.publication().sequence();
        status.profile=report.profile; status.implementation="PreparedNativeMetricLayer/GMP/original-channels-batch-v1";
        status.integer_bits=report.requirements.integer_bits; status.scalar_work=report.requirements.scalar_work;
        for(const auto& input:inputs)out.input_fields.push_back(input->key());
        out.cells=result.cells(); out.samples=result.samples(); out.workspace_capacity=workspace.capacity();
        out.workspace_slots=workspace.retained_slots();
        lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
struct PerceptronRows::State {
    std::size_t features{},entries{};
    std::vector<learning::SparseIntSample> samples;
};
PerceptronRows::PerceptronRows(std::shared_ptr<const State> state):state_(std::move(state)){}
std::size_t PerceptronRows::rows() const{return state_->samples.size();}
std::size_t PerceptronRows::entries() const{return state_->entries;}
std::size_t PerceptronRows::features() const{return state_->features;}
struct Perceptron::State {
    learning::ExactIntegerPerceptron model;
    runtime::ExecutionHostPolicy policy;
    std::size_t mistakes{};
    State(std::vector<std::int64_t> weights,std::int64_t bias,const std::vector<std::size_t>& limits)
        :model(std::move(weights),bias),policy(policy_for(limits)) {
        if(model.weights().empty() || model.weights().size()>UINT32_MAX)throw std::invalid_argument("integer perceptron requires 1 to UINT32_MAX features");
        runtime::ExecutionRequirements required;required.records=model.weights().size();required.integer_bits=64;
        runtime::admit_execution(required,policy);
    }
    std::vector<learning::SparseIntSample> samples(const std::vector<std::vector<SparseFeature>>& rows,
        const std::vector<std::int64_t>& labels,const runtime::ExecutionLease& lease) const {
        if(rows.empty() || (!labels.empty() && labels.size()!=rows.size()))throw std::invalid_argument("integer perceptron sample count mismatch");
        using inference::native_field_detail::add;using inference::native_field_detail::mul;
        std::size_t nnz=0;for(const auto& row:rows)nnz=add(nnz,row.size());
        runtime::ExecutionRequirements required;required.records=add(model.weights().size(),rows.size());
        required.sparse_entries=add(nnz,model.weights().size());required.integer_bits=64;
        required.scalar_work=mul(add(nnz,rows.size()),8);runtime::admit_execution(required,lease.policy());
        std::vector<learning::SparseIntSample> samples;samples.reserve(rows.size());
        for(std::size_t i=0;i<rows.size();++i) {
            lease.check_cancelled();
            learning::SparseIntSample sample;sample.features.reserve(rows[i].size());
            if(!labels.empty()) {
                if(labels[i]!=1 && labels[i]!=-1)throw std::invalid_argument("integer perceptron labels require -1 or +1");
                sample.label=static_cast<std::int8_t>(labels[i]);
            }
            for(const auto& f:rows[i]) {
                if(f.index>=model.weights().size())throw std::out_of_range("integer perceptron feature index");
                sample.features.push_back({f.index,f.value});
            }
            samples.push_back(std::move(sample));
        }
        return samples;
    }
};
Perceptron::Perceptron(const std::vector<std::int64_t>& weights,std::int64_t bias,const std::vector<std::size_t>& limits)
    :state_(std::make_shared<State>(weights,bias,limits)){}
Perceptron::Perceptron(std::shared_ptr<State> state):state_(std::move(state)){}
PerceptronInfo Perceptron::info() const{return {state_->model.weights(),state_->model.bias(),state_->mistakes};}
PackedPerceptronInfo Perceptron::packed_info() const {
    PackedPerceptronInfo out;out.mistakes=state_->mistakes;
    const auto& weights=state_->model.weights();
    out.parameters.reserve(inference::native_field_detail::add(weights.size(),1));
    out.parameters.insert(out.parameters.end(),weights.begin(),weights.end());
    out.parameters.push_back(state_->model.bias());return out;
}
PerceptronUpdate Source::perceptron(std::uint64_t weights_key,std::uint64_t bias_key,
    const std::vector<std::size_t>& limits,const std::shared_ptr<Cancellation>& cancellation) const {
    PerceptronUpdate out;
    try {
        const auto policy=policy_for(limits);
        const runtime::ExecutionLease lease(policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        const auto weights=field(weights_key)->state_->pin;
        const auto bias=field(bias_key)->state_->pin;
        const auto w=weights.tensor(),b=bias.tensor();
        const auto wc=weights.carrier(),bc=bias.carrier();
        const auto count=w->domain().logical_size();
        if(wc->kind!=storage::TemporalCarrierKind::Explicit || bc->kind!=storage::TemporalCarrierKind::Explicit ||
           w->domain().rank()!=1 || b->domain()!=numeric::IndexDomain::linear(1) ||
           w->domain()!=wc->domain || b->domain()!=bc->domain || !count || count>UINT32_MAX)
            throw std::invalid_argument("integer parameters require explicit feature and scalar carriers");
        if(w->support().count()!=count || b->support().count()!=1)
            throw std::invalid_argument("integer parameters require full declared support");
        runtime::ExecutionRequirements required;required.records=count;required.integer_bits=64;
        required.sparse_entries=count;
        required.scalar_work=inference::native_field_detail::add(w->stored_nonzeros(),1);
        runtime::admit_execution(required,policy);
        std::vector<std::int64_t> parameters(count,0);
        for(const auto& entry:w->values().entries()) {
            lease.check_cancelled();parameters[entry.index]=integer_parameter(entry.value);
        }
        const auto scalar=b->values().entries().empty()?0:integer_parameter(b->values().entries().front().value);
        auto state=std::make_shared<Perceptron::State>(std::move(parameters),scalar,limits);
        lease.check_cancelled();out.model=std::shared_ptr<Perceptron>(new Perceptron(std::move(state)));
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
PerceptronRowsBuild Perceptron::prepare_rows(const std::vector<std::vector<SparseFeature>>& rows,
    const std::shared_ptr<Cancellation>& cancellation) const {
    PerceptronRowsBuild out;
    try {
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        auto owned=std::make_shared<PerceptronRows::State>();
        owned->features=state_->model.weights().size();owned->samples=state_->samples(rows,{},lease);
        for(const auto& row:owned->samples)owned->entries=inference::native_field_detail::add(owned->entries,row.features.size());
        lease.check_cancelled();out.rows=std::shared_ptr<PerceptronRows>(new PerceptronRows(std::move(owned)));
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
PerceptronOutput Perceptron::forward_prepared(const std::shared_ptr<PerceptronRows>& rows,
    std::size_t workers,std::size_t rows_per_tile,const std::shared_ptr<Cancellation>& cancellation) const {
    PerceptronOutput out;
    try {
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        if(!rows || rows->features()!=state_->model.weights().size())
            throw std::invalid_argument("prepared integer rows require the same feature count");
        if((workers && (!rows_per_tile || rows_per_tile>1000000)) || (!workers && rows_per_tile))
            throw std::invalid_argument("CPU scoring requires positive workers and 1 to 1000000 rows per tile");
        using inference::native_field_detail::add;using inference::native_field_detail::mul;
        const auto count=rows->rows(),nnz=rows->entries();
        const auto tiles=workers?machine::ceil_div(count,rows_per_tile):0;
        runtime::ExecutionRequirements required;
        required.records=workers?add(add(state_->model.weights().size(),mul(count,3)),
                                     add(mul(tiles,3),std::min(workers,tiles))):add(state_->model.weights().size(),count);
        required.sparse_entries=add(nnz,state_->model.weights().size());required.integer_bits=64;
        required.scalar_work=mul(add(nnz,count),8);runtime::admit_execution(required,lease.policy());
        const auto& samples=rows->state_->samples;
        out.scores.resize(count);out.labels.resize(count);
        if(workers) {
            runtime::CpuTileExecutor executor({workers,64,state_->policy.limits.records});
            const auto report=executor.run_tiles(tiles,[&](std::size_t tile,std::size_t,const runtime::CpuTileStop& stop) {
                const auto begin=mul(tile,rows_per_tile),end=add(begin,std::min(rows_per_tile,count-begin));
                for(auto row=begin;row<end;++row) {
                    stop.check();lease.check_cancelled();
                    const auto score=state_->model.score(samples[row].features);
                    out.scores[row]=score;out.labels[row]=score>=0?1:-1;
                }
            });
            out.requested_workers=report.requested_workers;out.worker_slots=report.worker_slots;
            out.launched_threads=report.launched_threads;out.tiles=report.completed_tiles;out.rows_per_tile=rows_per_tile;
        } else {
            for(std::size_t row=0;row<count;++row) {
                lease.check_cancelled();const auto score=state_->model.score(samples[row].features);
                out.scores[row]=score;out.labels[row]=score>=0?1:-1;
            }
        }
        lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){out.scores.clear();out.labels.clear();failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){out.scores.clear();out.labels.clear();failure(out.status,error);}
    return out;
}
PerceptronOutput Perceptron::forward(const std::vector<std::vector<SparseFeature>>& rows,
    const std::shared_ptr<Cancellation>& cancellation) const {
    const auto prepared=prepare_rows(rows,cancellation);
    if(prepared.status.error){PerceptronOutput out;out.status=prepared.status;return out;}
    return forward_prepared(prepared.rows,0,0,cancellation);
}
PerceptronOutput Perceptron::forward_cpu(const std::vector<std::vector<SparseFeature>>& rows,
    std::size_t workers,std::size_t rows_per_tile,const std::shared_ptr<Cancellation>& cancellation) const {
    if(!workers || !rows_per_tile || rows_per_tile>1000000)
        throw std::invalid_argument("CPU scoring requires 1 to 64 workers and 1 to 1000000 rows per tile");
    const auto prepared=prepare_rows(rows,cancellation);
    if(prepared.status.error){PerceptronOutput out;out.status=prepared.status;return out;}
    return forward_prepared(prepared.rows,workers,rows_per_tile,cancellation);
}
PerceptronUpdate Perceptron::advance(const std::vector<std::vector<SparseFeature>>& rows,
    const std::vector<std::int64_t>& labels,std::int64_t rate,const std::shared_ptr<Cancellation>& cancellation) const {
    PerceptronUpdate out;
    try {
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();const auto samples=state_->samples(rows,labels,lease);
        auto candidate=std::make_shared<State>(*state_);
        candidate->mistakes=candidate->model.train_epoch(samples,rate);
        lease.check_cancelled();out.model=std::shared_ptr<Perceptron>(new Perceptron(std::move(candidate)));
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
struct Transport::State {
    std::shared_ptr<Source::State> source,target;
    std::shared_ptr<const storage::TemporalEndpoint> from,to;
    runtime::ExecutionHostPolicy policy;
    std::unique_ptr<algebra::RelationalMorphism> morphism;
    std::map<std::uint64_t,CarrierLink> links;
    std::map<std::uint64_t,numeric::ExactMatrix> matrices;
    State(std::shared_ptr<Source::State> source_owner,std::shared_ptr<Source::State> target_owner,
          const std::vector<std::vector<unsigned>>& maps,const std::vector<CarrierLink>& carriers,
          runtime::ExecutionHostPolicy host,const runtime::ExecutionLease& lease)
        :source(std::move(source_owner)),target(std::move(target_owner)),from(source->history.tail().pin()),
         to(target->history.tail().pin()),policy(std::move(host)) {
        using namespace inference::native_field_detail;
        lease.check_cancelled();
        const auto& x=from->source;const auto& y=to->source;
        if(x.grade_count()!=y.grade_count() || maps.size()!=x.grade_count() || x.semantic_scale()!=y.semantic_scale())
            throw std::invalid_argument("transport requires matching retained grades and semantic scale");
        runtime::ExecutionRequirements required;
        for(std::size_t k=0;k<maps.size();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
            const auto& a_grade=x.grade(grade);const auto& b_grade=y.grade(grade);
            const auto n=a_grade.cell_count();
            if(n!=b_grade.cell_count() || maps[k].size()!=n)
                throw std::invalid_argument("transport requires a bijection on every retained grade");
            required.records=add(required.records,add(n,a_grade.incidence_count()));
            required.sparse_entries=add(required.sparse_entries,add(n,a_grade.incidence_count()));
            required.rational_slots=add(required.rational_slots,n);
            required.scalar_work=add(required.scalar_work,add(n,a_grade.incidence_count()));
            for(const auto& q:a_grade.cell_scales)required.integer_bits=std::max(required.integer_bits,integer_bits(q));
            for(const auto& q:b_grade.cell_scales)required.integer_bits=std::max(required.integer_bits,integer_bits(q));
            for(const auto& z:a_grade.lattice_coefficient)required.integer_bits=std::max(required.integer_bits,integer_bits(numeric::Rational{z}));
            for(const auto& z:b_grade.lattice_coefficient)required.integer_bits=std::max(required.integer_bits,integer_bits(numeric::Rational{z}));
            if(k) for(std::size_t i=0;i<n;++i) {
                const auto [begin,end]=a_grade.incidence_range(i);
                required.scalar_work=add(required.scalar_work,mul(end-begin,end-begin));
            }
        }
        for(const auto& link:carriers) {
            required.records=add(required.records,add(link.indices.size(),1));
            required.rational_slots=add(required.rational_slots,link.indices.size());
            required.sparse_entries=add(required.sparse_entries,link.indices.size());
            required.scalar_work=add(required.scalar_work,link.indices.size());
        }
        required.integer_bits=mul(required.integer_bits,4);
        runtime::admit_execution(required,policy);
        for(std::size_t k=0;k<maps.size();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
            std::vector<bool> used(maps[k].size(),false);
            for(std::size_t i=0;i<maps[k].size();++i) {
                const auto j=maps[k][i];
                if(j>=used.size() || used[j])throw std::invalid_argument("transport grade map is not a permutation");
                used[j]=true;
                if(x.grade(grade).cell_scales[i]!=y.grade(grade).cell_scales[j])
                    throw std::invalid_argument("transport changes retained lattice scales");
            }
        }
        morphism=std::make_unique<algebra::RelationalMorphism>(x,y,maps);
        for(std::size_t k=1;k<maps.size();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
            for(std::size_t i=0;i<maps[k].size();++i) {
                const auto [begin,end]=x.grade(grade).incidence_range(i);
                const auto target_begin=y.grade(grade).incidence_range(maps[k][i]).first;
                for(std::size_t p=begin;p<end;++p) {
                    if(morphism->occurrence_image(grade,p)!=target_begin+p-begin)
                        throw std::invalid_argument("transport changes retained occurrence order");
                }
            }
            lease.check_cancelled();
        }
        const auto old_carriers=storage::temporal_endpoint_detail::carriers(*from);
        const auto new_carriers=storage::temporal_endpoint_detail::carriers(*to);
        std::set<std::uint64_t> target_keys;
        for(const auto& link:carriers) {
            if(!old_carriers.contains(link.source) || !new_carriers.contains(link.target))
                throw std::invalid_argument("transport carrier is undeclared");
            const auto& a=*old_carriers.at(link.source);const auto& b=*new_carriers.at(link.target);
            const auto n=a.domain.logical_size();
            if(a.kind!=b.kind || a.domain!=numeric::IndexDomain::linear(n) || a.domain!=b.domain ||
               link.indices.size()!=n || a.kind==storage::TemporalCarrierKind::CellStalk || a.grade!=b.grade)
                throw std::invalid_argument("transport requires declared matching linear carriers");
            if(!links.emplace(link.source,link).second || !target_keys.insert(link.target).second)
                throw std::invalid_argument("transport carrier links must be distinct");
            if(a.kind==storage::TemporalCarrierKind::SourceGrade && link.indices!=maps.at(a.grade.value))
                throw std::invalid_argument("transport carrier map differs from its retained grade map");
            std::vector<bool> used(n,false);numeric::ExactMatrix matrix(n,n);
            for(std::size_t i=0;i<n;++i) {
                const auto j=link.indices[i];
                if(j>=n || used[j])throw std::invalid_argument("transport carrier map is not a permutation");
                used[j]=true;matrix.set(j,i,1);
            }
            matrices.emplace(link.source,std::move(matrix));
            lease.check_cancelled();
        }
    }
    const CarrierLink& link(std::uint64_t key) const {
        const auto found=links.find(key);
        if(found==links.end())throw std::invalid_argument("transport requires an explicit carrier link");
        return found->second;
    }
};
Transport::Transport(std::shared_ptr<State> state):state_(std::move(state)){}
TransportBuild Source::transport(const std::shared_ptr<Source>& target,
    const std::vector<std::vector<unsigned>>& maps,const std::vector<CarrierLink>& links,
    const std::vector<std::size_t>& limits,const std::shared_ptr<Cancellation>& cancellation) const {
    TransportBuild out;
    try {
        if(!target)throw std::invalid_argument("transport requires an owning destination source");
        const auto policy=policy_for(limits);
        const runtime::ExecutionLease lease(policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        auto state=std::make_shared<Transport::State>(state_,target->state_,maps,links,policy,lease);
        lease.check_cancelled();out.transport=std::shared_ptr<Transport>(new Transport(std::move(state)));
    } catch(const runtime::ExecutionCancelled& error){failure(out.status,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out.status,error);}
    return out;
}
TransportInfo Transport::info() const{return {state_->morphism->occurrence_maps()};}
ExecutionResult Transport::push(std::uint64_t carrier,const std::vector<std::string>& values,
    const std::shared_ptr<Cancellation>& cancellation) const {
    ExecutionResult out;
    try {
        using namespace inference::native_field_detail;
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();const auto& link=state_->link(carrier);
        if(values.size()!=link.indices.size())throw std::invalid_argument("transport values differ from their declared carrier");
        numeric::RationalVector input;input.reserve(values.size());std::size_t bits=0;
        for(const auto& value:values){input.push_back(coefficient(value));bits=std::max(bits,integer_bits(input.back()));}
        runtime::admit_execution({values.size(),mul(values.size(),2),values.size(),values.size(),bits},state_->policy);
        out.values=rationals(state_->matrices.at(carrier).apply(input));
        out.input_carrier=carrier;out.output_carrier=link.target;
        out.output_grade=storage::temporal_endpoint_detail::carriers(*state_->to).at(link.target)->grade.value;
        out.name_space=state_->target->manifest.name_space;out.sequence=state_->target->manifest.sequence;
        out.profile="native-bijection-transport-v1";out.implementation="RelationalMorphism/ExactMatrix/permutation-v1";
        out.integer_bits=bits;out.scalar_work=values.size();lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){failure(out,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out,error);}
    return out;
}
ExecutionResult Transport::check_tensor(std::uint64_t source_key,std::uint64_t target_key,
    const std::shared_ptr<Cancellation>& cancellation) const {
    ExecutionResult out;
    try {
        using namespace inference::native_field_detail;
        const runtime::ExecutionLease lease(state_->policy,cancellation?cancellation->state_->flag:std::shared_ptr<std::atomic_bool>{});
        lease.check_cancelled();
        const auto a=storage::temporal_endpoint_detail::tensors(*state_->from).at(source_key);
        const auto b=storage::temporal_endpoint_detail::tensors(*state_->to).at(target_key);
        const auto count=a->tensor.domain().logical_size();
        runtime::admit_execution({count,count,count,mul(count,4),std::max(tensor_bits(a->tensor),tensor_bits(b->tensor))},state_->policy);
        if(a->role!=b->role || a->interpretation!=b->interpretation || !a->output || !b->output ||
           a->owner.kind!=storage::TemporalOwnerKind::Global || b->owner.kind!=storage::TemporalOwnerKind::Global)
            throw std::invalid_argument("transport tensor role or global ownership differs");
        const auto& row=state_->link(*a->output);
        if(row.target!=*b->output)throw std::invalid_argument("transport tensor output carrier differs");
        const CarrierLink* col=nullptr;
        if(a->role==storage::TemporalTensorRole::Metric) {
            if(!a->input || !b->input)throw std::invalid_argument("transport metric input carrier absent");
            col=&state_->link(*a->input);
            if(col->target!=*b->input || a->tensor.domain()!=numeric::IndexDomain({row.indices.size(),col->indices.size()}))
                throw std::invalid_argument("transport metric axes differ");
        } else if(a->role!=storage::TemporalTensorRole::Field || a->input || b->input ||
                  a->tensor.domain()!=numeric::IndexDomain::linear(row.indices.size()))
            throw std::invalid_argument("transport supports stored fields and metric forms only");
        if(a->tensor.domain()!=b->tensor.domain())throw std::invalid_argument("transport tensor shape differs");
        for(std::size_t i=0;i<count;++i) {
            const auto j=col?row.indices[i/col->indices.size()]*col->indices.size()+col->indices[i%col->indices.size()]:row.indices[i];
            if(a->tensor.state(i)!=b->tensor.state(j))throw std::invalid_argument("transport tensor changes exact values or structural support");
        }
        lease.check_cancelled();
    } catch(const runtime::ExecutionCancelled& error){failure(out,error);}
      catch(const runtime::ExecutionLimitExceeded& error){failure(out,error);}
    return out;
}
std::string engine_digest(){return RFC_ENGINE_DIGEST;}
std::string engine_version(){return RFC_ENGINE_VERSION;}
std::string binding_digest(){return RFC_BINDING_DIGEST;}
} // namespace rexgraph::python
