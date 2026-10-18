#include "algebraic_bridge.hpp"
#include "rfc/database/prepared_algebraic_query.hpp"
#include "rfc/database/algebraic_graph_database.hpp"
#include "rfc/database/algebraic_grade_tensor_window.hpp"
#include "rfc/field/algebraic_metric_field.hpp"
#include "rfc/storage/algebraic_archive_codec.hpp"
#include "rfc/storage/algebraic_archive_history.hpp"
#include "rfc/storage/algebraic_archive_graph_codec.hpp"
#include <mutex>
#include <type_traits>

namespace rexgraph::python {
namespace {
using namespace rfc;
using Value = std::variant<local::AlgebraicModuleVector, local::AlgebraicCovector,
    local::AlgebraicModuleMap, field::AlgebraicMeasurementForm, field::AlgebraicModuleMetric,
    numeric::AlgebraicElement>;
std::string bytes(const std::vector<std::byte>& value) {
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::vector<std::string> coefficients(const numeric::RationalVector& values) {
    std::vector<std::string> result; result.reserve(values.size());
    for (const auto& value : values) result.push_back(value.get_str(16));
    return result;
}
AlgebraicCarrierInfo carrier(const local::AlgebraicModule& value) {
    return {true, value.cell().grade.value, value.cell().index, value.rank(), value.field().degree()};
}
storage::AlgebraicArchiveLimits limits(const std::vector<std::size_t>& v) {
    if (v.size() != 21) throw std::invalid_argument("algebraic archive requires 21 admission limits");
    storage::AlgebraicArchiveLimits out;
    out.max_file_bytes=v[0]; out.max_records=v[1]; out.max_rational_coordinates=v[2]; out.max_integer_bytes=v[3];
    out.field={v[4],v[5],v[6],v[7],v[8]}; out.modules={v[9],v[10],v[11],v[12]};
    out.fibers={v[13],v[14]}; out.real_root={v[15]}; out.complex_root={v[16],v[17],v[18]};
    out.metric={v[19],v[20]}; return out;
}
numeric::RationalVector exact_coordinates(const std::vector<std::string>& values) {
    numeric::RationalVector result; result.reserve(values.size());
    for (const auto& text : values) {
        numeric::Rational q;
        if (q.set_str(text,16)!=0 || q.get_den()==0) throw std::invalid_argument("invalid exact algebraic coordinate");
        q.canonicalize(); result.push_back(std::move(q));
    }
    return result;
}
const local::AlgebraicModuleVector& vector_value(const std::shared_ptr<AlgebraicValue>& value,
                                               const Value& data) {
    if (!value) throw std::invalid_argument("algebraic operation requires an owning value");
    const auto* result=std::get_if<local::AlgebraicModuleVector>(&data);
    if (!result) throw std::invalid_argument("algebraic operation requires a module vector");
    return *result;
}
}
struct AlgebraicArchive::State {
    storage::AlgebraicArchive archive;
    std::string image;
    std::shared_ptr<Source> history_owner;
    mutable std::mutex reconstruction_mutex;
    mutable std::uint64_t reconstruction_key{};
    mutable std::shared_ptr<const local::AlgebraicReconstruction> prepared_reconstruction;
    State(storage::AlgebraicArchive value, std::string raw) : archive(std::move(value)), image(std::move(raw)) {}
    std::shared_ptr<const local::AlgebraicReconstruction> reconstruction(std::uint64_t key) const {
        std::lock_guard lock(reconstruction_mutex);
        if (!prepared_reconstruction || reconstruction_key != key) {
            auto prepared=std::make_shared<const local::AlgebraicReconstruction>(archive.reconstruction(key));
            prepared_reconstruction=std::move(prepared); reconstruction_key=key;
        }
        return prepared_reconstruction;
    }
};
struct AlgebraicValue::State {
    Value value;
    explicit State(Value data) : value(std::move(data)) {}
};
AlgebraicArchive::AlgebraicArchive(std::shared_ptr<const State> state) : state_(std::move(state)) {}
AlgebraicValue::AlgebraicValue(std::shared_ptr<const State> state) : state_(std::move(state)) {}
std::vector<std::size_t> AlgebraicArchive::default_limits() {
    const storage::AlgebraicArchiveLimits p;
    return {p.max_file_bytes,p.max_records,p.max_rational_coordinates,p.max_integer_bytes,
        p.field.max_degree,p.field.max_input_coefficients,p.field.max_integer_bits,p.field.max_scalar_steps,p.field.max_matrix_coordinates,
        p.modules.max_cells,p.modules.max_rank,p.modules.max_retained_coordinates,p.modules.max_matrix_coordinates,
        p.fibers.max_occurrences,p.fibers.max_retained_map_coordinates,p.real_root.max_bisections,
        p.complex_root.max_refinements,p.complex_root.max_rounding_bits,p.complex_root.max_trace_coordinates,
        p.metric.max_sign_refinements,p.metric.max_factor_steps};
}
std::shared_ptr<AlgebraicArchive> AlgebraicArchive::load(const char* data, std::size_t size,
                                                      const std::vector<std::size_t>& bounds) {
    if (!data) throw std::invalid_argument("algebraic archive requires a byte buffer");
    const auto policy=limits(bounds);
    const auto decoded=storage::decode_algebraic_archive(
        {reinterpret_cast<const std::byte*>(data),size},policy);
    return std::shared_ptr<AlgebraicArchive>(new AlgebraicArchive(
        std::make_shared<const State>(decoded,std::string(data,size))));
}
std::shared_ptr<AlgebraicArchive> AlgebraicArchive::from_source(const std::shared_ptr<Source>& source,
    const std::vector<std::uint64_t>& channel, const std::vector<std::size_t>& bounds) {
    if (!source || channel.size()!=3) throw std::invalid_argument("archive history requires an owning endpoint and three channel names");
    const auto policy=limits(bounds);
    const auto pin=source->endpoint();
    const auto archive=storage::read_algebraic_archive(*pin,{channel[0],channel[1],channel[2]},policy);
    auto state=std::make_shared<State>(archive,bytes(storage::encode_algebraic_archive(archive,policy)));
    state->history_owner=source;
    return std::shared_ptr<AlgebraicArchive>(new AlgebraicArchive(std::move(state)));
}
AlgebraicArchiveInfo AlgebraicArchive::info() const {
    const auto& archive=state_->archive; const auto& field=archive.bundle().field();
    const auto& contents=archive.contents(); AlgebraicArchiveInfo out;
    out.archive=state_->image; out.source=bytes(storage::encode_rfcx(archive.bundle().source()).bytes);
    out.degree=field.degree(); out.polynomial=coefficients(field.minimal_polynomial());
    out.certificate_kind=static_cast<unsigned>(field.certificate().index());
    std::visit([&](const auto& certificate) {
        using T=std::decay_t<decltype(certificate)>;
        if constexpr (std::same_as<T,numeric::CyclotomicIrreducibility>) {
            out.certificate_order=certificate.order;
        } else {
            out.certificate_prime=certificate.prime;
            if constexpr (std::same_as<T,numeric::EisensteinShift>)
                out.certificate_shift=certificate.shift.get_str(16);
        }
    },field.certificate());
    out.has_involution=contents.involution.has_value();
    if (contents.involution) out.involution=coefficients(contents.involution->generator_image().coordinates());
    if (contents.embedding) std::visit([&](const auto& embedding) {
        using T=std::decay_t<decltype(embedding)>;
        if constexpr (std::same_as<T,geometry::SelectedRealAlgebraicEmbedding>) {
            const auto& interval=embedding.certificate().interval;
            out.embedding_kind=1; out.embedding=coefficients({interval.lower,interval.upper});
        } else {
            const auto& region=embedding.certificate().region;
            out.embedding_kind=2; out.embedding=coefficients({region.center.real,region.center.imaginary,region.radius});
        }
    },*contents.embedding);
    out.has_metric=contents.metric_domain.has_value(); out.has_fibers=contents.fibers.has_value();
    if (contents.fibers) {
        out.flat=contents.fibers->flat(); out.chain_policy=static_cast<unsigned>(contents.fibers->chain_policy());
    }
    for (std::size_t k=0;k<archive.bundle().source().grade_count();++k)
        out.source_sizes.push_back(archive.bundle().source().grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count());
    for (const auto& record : contents.records) {
        out.record_keys.push_back(record.key); out.record_kinds.push_back(static_cast<unsigned>(record.value.index()));
    }
    for (const auto& pair : contents.reconstructions) out.reconstruction_keys.push_back(pair.key);
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicArchive::record(std::uint64_t key) const {
    return std::visit([](const auto& value) {
        return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(Value{value})));
    },state_->archive.at(key));
}
std::shared_ptr<AlgebraicValue> AlgebraicArchive::vector(std::uint64_t key, unsigned side,
                                                      const std::vector<std::vector<std::string>>& rows) const {
    if (side>1) throw std::invalid_argument("unknown algebraic carrier side");
    const auto module=std::visit([&](const auto& value) -> local::AlgebraicModule {
        using T=std::decay_t<decltype(value)>;
        if constexpr (std::same_as<T,local::AlgebraicModuleMap>)
            return side==0 ? value.source() : value.target();
        else if constexpr (!std::same_as<T,numeric::AlgebraicElement>) return value.carrier();
        else throw std::invalid_argument("scalar record has no module carrier");
    },state_->archive.at(key));
    const auto& field=module.field();
    if (rows.size()!=module.rank()) throw std::invalid_argument("algebraic vector rank differs");
    std::vector<numeric::AlgebraicElement> values; values.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size()!=field.degree()) throw std::invalid_argument("algebraic vector requires complete polynomial coordinates");
        values.push_back(field.element(exact_coordinates(row)));
    }
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
        std::make_shared<const AlgebraicValue::State>(Value{module.vector(std::move(values))})));
}
std::shared_ptr<AlgebraicValue> AlgebraicArchive::scalar(const std::vector<std::string>& row) const {
    const auto& f=state_->archive.bundle().field();
    if (row.size()!=f.degree()) throw std::invalid_argument("algebraic scalar requires complete polynomial coordinates");
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
        std::make_shared<const AlgebraicValue::State>(Value{f.element(exact_coordinates(row))})));
}
AlgebraicValueInfo AlgebraicValue::info(std::size_t maximum) const {
    AlgebraicValueInfo out; out.kind=static_cast<unsigned>(state_->value.index());
    auto retain=[&](const auto& values) {
        out.coefficient_count=values.size();
        const auto count=std::min(maximum,values.size()); out.coefficients.reserve(count);
        for (std::size_t i=0;i<count;++i) out.coefficients.push_back(coefficients(values[i].coordinates()));
    };
    std::visit([&](const auto& value) {
        using T=std::decay_t<decltype(value)>;
        if constexpr (std::same_as<T,numeric::AlgebraicElement>) {
            out.coefficient_count=1;
            if (maximum) out.coefficients.push_back(coefficients(value.coordinates()));
        } else if constexpr (std::same_as<T,local::AlgebraicModuleMap>) {
            out.source=carrier(value.source()); out.target=carrier(value.target());
            out.rows=value.target().rank(); out.columns=value.source().rank(); retain(value.coefficients());
        } else {
            out.source=carrier(value.carrier()); out.target=out.source;
            if constexpr (std::same_as<T,local::AlgebraicModuleVector>) {
                out.rows=value.carrier().rank(); out.columns=1; retain(value.coordinates());
            } else if constexpr (std::same_as<T,local::AlgebraicCovector>) {
                out.rows=1; out.columns=value.carrier().rank(); retain(value.coefficients());
            } else if constexpr (std::same_as<T,field::AlgebraicMeasurementForm>) {
                out.rows=out.columns=value.carrier().rank(); retain(value.coefficient_form().coefficients());
            } else {
                out.rows=out.columns=value.carrier().rank(); retain(value.measurement().coefficient_form().coefficients());
            }
        }
    },state_->value);
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicValue::operation(unsigned operation,
                                                        const std::shared_ptr<AlgebraicValue>& input) const {
    if (!input) throw std::invalid_argument("algebraic operation requires an owning value");
    const auto& x=vector_value(input,input->state_->value);
    Value output=std::visit([&](const auto& value) -> Value {
        using T=std::decay_t<decltype(value)>;
        if constexpr (std::same_as<T,local::AlgebraicModuleMap>) {
            if (operation==0) return value.apply(x);
        } else if constexpr (std::same_as<T,local::AlgebraicCovector>) {
            if (operation==2) return value.evaluate(x);
        } else if constexpr (std::same_as<T,field::AlgebraicMeasurementForm>) {
            if (operation==0) return value.coefficient_form().apply(x);
            if (operation==3) return value.quadrance(x);
            if (operation==4) return value.as_covector(x);
        } else if constexpr (std::same_as<T,field::AlgebraicModuleMetric>) {
            if (operation==0) return value.apply(x);
            if (operation==1) return value.solve(x);
            if (operation==3) return value.quadrance(x);
            if (operation==4) return value.measurement().as_covector(x);
        }
        throw std::invalid_argument("operation is not defined for this algebraic record kind");
    },state_->value);
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const State>(std::move(output))));
}
std::shared_ptr<AlgebraicValue> AlgebraicArchive::reconstruction(std::uint64_t key, unsigned operation,
                                                              const std::shared_ptr<AlgebraicValue>& input) const {
    if (!input) throw std::invalid_argument("algebraic reconstruction requires an owning value");
    const auto& x=vector_value(input,input->state_->value); const auto pair=state_->reconstruction(key);
    auto result=operation==0 ? pair->accession().apply(x) : operation==1 ? pair->reconstruction().apply(x) :
        operation==2 ? pair->physical_projector().apply(x) :
        throw std::invalid_argument("unknown algebraic reconstruction operation");
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(Value{std::move(result)})));
}
struct AlgebraicGraph::State {
    storage::AlgebraicArchiveGraph graph;
    std::string image;
    storage::AlgebraicArchiveGraphLimits policy;
};
AlgebraicGraph::AlgebraicGraph(std::shared_ptr<const State> state) : state_(std::move(state)) {}
std::vector<std::size_t> AlgebraicGraph::default_limits() {
    const storage::AlgebraicArchiveGraphLimits p;
    auto result=AlgebraicArchive::default_limits();
    result.insert(result.end(),{p.max_file_bytes,p.max_owner_records,p.max_edges,p.max_rational_coordinates,
                               p.measurement.max_coefficient_products});
    return result;
}
std::shared_ptr<AlgebraicGraph> AlgebraicGraph::load(const char* data, std::size_t size,
                                                  const std::vector<std::size_t>& bounds) {
    if (!data || bounds.size()!=26) throw std::invalid_argument("algebraic graph requires bytes and 26 admission limits");
    storage::AlgebraicArchiveGraphLimits p;
    p.archive=limits({bounds.begin(),bounds.begin()+21}); p.max_file_bytes=bounds[21];
    p.max_owner_records=bounds[22]; p.max_edges=bounds[23]; p.max_rational_coordinates=bounds[24];
    p.measurement.max_coefficient_products=bounds[25];
    auto graph=storage::decode_algebraic_archive_graph({reinterpret_cast<const std::byte*>(data),size},p);
    return std::shared_ptr<AlgebraicGraph>(new AlgebraicGraph(
        std::make_shared<const State>(State{std::move(graph),std::string(data,size),p})));
}
AlgebraicGraphInfo AlgebraicGraph::info() const {
    AlgebraicGraphInfo out; out.image=state_->image;
    const auto& c=state_->graph.contents();
    auto retain=[&](const auto& rows) {
        std::vector<std::uint64_t> keys; keys.reserve(rows.size());
        for (const auto& row : rows) keys.push_back(row.key);
        out.keys.push_back(std::move(keys));
    };
    retain(c.fields); retain(c.involutions); retain(c.embeddings); retain(c.domains); retain(c.bundles);
    retain(c.fibers); retain(c.archives); retain(c.scalars); retain(c.carriers); retain(c.metric_comparisons);
    retain(c.reconstructions); retain(c.measurements); return out;
}
std::vector<std::uint64_t> AlgebraicGraph::archive_refs(std::uint64_t key) const {
    using namespace storage::algebraic_graph_detail;
    const auto& c=state_->graph.contents(); const auto& a=state_->graph.archive(key);
    return {owner_key(c.bundles,a.bundle()),owner_key(c.fields,a.bundle().field()),
            a.contents().metric_domain ? owner_key(c.domains,*a.contents().metric_domain) : 0};
}
std::shared_ptr<AlgebraicArchive> AlgebraicGraph::archive(std::uint64_t key) const {
    const auto& a=state_->graph.archive(key);
    const auto p=storage::algebraic_graph_codec_detail::aggregate_policy(state_->policy);
    return std::shared_ptr<AlgebraicArchive>(new AlgebraicArchive(
        std::make_shared<const AlgebraicArchive::State>(a,bytes(storage::encode_algebraic_archive(a,p)))));
}
AlgebraicComparisonInfo AlgebraicGraph::comparison(unsigned kind, std::uint64_t key, std::size_t maximum) const {
    using namespace storage::algebraic_graph_detail;
    const auto& g=state_->graph; const auto& c=g.contents(); AlgebraicComparisonInfo out;
    auto scalar=[&](const auto& s) {
        out.source_field=owner_key(c.fields,s.source()); out.target_field=owner_key(c.fields,s.target());
        out.scalar_key=owner_key(c.scalars,s);
    };
    auto retain=[&](const auto& values) {
        out.coefficient_count=values.size(); const auto n=std::min(maximum,values.size());
        for (std::size_t i=0;i<n;++i) out.coefficients.push_back(coefficients(values[i].coordinates()));
    };
    auto action=[&](const auto& a) {
        out.action_rows=a.rows(); out.action_columns=a.cols();
        const auto n=std::min(maximum,machine::checked_size_mul(a.rows(),a.cols()));
        for (std::size_t i=0;i<n;++i) out.action.push_back(a.get(i/a.cols(),i%a.cols()).get_str(16));
    };
    auto module=[&](const auto& m) {
        scalar(m.scalar_comparison()); out.source_bundle=bundle_key(c,m.source()); out.target_bundle=bundle_key(c,m.target());
        out.source=carrier(m.source()); out.target=carrier(m.target());
        retain(m.coefficients()); action(m.rational_action());
    };
    auto domains=[&](const auto& d) {
        scalar(d.scalar_comparison()); out.source_domain=owner_key(c.domains,d.source());
        out.target_domain=owner_key(c.domains,d.target());
    };
    if (kind==0) {
        const auto& s=g.scalars(key); scalar(s);
        retain(std::vector<numeric::AlgebraicElement>{s.generator_image()}); action(s.rational_action());
    } else if (kind==1) { module(g.carrier(key)); out.carrier_key=key;
    } else if (kind==2) {
        const auto& m=g.measurement(key); module(m.carrier()); domains(m.domains());
        const auto row=std::lower_bound(c.measurements.begin(),c.measurements.end(),key,[](const auto& r,auto k){return r.key<k;});
        out.carrier_key=row->carrier; out.source_archive=row->source_archive; out.target_archive=row->target_archive;
        out.source_record=row->source_record; out.target_record=row->target_record;
    } else if (kind==3) {
        const auto& r=g.reconstruction(key); module(r.accession());
        const auto row=std::lower_bound(c.reconstructions.begin(),c.reconstructions.end(),key,[](const auto& r0,auto k){return r0.key<k;});
        out.accession_key=row->accession; out.reconstruction_key=row->reconstruction;
    } else if (kind==4) { domains(at(c.metric_comparisons,key));
    } else throw std::invalid_argument("unknown algebraic graph comparison kind");
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicGraph::compare(unsigned kind, std::uint64_t key, unsigned operation,
    const std::shared_ptr<AlgebraicValue>& input, std::uint64_t target_archive) const {
    if (!input) throw std::invalid_argument("algebraic graph comparison requires an owning value");
    const auto& g=state_->graph; const auto& target=g.archive(target_archive);
    const auto& data=input->state_->value;
    Value result=[&]() -> Value {
        if (kind==0 && operation==0) {
            const auto* q=std::get_if<numeric::AlgebraicElement>(&data);
            if (!q) throw std::invalid_argument("scalar comparison requires an algebraic scalar");
            return g.scalars(key).apply(*q);
        }
        const auto& x=vector_value(input,data);
        if (kind==1 && operation==0) return g.carrier(key).apply(x);
        if (kind==2) {
            const auto& m=g.measurement(key); const auto y=m.carrier().apply(x);
            if (!target.contents().metric_domain || !target.contents().metric_domain->same_owner(m.domains().target()))
                throw std::invalid_argument("measurement comparison target domain owner differs");
            if (!target.bundle().stalk(m.target().carrier().cell()).same_owner(m.target().carrier()))
                throw std::invalid_argument("measurement comparison target carrier owner differs");
            if (operation==0) return y;
            if (operation==3) return m.target().quadrance(y);
        }
        if (kind==3) {
            const auto& r=g.reconstruction(key);
            if (operation==0) return r.accession().apply(x);
            if (operation==1) return r.reconstruction().apply(x);
            if (operation==2) return r.physical_projector().apply(x);
        }
        throw std::invalid_argument("operation is not defined for this graph comparison kind");
    }();
    std::visit([&](const auto& value) {
        using T=std::decay_t<decltype(value)>;
        if constexpr (std::same_as<T,numeric::AlgebraicElement>) {
            if (!value.field().same_owner(target.bundle().field()))
                throw std::invalid_argument("scalar comparison target field owner differs");
        } else if constexpr (std::same_as<T,local::AlgebraicModuleVector>) {
            if (!value.carrier().same_owner(target.bundle().stalk(value.carrier().cell())))
                throw std::invalid_argument("comparison target carrier owner differs");
        } else throw std::invalid_argument("graph comparisons return only scalars or vectors");
    },result);
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(std::move(result))));
}

struct AlgebraicExecutionAccess {
    static runtime::ExecutionLease phase_lease(const std::shared_ptr<Cancellation>& token) {
        return runtime::ExecutionLease(runtime::ExecutionHostPolicy{},token?token->flag():std::shared_ptr<const std::atomic_bool>{});
    }
    static runtime::ExecutionLease lease(const std::vector<std::size_t>& bounds,
                                         const std::shared_ptr<Cancellation>& token) {
        if (bounds.size()!=5) throw std::invalid_argument("execution policy requires five resource limits");
        runtime::ExecutionHostPolicy policy;
        policy.limits={bounds[0],bounds[1],bounds[2],bounds[3],bounds[4]};
        return runtime::ExecutionLease(policy,token?token->flag():std::shared_ptr<const std::atomic_bool>{});
    }
};
namespace {
std::vector<std::size_t> requirements(const runtime::ExecutionRequirements& r) {
    return {r.records,r.rational_slots,r.sparse_entries,r.scalar_work,r.integer_bits};
}
template<class Function> void query_status(ExecutionResult& status,const Function& function) {
    try { function(); }
    catch (const runtime::ExecutionCancelled& error) { status.error=1;status.error_message=error.what(); }
    catch (const runtime::ExecutionLimitExceeded& error) {
        status.error=2;status.error_message=error.what();status.error_resource=error.resource();
        status.error_required=error.required();status.error_limit=error.limit();
    }
}
local::AlgebraicModule selected_carrier(const Value& data,bool target) {
    return std::visit([&](const auto& value) -> local::AlgebraicModule {
        using T=std::decay_t<decltype(value)>;
        if constexpr (std::same_as<T,local::AlgebraicModuleMap>) return target?value.target():value.source();
        else if constexpr (!std::same_as<T,numeric::AlgebraicElement>) return value.carrier();
        else throw std::invalid_argument("query carrier requires a module value");
    },data);
}
}
struct AlgebraicSnapshot::State {
    database::AlgebraicSnapshot snapshot;
    explicit State(database::AlgebraicSnapshot value):snapshot(std::move(value)) {}
};
struct AlgebraicGraphRevision::State {
    storage::AlgebraicGraphRevision revision;
    storage::AlgebraicGraphSnapshotLimits policy;
};
AlgebraicGraphRevision::AlgebraicGraphRevision(std::shared_ptr<const State> s):state_(std::move(s)) {}
std::vector<std::size_t> AlgebraicGraphRevision::default_limits() {
    const storage::AlgebraicGraphSnapshotLimits p;
    auto result=AlgebraicGraph::default_limits();
    result.insert(result.end(),{p.max_file_bytes,p.max_source_bindings,p.max_archive_bindings,
                               p.max_source_coordinates,p.max_history_revisions});
    return result;
}
std::shared_ptr<AlgebraicGraphRevision> AlgebraicGraphRevision::capture(const std::shared_ptr<Source>& source,
    const std::vector<std::uint64_t>& channel,const std::vector<std::size_t>& bounds) {
    if (!source || channel.size()!=3 || bounds.size()!=31)
        throw std::invalid_argument("graph revision requires a selected source, three channel names and 31 limits");
    storage::AlgebraicGraphSnapshotLimits p;
    p.graph.archive=limits({bounds.begin(),bounds.begin()+21});p.graph.max_file_bytes=bounds[21];
    p.graph.max_owner_records=bounds[22];p.graph.max_edges=bounds[23];p.graph.max_rational_coordinates=bounds[24];
    p.graph.measurement.max_coefficient_products=bounds[25];p.max_file_bytes=bounds[26];
    p.max_source_bindings=bounds[27];p.max_archive_bindings=bounds[28];p.max_source_coordinates=bounds[29];p.max_history_revisions=bounds[30];
    storage::algebraic_archive_detail::bound(source->retained_revisions(),p.max_history_revisions,"native graph prefix revision budget");
    auto revision=storage::AlgebraicGraphRevision::capture(source->native_endpoint(),{channel[0],channel[1],channel[2]},p);
    return std::shared_ptr<AlgebraicGraphRevision>(new AlgebraicGraphRevision(
        std::make_shared<const State>(State{std::move(revision),p})));
}
AlgebraicGraphRevisionInfo AlgebraicGraphRevision::info() const {
    const auto& r=state_->revision;const auto& snapshot=r.snapshot();
    AlgebraicGraphRevisionInfo out;out.name_space=r.source_namespace();out.sequence=r.sequence();out.anchor=snapshot.anchor_bundle();
    out.image=bytes(storage::encode_algebraic_graph_snapshot(snapshot,state_->policy));
    for (const auto& row:snapshot.source_bindings()) {
        AlgebraicGraphSourceInfo binding;binding.bundle=row.bundle;binding.kind=static_cast<unsigned>(row.kind);
        for (const auto& grade:row.cells) binding.cells.emplace_back(grade.begin(),grade.end());
        binding.occurrences=row.occurrences;out.sources.push_back(std::move(binding));
    }
    for (const auto& row:snapshot.archive_bindings()) out.archives.push_back({row.archive,row.bundle});
    return out;
}
std::shared_ptr<AlgebraicGraph> AlgebraicGraphRevision::graph() const {
    const auto& graph=state_->revision.snapshot().graph();
    return std::shared_ptr<AlgebraicGraph>(new AlgebraicGraph(std::make_shared<const AlgebraicGraph::State>(
        AlgebraicGraph::State{graph,bytes(storage::encode_algebraic_archive_graph(graph,state_->policy.graph)),state_->policy.graph})));
}
bool AlgebraicGraphRevision::same_owner(const std::shared_ptr<AlgebraicGraphRevision>& other) const {
    return other && state_->revision.same_owner(other->state_->revision);
}
struct AlgebraicGraphDatabase::State {
    database::AlgebraicGraphDatabase database;
    explicit State(database::AlgebraicGraphDatabase value):database(std::move(value)) {}
};
AlgebraicGraphDatabase::AlgebraicGraphDatabase(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicGraphDatabaseBuild AlgebraicGraphDatabase::select(const std::shared_ptr<AlgebraicGraphRevision>& revision,
    std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) {
    if (!revision) throw std::invalid_argument("graph database requires an owning revision");
    AlgebraicGraphDatabaseBuild out;
    query_status(out.status,[&] {
        auto value=database::AlgebraicGraphDatabase::select(revision->state_->revision,{ns,cutoff},AlgebraicExecutionAccess::lease(bounds,token));
        out.database=std::shared_ptr<AlgebraicGraphDatabase>(new AlgebraicGraphDatabase(std::make_shared<const State>(std::move(value))));
    });
    return out;
}
AlgebraicGraphArchiveSelection AlgebraicGraphDatabase::archive(std::uint64_t key,std::uint64_t ns,std::uint64_t cutoff,
    const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) const {
    AlgebraicGraphArchiveSelection out;
    query_status(out.status,[&] {
        auto value=state_->database.archive(key,{ns,cutoff},AlgebraicExecutionAccess::lease(bounds,token));
        out.snapshot=std::shared_ptr<AlgebraicSnapshot>(new AlgebraicSnapshot(std::make_shared<const AlgebraicSnapshot::State>(std::move(value))));
    });
    return out;
}
std::vector<std::size_t> AlgebraicGraphDatabase::requirements() const {
    return rexgraph::python::requirements(state_->database.requirements());
}

struct AlgebraicRecord::State {
    database::AlgebraicRecordPin pin;
    explicit State(database::AlgebraicRecordPin value):pin(std::move(value)) {}
};
struct AlgebraicWindow::State {
    database::AlgebraicTensorWindow window;
    explicit State(database::AlgebraicTensorWindow value):window(std::move(value)) {}
};
struct AlgebraicQuery::State {
    database::PreparedAlgebraicQuery query;
    explicit State(database::PreparedAlgebraicQuery value):query(std::move(value)) {}
};
struct AlgebraicQueryResult::State {
    database::DerivedAlgebraicQuery result;
    explicit State(database::DerivedAlgebraicQuery value):result(std::move(value)) {}
};
struct AlgebraicQueryCache::State {
    database::AlgebraicQueryCache cache;
    explicit State(std::size_t records):cache(records) {}
};
AlgebraicSnapshot::AlgebraicSnapshot(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicRecord::AlgebraicRecord(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicWindow::AlgebraicWindow(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicQuery::AlgebraicQuery(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicQueryResult::AlgebraicQueryResult(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicQueryCache::AlgebraicQueryCache(std::size_t records):state_(std::make_shared<State>(records)) {}
std::shared_ptr<AlgebraicSnapshot> AlgebraicSnapshot::publish(const std::shared_ptr<AlgebraicArchive>& archive,
                                                            std::uint64_t ns,std::uint64_t sequence) {
    if (!archive) throw std::invalid_argument("snapshot requires an owning algebraic archive");
    auto snapshot=database::AlgebraicSnapshot::publish(archive->state_->archive,ns,sequence);
    return std::shared_ptr<AlgebraicSnapshot>(new AlgebraicSnapshot(std::make_shared<const State>(std::move(snapshot))));
}
bool AlgebraicSnapshot::same_publication(const std::shared_ptr<AlgebraicSnapshot>& other) const {
    return other && state_->snapshot.same_publication(other->state_->snapshot);
}
std::vector<std::uint64_t> AlgebraicSnapshot::graph_origin() const {
    const auto origin=state_->snapshot.graph_origin();
    if (!origin) return {};
    if (state_->snapshot.source_publication()!=origin->revision.source_publication())
        throw std::logic_error("graph archive lost its selected endpoint pin");
    return {origin->archive_key,origin->bundle_key,origin->revision.snapshot().anchor_bundle(),
            origin->revision.source_namespace(),origin->revision.sequence()};
}
std::shared_ptr<AlgebraicRecord> AlgebraicSnapshot::record(std::uint64_t key) const {
    return std::shared_ptr<AlgebraicRecord>(new AlgebraicRecord(
        std::make_shared<const AlgebraicRecord::State>(state_->snapshot.record(key))));
}
AlgebraicOccurrenceInfo AlgebraicSnapshot::occurrence(std::uint16_t grade,std::size_t slot) const {
    const auto occurrence=state_->snapshot.occurrence(core::Grade{grade},slot);
    const auto wrap=[](local::AlgebraicModuleMap map) {
        return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
            std::make_shared<const AlgebraicValue::State>(Value{std::move(map)})));
    };
    return {occurrence.key.value,occurrence.coefficient.get_str(16),occurrence.active,
            wrap(occurrence.underlying),wrap(occurrence.weighted_map()),wrap(occurrence.contribution_map())};
}
AlgebraicRecordInfo AlgebraicRecord::info(std::size_t maximum) const {
    const auto& pin=state_->pin;AlgebraicRecordInfo out;
    out.key=pin.key();out.domain=pin.domain().extents();auto& v=out.value;
    v.kind=static_cast<unsigned>(pin.kind());v.source=carrier(pin.source_carrier());v.target=carrier(pin.target_carrier());
    v.rows=out.domain[0];v.columns=out.domain.size()==1?1:out.domain[1];
    v.coefficient_count=pin.coefficients().size();
    const auto count=std::min(maximum,v.coefficient_count);v.coefficients.reserve(count);
    for (std::size_t i=0;i<count;++i) v.coefficients.push_back(coefficients(pin.coefficients()[i].coordinates()));
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicRecord::value() const {
    return std::visit([](const auto& v) {
        return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(Value{v})));
    },state_->pin.value());
}
std::shared_ptr<AlgebraicValue> AlgebraicRecord::coordinate(const std::vector<std::size_t>& position) const {
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
        std::make_shared<const AlgebraicValue::State>(Value{state_->pin.coordinate(position)})));
}
AlgebraicWindowBuild AlgebraicWindow::prepare(const std::shared_ptr<AlgebraicRecord>& record,
    const std::vector<std::size_t>& lower,const std::vector<std::size_t>& upper,
    std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) {
    if (!record) throw std::invalid_argument("window requires an owning algebraic record pin");
    AlgebraicWindowBuild out;
    query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::lease(bounds,token);
        auto window=database::AlgebraicTensorWindow::prepare(record->state_->pin,lower,upper,{ns,cutoff},lease);
        out.window=std::shared_ptr<AlgebraicWindow>(new AlgebraicWindow(std::make_shared<const State>(std::move(window))));
    });return out;
}
AlgebraicWindowInfo AlgebraicWindow::info() const {
    const auto& w=state_->window;
    return {w.domain().extents(),w.source_domain().extents(),w.lower(),w.upper(),requirements(w.requirements()),w.supported_count()};
}
AlgebraicWindowValues AlgebraicWindow::values(std::size_t maximum) const {
    AlgebraicWindowValues out;const auto& w=state_->window;
    const auto count=std::min(maximum,w.supported_count());
    out.coefficients.reserve(count);out.source_indices.reserve(count);
    for (std::size_t i=0;i<count;++i) {
        const auto value=w.coordinate(i);
        out.coefficients.push_back(coefficients(value.coordinates()));
        out.source_indices.push_back(w.source_index(i));
    }return out;
}
std::size_t AlgebraicWindow::source_index(std::size_t coordinate) const {return state_->window.source_index(coordinate);}
std::vector<std::size_t> AlgebraicWindow::window_index(std::size_t coordinate) const {
    const auto result=state_->window.window_index(coordinate);
    return result?std::vector<std::size_t>{*result}:std::vector<std::size_t>{};
}
std::shared_ptr<AlgebraicValue> AlgebraicWindow::coordinate(std::size_t coordinate) const {
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
        std::make_shared<const AlgebraicValue::State>(Value{state_->window.coordinate(coordinate)})));
}
AlgebraicSum AlgebraicWindow::sum(std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,
                                 const std::shared_ptr<Cancellation>& token) const {
    AlgebraicSum out;query_status(out.status,[&] {
        const auto result=state_->window.sum({ns,cutoff},AlgebraicExecutionAccess::lease(bounds,token));
        if (result) out.value=std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
            std::make_shared<const AlgebraicValue::State>(Value{*result})));
    });return out;
}
struct AlgebraicGradeTensor::State {
    database::AlgebraicGradeTensor tensor;
    explicit State(database::AlgebraicGradeTensor value):tensor(std::move(value)) {}
};
struct AlgebraicGradeWindow::State {
    database::AlgebraicGradeTensorWindow window;
    std::vector<std::size_t> lower,upper;
};
AlgebraicGradeTensor::AlgebraicGradeTensor(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicGradeWindow::AlgebraicGradeWindow(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicGradeBuild AlgebraicGradeTensor::select(const std::shared_ptr<AlgebraicSnapshot>& snapshot,std::uint16_t grade,
    const std::vector<std::uint64_t>& keys,std::uint64_t ns,std::uint64_t cutoff,
    const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) {
    if (!snapshot) throw std::invalid_argument("grade tensor requires an owning publication");
    AlgebraicGradeBuild out;query_status(out.status,[&] {
        auto tensor=database::AlgebraicGradeTensor::select(snapshot->state_->snapshot,core::Grade{grade},keys,{ns,cutoff},
            AlgebraicExecutionAccess::lease(bounds,token));
        out.tensor=std::shared_ptr<AlgebraicGradeTensor>(new AlgebraicGradeTensor(std::make_shared<const State>(std::move(tensor))));
    });return out;
}
AlgebraicGradeInfo AlgebraicGradeTensor::info() const {
    const auto& t=state_->tensor;return {t.grade().value,t.derived(),t.cell_count(),t.domain().logical_size(),
        t.record_keys(),t.stalk_offsets(),requirements(t.requirements())};
}
std::vector<std::size_t> AlgebraicGradeTensor::source_coordinate(std::size_t position) const {
    const auto c=state_->tensor.source_coordinate(position);return {c.cell.grade.value,c.cell.index,c.coordinate};
}
std::size_t AlgebraicGradeTensor::source_index(std::uint16_t grade,std::uint32_t cell,std::size_t position) const {
    return state_->tensor.source_index({{core::Grade{grade},cell},position});
}
AlgebraicGradeValues AlgebraicGradeTensor::values(std::size_t maximum) const {
    AlgebraicGradeValues out;const auto n=std::min(maximum,state_->tensor.domain().logical_size());
    out.coefficients.reserve(n);out.coordinates.reserve(n);
    for (std::size_t i=0;i<n;++i) {const auto value=state_->tensor.coordinate(i);out.coefficients.push_back(coefficients(value.coordinates()));out.coordinates.push_back(source_coordinate(i));}
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicGradeTensor::coordinate(std::size_t position) const {
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(state_->tensor.coordinate(position))));
}
std::shared_ptr<AlgebraicValue> AlgebraicGradeTensor::stalk(std::uint32_t cell) const {
    const auto pin=state_->tensor.stalk_pin({state_->tensor.grade(),cell});
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(*pin)));
}
bool AlgebraicGradeTensor::same_owner(const std::shared_ptr<AlgebraicGradeTensor>& other) const {
    return other && state_->tensor.same_owner(other->state_->tensor);
}
bool AlgebraicGradeTensor::matches_source(const std::shared_ptr<AlgebraicSnapshot>& source) const {
    return source && state_->tensor.source().same_publication(source->state_->snapshot);
}
AlgebraicGradeBuild AlgebraicGradeTensor::boundary(std::uint64_t ns,std::uint64_t cutoff,
    const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) const {
    AlgebraicGradeBuild out;query_status(out.status,[&] {
        auto tensor=state_->tensor.boundary({ns,cutoff},AlgebraicExecutionAccess::lease(bounds,token));
        if (!tensor.input() || !tensor.input()->same_owner(state_->tensor)) throw std::logic_error("grade boundary lost input owner");
        out.tensor=std::shared_ptr<AlgebraicGradeTensor>(new AlgebraicGradeTensor(std::make_shared<const State>(std::move(tensor))));
    });return out;
}
namespace {
void metric_admit(const database::AlgebraicSnapshot& snapshot,std::uint64_t ns,std::uint64_t cutoff,
    const runtime::ExecutionLease& lease) {
    lease.check_cancelled();runtime::admit_work_availability(snapshot.source_namespace(),snapshot.sequence(),runtime::WorkAvailability{ns,cutoff});
}
field::AlgebraicFiberMetricField::Values metric_inputs(const database::AlgebraicGradeTensor& tensor,
    const database::AlgebraicSnapshot& snapshot,const runtime::ExecutionLease& lease) {
    if (!tensor.source().same_publication(snapshot)) throw std::invalid_argument("metric input publication owner differs");
    field::AlgebraicFiberMetricField::Values values;values.reserve(tensor.cell_count());
    for (std::size_t i=0;i<tensor.cell_count();++i) {
        lease.check_cancelled();values.push_back(*tensor.stalk_pin({tensor.grade(),static_cast<std::uint32_t>(i)}));
    }
    return values;
}
}
struct AlgebraicMetricField::State {
    database::AlgebraicSnapshot snapshot;
    field::AlgebraicFiberMetricField field;
};
struct AlgebraicMetricValues::State {
    database::AlgebraicSnapshot snapshot;
    field::AlgebraicFiberMetricField field;
    database::AlgebraicGradeTensor input;
    core::Grade grade;
    field::AlgebraicFiberMetricField::Values values;
};
AlgebraicMetricField::AlgebraicMetricField(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicMetricValues::AlgebraicMetricValues(std::shared_ptr<const State> s):state_(std::move(s)) {}
AlgebraicMetricBuild AlgebraicMetricField::prepare(const std::shared_ptr<AlgebraicSnapshot>& source,
    const std::vector<std::uint64_t>& keys,std::size_t maximum,std::uint64_t ns,std::uint64_t cutoff,
    const std::shared_ptr<Cancellation>& token) {
    if (!source) throw std::invalid_argument("metric field requires an owning publication");
    AlgebraicMetricBuild out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::phase_lease(token);
        const auto& snapshot=source->state_->snapshot;metric_admit(snapshot,ns,cutoff,lease);
        const auto& contents=snapshot.archive().contents();
        if (!contents.fibers || !contents.metric_domain) throw std::invalid_argument("metric field requires declared fibers and metric domain");
        std::size_t count{};
        const auto& native_source=snapshot.archive().bundle().source();
        for (std::size_t k=0;k<native_source.grade_count();++k) count=machine::checked_size_add(count,
            native_source.grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count());
        if (keys.size()!=count) throw std::invalid_argument("metric field requires one explicit metric record per retained cell");
        std::vector<field::AlgebraicModuleMetric> metrics;metrics.reserve(count);
        for (const auto key:keys) {
            lease.check_cancelled();const auto pin=snapshot.record(key);
            const auto* metric=std::get_if<field::AlgebraicModuleMetric>(&pin.value());
            if (!metric) throw std::invalid_argument("metric field record is not a certified metric");
            metrics.push_back(*metric);
        }
        auto field=field::AlgebraicFiberMetricField::declare(*contents.fibers,*contents.metric_domain,metrics,{maximum});
        lease.check_cancelled();out.field=std::shared_ptr<AlgebraicMetricField>(new AlgebraicMetricField(
            std::make_shared<const State>(State{snapshot,std::move(field)})));
    });return out;
}
AlgebraicMetricAction AlgebraicMetricField::apply(const std::shared_ptr<AlgebraicGradeTensor>& tensor,unsigned operation,
    std::uint64_t ns,std::uint64_t cutoff,const std::shared_ptr<Cancellation>& token) const {
    if (!tensor) throw std::invalid_argument("metric action requires an owning grade tensor");
    AlgebraicMetricAction out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::phase_lease(token);
        metric_admit(state_->snapshot,ns,cutoff,lease);const auto& input=tensor->state_->tensor;
        auto values=metric_inputs(input,state_->snapshot,lease);auto grade=input.grade();
        if (operation==0) values=state_->field.hodge(grade,values);
        else if (operation==1) {
            if (std::size_t(grade.value)+1>=state_->field.system().bundle().source().grade_count())
                throw std::out_of_range("top grade has no coboundary target");
            grade=core::Grade{static_cast<std::uint16_t>(grade.value+1u)};values=state_->field.coboundary(grade,values);
        } else throw std::invalid_argument("unknown whole-grade metric action");
        lease.check_cancelled();out.values=std::shared_ptr<AlgebraicMetricValues>(new AlgebraicMetricValues(
            std::make_shared<const AlgebraicMetricValues::State>(AlgebraicMetricValues::State{
                state_->snapshot,state_->field,input,grade,std::move(values)})));
    });return out;
}
AlgebraicSum AlgebraicMetricField::scalar(const std::shared_ptr<AlgebraicGradeTensor>& tensor,
    const std::shared_ptr<AlgebraicGradeTensor>& other,unsigned operation,std::uint64_t ns,std::uint64_t cutoff,
    const std::shared_ptr<Cancellation>& token) const {
    if (!tensor) throw std::invalid_argument("metric scalar action requires an owning grade tensor");
    AlgebraicSum out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::phase_lease(token);
        metric_admit(state_->snapshot,ns,cutoff,lease);const auto& input=tensor->state_->tensor;
        auto values=metric_inputs(input,state_->snapshot,lease);
        const auto result=[&] {
            if (operation==1 && !other) return state_->field.adjacent_quadrance(input.grade(),values);
            if (operation==0 && other) {
                if (other->state_->tensor.grade()!=input.grade()) throw std::invalid_argument("metric pairing grades differ");
                auto right=metric_inputs(other->state_->tensor,state_->snapshot,lease);
                return state_->field.inner(input.grade(),values,right);
            }
            throw std::invalid_argument("unknown whole-grade metric scalar action");
        }();
        lease.check_cancelled();out.value=std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
            std::make_shared<const AlgebraicValue::State>(result)));
    });return out;
}
AlgebraicGradeInfo AlgebraicMetricValues::info() const {
    const auto offsets=state_->field.system().bundle().offsets(state_->grade);
    return {state_->grade.value,true,state_->values.size(),offsets.back(),{},offsets,{}};
}
AlgebraicGradeValues AlgebraicMetricValues::values(std::size_t maximum) const {
    AlgebraicGradeValues out;const auto n=std::min(maximum,info().coordinates);
    out.coefficients.reserve(n);out.coordinates.reserve(n);
    for (std::size_t i=0;i<state_->values.size();++i) {
        const auto& value=state_->values[i];
        for (std::size_t j=0;j<value.coordinates().size() && out.coefficients.size()<n;++j) {
            out.coefficients.push_back(coefficients(value.coordinates()[j].coordinates()));
            out.coordinates.push_back({state_->grade.value,i,j});
        }
        if (out.coefficients.size()==n) break;
    }
    return out;
}
std::shared_ptr<AlgebraicValue> AlgebraicMetricValues::stalk(std::uint32_t cell) const {
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(
        std::make_shared<const AlgebraicValue::State>(state_->values.at(cell))));
}
AlgebraicGradeWindowBuild AlgebraicGradeWindow::prepare(const std::shared_ptr<AlgebraicGradeTensor>& tensor,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& lower,const std::vector<std::size_t>& upper,
    std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,const std::shared_ptr<Cancellation>& token) {
    if (!tensor) throw std::invalid_argument("grade window requires an owning field");
    AlgebraicGradeWindowBuild out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::lease(bounds,token);lease.check_cancelled();
        runtime::ExecutionRequirements required;required.records=machine::checked_size_add(1,machine::checked_size_mul(4,shape.size()));
        runtime::admit_execution(required,lease.policy());
        auto window=database::AlgebraicGradeTensorWindow::prepare(tensor->state_->tensor,numeric::IndexDomain(shape),lower,upper,{ns,cutoff},lease);
        if (!window.tensor().same_owner(tensor->state_->tensor)) throw std::logic_error("grade window changed input owner");
        out.window=std::shared_ptr<AlgebraicGradeWindow>(new AlgebraicGradeWindow(std::make_shared<const State>(State{std::move(window),lower,upper})));
    });return out;
}
AlgebraicWindowInfo AlgebraicGradeWindow::info() const {
    const auto& w=state_->window;return {w.domain().extents(),w.source_domain().extents(),state_->lower,state_->upper,
        requirements(w.requirements()),w.supported_count()};
}
AlgebraicGradeValues AlgebraicGradeWindow::values(std::size_t maximum) const {
    AlgebraicGradeValues out;const auto n=std::min(maximum,state_->window.supported_count());
    out.coefficients.reserve(n);out.coordinates.reserve(n);
    for (std::size_t i=0;i<n;++i) {const auto value=state_->window.coordinate(i);out.coefficients.push_back(coefficients(value.coordinates()));out.coordinates.push_back(source_coordinate(i));}
    return out;
}
std::size_t AlgebraicGradeWindow::source_index(std::size_t position) const {return state_->window.source_index(position);}
std::vector<std::size_t> AlgebraicGradeWindow::source_coordinate(std::size_t position) const {
    const auto c=state_->window.source_coordinate(position);return {c.cell.grade.value,c.cell.index,c.coordinate};
}
std::vector<std::size_t> AlgebraicGradeWindow::window_index(std::size_t position) const {
    const auto index=state_->window.window_index(position);return index?std::vector<std::size_t>{*index}:std::vector<std::size_t>{};
}
std::shared_ptr<AlgebraicValue> AlgebraicGradeWindow::coordinate(std::size_t position) const {
    return std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(state_->window.coordinate(position))));
}
AlgebraicSum AlgebraicGradeWindow::sum(std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,
    const std::shared_ptr<Cancellation>& token) const {
    AlgebraicSum out;query_status(out.status,[&] {
        const auto value=state_->window.sum({ns,cutoff},AlgebraicExecutionAccess::lease(bounds,token));
        if (value) out.value=std::shared_ptr<AlgebraicValue>(new AlgebraicValue(std::make_shared<const AlgebraicValue::State>(*value)));
    });return out;
}

AlgebraicQueryBuild AlgebraicQuery::prepare(const std::shared_ptr<AlgebraicSnapshot>& snapshot,int kind,
    const std::shared_ptr<AlgebraicValue>& source,const std::shared_ptr<AlgebraicValue>& target,
    const std::vector<AlgebraicPredicate>& predicates,std::uint64_t ns,std::uint64_t cutoff,
    const std::vector<std::size_t>& bounds,std::size_t rows_per_tile,const std::shared_ptr<Cancellation>& token) {
    if (!snapshot) throw std::invalid_argument("query requires an owning algebraic snapshot");
    if (kind < -1 || kind > 4) throw std::invalid_argument("unknown algebraic record kind");
    AlgebraicQueryBuild out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::lease(bounds,token);lease.check_cancelled();
        const auto& native=snapshot->state_->snapshot;
        runtime::admit_work_availability(native.source_namespace(),native.sequence(),runtime::WorkAvailability{ns,cutoff});
        const auto rows=native.archive().contents().records.size(),clauses=predicates.size();
        runtime::ExecutionRequirements required;
        required.records=machine::checked_size_add(machine::checked_size_mul(6,rows),machine::checked_size_mul(rows,clauses));
        required.records=machine::checked_size_add(required.records,clauses);
        required.rational_slots=machine::checked_size_mul(clauses,native.archive().bundle().field().degree());
        required.scalar_work=machine::checked_size_mul(machine::checked_size_mul(rows,clauses),
            machine::checked_size_add(native.archive().bundle().field().degree(),2));
        for (const auto& p:predicates) required.records=machine::checked_size_add(required.records,p.coordinate.size());
        runtime::admit_execution(required,lease.policy());
        database::AlgebraicQueryRequest request;
        if (kind>=0) request.kind=static_cast<database::AlgebraicRecordKind>(kind);
        if (source) request.source_carrier=selected_carrier(source->state_->value,false);
        if (target) request.target_carrier=selected_carrier(target->state_->value,true);
        request.all_of.reserve(clauses);
        for (const auto& p:predicates) {
            lease.check_cancelled();
            if (!p.literal) throw std::invalid_argument("algebraic predicate requires an owning scalar literal");
            const auto* literal=std::get_if<numeric::AlgebraicElement>(&p.literal->state_->value);
            if (!literal) throw std::invalid_argument("algebraic predicate requires a scalar literal");
            request.all_of.push_back({p.coordinate,static_cast<database::AlgebraicEquality>(p.operation),*literal});
        }
        auto query=database::PreparedAlgebraicQuery::prepare(native,request,{ns,cutoff},lease,rows_per_tile);
        out.query=std::shared_ptr<AlgebraicQuery>(new AlgebraicQuery(std::make_shared<const State>(std::move(query))));
    });return out;
}
AlgebraicQueryInfo AlgebraicQuery::info() const {
    return {state_->query.candidate_count(),requirements(state_->query.requirements())};
}
AlgebraicQueryExecution AlgebraicQuery::execute(std::uint64_t ns,std::uint64_t cutoff,const std::vector<std::size_t>& bounds,
    std::size_t workers,std::size_t worker_limit,std::size_t tile_limit,const std::shared_ptr<Cancellation>& token,
    const std::shared_ptr<AlgebraicQueryCache>& cache) const {
    AlgebraicQueryExecution out;query_status(out.status,[&] {
        const auto lease=AlgebraicExecutionAccess::lease(bounds,token);lease.check_cancelled();
        runtime::CpuTileExecutor executor({workers,worker_limit,tile_limit});
        auto result=cache?cache->state_->cache.execute(state_->query,{ns,cutoff},executor,lease):
                          state_->query.execute({ns,cutoff},executor,lease);
        out.result=std::shared_ptr<AlgebraicQueryResult>(new AlgebraicQueryResult(
            std::make_shared<const AlgebraicQueryResult::State>(std::move(result))));
    });return out;
}
AlgebraicQueryResultInfo AlgebraicQueryResult::info() const {
    const auto& r=state_->result;AlgebraicQueryResultInfo out;
    out.name_space=r.availability().source_namespace;out.max_sequence=r.availability().max_sequence;out.cache_hit=r.cache_hit();
    out.keys.reserve(r.records().size());for (const auto& record:r.records()) out.keys.push_back(record.key());
    const auto cpu=r.cpu_report();out.cpu={cpu.requested_workers,cpu.worker_slots,cpu.launched_threads,cpu.tiles,cpu.completed_tiles};return out;
}
std::shared_ptr<AlgebraicRecord> AlgebraicQueryResult::record(std::size_t index) const {
    return std::shared_ptr<AlgebraicRecord>(new AlgebraicRecord(
        std::make_shared<const AlgebraicRecord::State>(state_->result.records().at(index))));
}
void AlgebraicQueryCache::clear() {state_->cache.clear();}
std::vector<std::size_t> AlgebraicQueryCache::counts() const {
    const auto [applications,hits]=state_->cache.applications_hits();return {applications,hits};
}
} // namespace rexgraph::python
