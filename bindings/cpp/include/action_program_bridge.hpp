#pragma once

#include "temporal_program_bridge.hpp"
#include "rex_temporal/action_programs.hpp"
#include "rexlogic/model/evidence_envelope.hpp"
#include <mutex>

namespace rexgraph::python {
namespace ap = rex_temporal::action_programs;
namespace am = rexlogic::model;
namespace ast = rfc::storage;

struct ActionNode {
    std::uint64_t id{}, opcode{}, selector{};
    std::vector<std::uint64_t> operands;
    std::string scale;
};
struct ActionRecipe {
    std::uint64_t version{}, name_space{}, sequence{}, output{}, result{}, metric{}, reference{};
    bool has_reference{};
    std::string identity, meaning;
    std::vector<ActionNode> nodes;
    std::vector<std::string> labels;
    std::vector<std::uint64_t> channels;
};
struct ActionPolicy {
    std::vector<std::uint64_t> request, observation, coordinates, selection, law;
    std::vector<std::vector<std::uint64_t>> selected, fields, rules, program_rules;
    std::vector<std::string> identities, languages, descriptions, frontier;
    unsigned purpose{};
};
inline rfc::numeric::Rational action_rational(const std::string& s) {
    return rfc::numeric::canonical_rational(rfc::numeric::Rational(s,16));
}
inline am::EvidenceDependency action_dependency(const std::vector<std::uint64_t>& row) {
    if(row.size()!=4 || row[3]>1)throw std::invalid_argument("evidence dependency needs identity and read mode");
    return {am::EvidenceId(row[0],row[1],row[2]),static_cast<am::EvidenceRead>(row[3])};
}
inline std::vector<am::EvidenceDependency> action_dependencies(const std::vector<std::vector<std::uint64_t>>& rows) {
    std::vector<am::EvidenceDependency> out;for(const auto& row:rows)out.push_back(action_dependency(row));return out;
}
inline rex_temporal::UncertaintyAvailability action_available(const std::vector<std::vector<std::uint64_t>>& rows) {
    rex_temporal::UncertaintyAvailability out;
    for(const auto& row:rows){if(row.size()!=2)throw std::invalid_argument("namespace and sequence cutoff required");out.push_back({row[0],row[1]});}
    return out;
}
inline ap::Recipe action_recipe(const ActionRecipe& value) {
    ap::Recipe r;r.version=value.version;r.source_namespace=value.name_space;r.sequence=value.sequence;
    r.identity=value.identity;r.meaning=value.meaning;r.output=static_cast<ap::Output>(value.output);
    r.result=value.result;r.metric=value.metric;if(value.has_reference)r.reference=value.reference;
    for(const auto& node:value.nodes)r.nodes.push_back({node.id,static_cast<ap::Opcode>(node.opcode),node.selector,node.operands,action_rational(node.scale)});
    if(value.labels.size()!=value.channels.size())throw std::invalid_argument("action channel labels differ");
    for(std::size_t i=0;i<value.labels.size();++i)r.channels.push_back({value.labels[i],value.channels[i]});
    return r;
}

class ActionEvidenceBridge {
    am::DecodedEvidenceEnvelope decoded_;
    rex_temporal::UncertaintyEvidence evidence_;
    am::EvidenceId observation_;
    std::string archive_;
    static rex_temporal::UncertaintyEvidence policy(const am::EvidenceGraph& g,const ActionPolicy& p) {
        if(p.request.size()!=3 || p.observation.size()!=3 || p.frontier.size()!=3 || p.purpose>2)
            throw std::invalid_argument("complete current evidence request required");
        am::EvidenceFrontier f{am::EvidenceTime(g.event_clock(),action_rational(p.frontier[0])),
            am::EvidenceTime(g.knowledge_clock(),action_rational(p.frontier[1])),
            am::EvidenceTime(g.event_clock(),action_rational(p.frontier[2])),static_cast<am::EvidencePurpose>(p.purpose)};
        auto request=am::EvidenceAdmissionRequest::create(am::RequestId(p.request[0],p.request[1],p.request[2]),g,std::move(f),action_dependencies(p.selected));
        if(p.identities.size()!=p.languages.size() || p.identities.size()!=p.descriptions.size() || p.identities.size()!=p.program_rules.size())
            throw std::invalid_argument("complete evidence producer declarations required");
        std::vector<rex_temporal::UncertaintySourceProgram> programs;
        for(std::size_t i=0;i<p.identities.size();++i)programs.push_back({p.identities[i],p.languages[i],p.descriptions[i],action_dependency(p.program_rules[i])});
        return {std::move(request),action_dependencies(p.fields),p.law.empty()?std::nullopt:std::optional{action_dependency(p.law)},
            action_dependency(p.coordinates),action_dependency(p.selection),action_dependencies(p.rules),std::move(programs)};
    }
    ActionEvidenceBridge(am::DecodedEvidenceEnvelope d,std::string archive,const ActionPolicy& p)
        :decoded_(std::move(d)),evidence_(policy(decoded_.graph(),p)),
         observation_(p.observation.at(0),p.observation.at(1),p.observation.at(2)),archive_(std::move(archive)) {
        (void)decoded_.record(observation_);
    }
public:
    static std::shared_ptr<ActionEvidenceBridge> decode(const std::string& archive,const ActionPolicy& p) {
        const auto raw=std::as_bytes(std::span(archive.data(),archive.size()));
        const am::EvidenceEnvelopeLimits limits;
        auto d=am::EvidenceEnvelopeCodec::decode(raw,limits);
        const auto canonical=am::EvidenceEnvelopeCodec::encode_source(d.graph(),limits).bytes;
        // The typed decoder admits exact historical pins. Compare only their
        // declared framing with the existing native migration contract; opaque
        // assertions and original source bytes stay unchanged in archive_.
        if(!am::pin_migration_detail::equivalent(canonical,raw,limits.max_bytes))
            throw std::invalid_argument("canonical source evidence envelope required");
        return std::shared_ptr<ActionEvidenceBridge>(new ActionEvidenceBridge(std::move(d),archive,p));
    }
    const auto& evidence() const {return evidence_;}
    const auto& observation() const {return decoded_.record(observation_).observation;}
    std::string archive() const {return archive_;}
};

class ActionLeaseBridge {
    rfc::runtime::ExecutionHostPolicy policy_;
    std::shared_ptr<std::atomic_bool> cancelled_=std::make_shared<std::atomic_bool>(false);
public:
    explicit ActionLeaseBridge(const std::vector<std::size_t>& values) {
        if(values.size()!=5)throw std::invalid_argument("five execution lease limits required");
        policy_.limits.records=values[0];policy_.limits.rational_slots=values[1];policy_.limits.sparse_entries=values[2];
        policy_.limits.scalar_work=values[3];policy_.limits.integer_bits=values[4];
    }
    void cancel(){cancelled_->store(true);}
    rfc::runtime::ExecutionLease lease() const {return rfc::runtime::ExecutionLease(policy_,cancelled_);}
};
struct ActionFieldData {
    std::string label, archive;
    TemporalTensorData tensor;
    std::vector<std::uint64_t> carrier;
    std::vector<std::vector<std::uint64_t>> dependencies;
};
struct ActionSnapshot {
    bool moments{}, has_reference{}, matches_reference{};
    std::vector<ActionFieldData> fields;
    std::vector<std::vector<std::string>> components, interactions;
    std::vector<std::string> totals;
    std::vector<std::vector<std::uint64_t>> admitted;
    std::size_t steps{}, retained_entries{};
};
class ActionProgramBridge;
class ActionResultBridge {
    std::shared_ptr<ActionProgramBridge> program_;
    std::shared_ptr<ActionEvidenceBridge> evidence_;
    std::variant<ap::FieldExecution,ap::MomentExecution> result_;
public:
    ActionResultBridge(std::shared_ptr<ActionProgramBridge> p,std::shared_ptr<ActionEvidenceBridge> e,
        std::variant<ap::FieldExecution,ap::MomentExecution> r):program_(std::move(p)),evidence_(std::move(e)),result_(std::move(r)){}
    ActionSnapshot inspect(std::size_t maximum) const {
        ActionSnapshot out;std::size_t entries=0;
        const auto field=[&](std::string name,const rex_temporal::UncertaintyField& f) {
            entries=rfc::machine::checked_size_add(entries,f.sparse_values().nonzeros());
            if(entries>maximum)throw std::length_error("action field inspection admission");
            const auto& c=f.carrier();
            rfc::numeric::SparseRationalTensor tensor(c.declaration->domain,rfc::numeric::StructuralSupport::full(c.size()),f.sparse_values());
            ast::temporal_codec_detail::Writer w;ast::temporal_codec_detail::tensor(w,tensor);
            ActionFieldData row;row.label=std::move(name);row.archive={reinterpret_cast<const char*>(w.bytes().data()),w.bytes().size()};
            row.tensor=temporal_tensor_data(tensor,maximum);row.carrier={c.publication.source_namespace(),c.publication.sequence(),c.key};
            for(const auto& p:f.publications())row.dependencies.push_back({p.source_namespace(),p.sequence(),static_cast<std::uint64_t>(p.kind()),p.tensor_declaration()?p.tensor_declaration()->key:0});
            out.fields.push_back(std::move(row));
        };
        const auto admission=[&](const am::EvidenceAdmission& a) {
            a.require_for(evidence_->evidence().request());
            for(const auto& d:a.dependencies())out.admitted.push_back({d.subject.id.name_space(),d.subject.id.key(),d.subject.id.version(),static_cast<std::uint64_t>(d.subject.read),d.admitted()});
        };
        if(const auto* f=std::get_if<ap::FieldExecution>(&result_)){field("output",f->field);admission(f->admission);}
        else {
            out.moments=true;const auto& m=std::get<ap::MomentExecution>(result_);const auto& r=m.fields_and_moments;
            admission(m.admission);field("input",r.input);field("aggregate",r.aggregate);
            for(const auto& c:r.components){field(c.label,c.field);out.components.push_back({c.label,c.self.get_str(16),c.with_previous.get_str(16),c.increment.get_str(16),c.prefix_quadrance.get_str(16)});}
            for(const auto& p:r.interactions)out.interactions.push_back({std::to_string(p.left),std::to_string(p.right),p.pairing.get_str(16),p.doubled.get_str(16)});
            out.totals={r.diagonal_sum.get_str(16),r.cross_sum.get_str(16),r.total.get_str(16),r.unreported_cross_sum.get_str(16)};
            out.has_reference=r.reference.has_value();out.matches_reference=r.matches_reference.value_or(false);
            if(r.reference)field("reference",*r.reference);
            if(r.residual)field("residual",*r.residual);
            out.steps=r.execution_steps;out.retained_entries=r.retained_sparse_entries;
        }
        return out;
    }
};

class ActionProgramBridge : public std::enable_shared_from_this<ActionProgramBridge> {
    std::shared_ptr<ast::NativeAccessionDirectory> directory_;
    ap::BoundActionProgram program_;
    std::uint64_t publication_;
    rex_temporal::UncertaintyField prepared_input_;
    std::mutex directory_mutex_;
    ActionProgramBridge(std::shared_ptr<ast::NativeAccessionDirectory> directory,ap::BoundActionProgram program,
            std::uint64_t publication,rex_temporal::UncertaintyField input)
        :directory_(std::move(directory)),program_(std::move(program)),publication_(publication),prepared_input_(std::move(input)){}
    static ap::Pin pin(const std::shared_ptr<ast::NativeAccessionDirectory>& directory,ap::Kind kind,std::uint64_t key) {
        ast::NativeAccessionRequest r;r.kind=kind;r.subject=key;return directory->pin(r);
    }
public:
    static std::shared_ptr<ActionProgramBridge> prepare(const std::string& source,const std::string& endpoint,
            const ActionRecipe& value,std::uint64_t publication,const std::vector<std::vector<std::uint64_t>>& available,
            const std::vector<std::size_t>& bounds) {
        if(bounds.size()!=11)throw std::invalid_argument("eleven action preparation limits required");
        const auto raw=std::as_bytes(std::span(source.data(),source.size()));const ast::MappedRfcx mapped(raw);
        auto decoded=ast::decode_temporal_endpoint(mapped.materialize(),std::as_bytes(std::span(endpoint.data(),endpoint.size())));
        if(ast::encode_rfcx(decoded.source).bytes!=std::vector<std::byte>(raw.begin(),raw.end()) ||
           ast::encode_temporal_endpoint(decoded)!=std::vector<std::byte>(std::as_bytes(std::span(endpoint.data(),endpoint.size())).begin(),std::as_bytes(std::span(endpoint.data(),endpoint.size())).end()))
            throw std::invalid_argument("canonical native source and endpoint required");
        auto directory=std::make_shared<ast::NativeAccessionDirectory>(std::move(decoded),value.name_space);
        ap::Limits limits;limits.nodes=bounds[0];limits.channels=bounds[1];limits.text_bytes=bounds[2];limits.dimension=bounds[3];
        limits.moments={bounds[4],bounds[5],bounds[6]};limits.exact={bounds[7],bounds[8],bounds[9],bounds[10]};
        const auto published=pin(directory,ap::Kind::Field,publication);
        auto input=rex_temporal::UncertaintyField::stored(published);
        auto program=ap::BoundActionProgram::prepare(action_recipe(value),published,
            [directory](auto kind,auto key){return pin(directory,kind,key);},action_available(available),limits);
        return std::shared_ptr<ActionProgramBridge>(new ActionProgramBridge(std::move(directory),std::move(program),publication,std::move(input)));
    }
    static std::string description(const ActionRecipe& value){return ap::description(action_recipe(value));}
    std::shared_ptr<ActionResultBridge> execute(std::uint64_t input,const std::shared_ptr<ActionEvidenceBridge>& evidence,
            const std::vector<std::vector<std::uint64_t>>& availability,const std::vector<std::vector<std::uint64_t>>& pairs,
            const std::shared_ptr<ActionLeaseBridge>& lease) {
        if(!evidence || !lease)throw std::invalid_argument("actual evidence and execution lease owners required");
        const auto available=action_available(availability);
        const auto current=lease->lease();
        current.check_cancelled();
        const auto field=[&] {
            if(input==publication_)return prepared_input_;
            const auto selected=[&] {
                std::lock_guard lock(directory_mutex_);
                return pin(directory_,ap::Kind::Field,input);
            }();
            return rex_temporal::UncertaintyField::stored(selected);
        }();
        if(program_.recipe().output==ap::Output::Field){
            if(!pairs.empty())throw std::invalid_argument("field output has unused moment pairs");
            return std::make_shared<ActionResultBridge>(shared_from_this(),evidence,program_.apply(field,evidence->observation(),evidence->evidence(),available,current));
        }
        std::vector<rfc::inference::NativeActionInteractionRequest> requested;
        for(const auto& row:pairs){if(row.size()!=2)throw std::invalid_argument("interaction pair arity");requested.push_back({row[0],row[1]});}
        return std::make_shared<ActionResultBridge>(shared_from_this(),evidence,program_.moments(field,evidence->observation(),evidence->evidence(),available,requested,current));
    }
};
} // namespace rexgraph::python
