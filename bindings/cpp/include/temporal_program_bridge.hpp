#pragma once

#include "rex_temporal/relational_program_learning.hpp"
#include "rex_temporal/layer_product.hpp"
#include "rexlogic/model/envelope.hpp"
#include <memory>
#include <string>
#include <vector>

namespace rexgraph::python {
namespace tp = rex_temporal::programs;
namespace to = rex_temporal::temporal_objects;

struct TemporalDomain {
    bool has_lower{}, has_upper{}, lower_closed{}, upper_closed{};
    std::string lower, upper;
};
struct TemporalPiece {
    std::uint64_t identity{};
    TemporalDomain domain;
    std::vector<std::string> numerator, denominator;
    std::string provenance;
};
struct TemporalFunction {
    std::uint64_t identity{}, coordinate{};
    std::string name, provenance;
    std::vector<TemporalPiece> pieces;
};
struct TemporalTransition {
    std::uint64_t left{}, right{};
    bool same_program{}, has_pairs{};
    std::vector<std::vector<std::uint64_t>> pairs, participation;
};
struct TemporalRequest {
    std::string operation;
    std::vector<std::uint64_t> ids;
    std::vector<std::vector<std::uint64_t>> rows;
    std::vector<std::string> texts, scalars;
    std::vector<TemporalDomain> domains;
    TemporalFunction function;
    std::vector<TemporalTransition> transitions;
};
struct TemporalFit {
    std::uint64_t family{}, context{}, function_set{};
    bool complete{}, forced{};
    std::size_t examined{};
    std::vector<std::uint64_t> programs;
    std::vector<std::vector<std::vector<std::uint64_t>>> inputs;
};
struct TemporalEvolution {
    bool complete{}, forced{};
    unsigned status{};
    std::size_t steps{};
    std::vector<std::uint64_t> contexts;
    std::vector<std::vector<std::size_t>> possible;
    std::vector<std::vector<std::vector<std::uint64_t>>> witnesses;
};
struct TemporalSelection {
    std::uint64_t premise{}, evolution{}, context{}, local_family{}, program{};
    std::vector<std::vector<std::uint64_t>> inputs;
};
struct TemporalInfo {
    std::uint64_t name_space{}, sequence{}, selected{};
    std::vector<std::vector<std::uint64_t>> objects, applications, ports;
    std::vector<std::vector<std::uint64_t>> references;
    std::vector<std::uint64_t> verified_layers, structural_interfaces;
    std::vector<std::string> labels, application_labels;
};

class TemporalProgramBridge {
    tp::System system_;
    std::uint64_t selected_{};
    static tp::Q rational(const std::string& text) {
        auto q = tp::Q(text, 16);
        return rfc::numeric::canonical_rational(std::move(q));
    }
    static rex_temporal::FunctionDomain domain(const TemporalDomain& d) {
        return {d.has_lower ? std::optional<tp::Q>{rational(d.lower)} : std::nullopt,
                d.has_upper ? std::optional<tp::Q>{rational(d.upper)} : std::nullopt,
                d.lower_closed, d.upper_closed};
    }
    static TemporalDomain domain(const rex_temporal::FunctionDomain& d) {
        return {bool(d.lower), bool(d.upper), d.lower_closed, d.upper_closed,
                d.lower ? d.lower->get_str(16) : "", d.upper ? d.upper->get_str(16) : ""};
    }
    static tp::LearningLimits limits(const std::vector<std::size_t>& values) {
        if (values.size() != 7) throw std::invalid_argument("seven temporal learning bounds required");
        tp::LearningLimits out;
        out.candidate_applications=values[0]; out.program_depth=values[1];
        out.contexts=values[2]; out.joint_pairs=values[3];
        out.search.max_variables=values[4]; out.search.max_steps=values[5]; out.search.max_entries=values[6];
        return out;
    }
    static std::vector<std::byte> bytes(const std::string& value) {
        const auto* start=reinterpret_cast<const std::byte*>(value.data());
        return {start,start+value.size()};
    }
    static std::string bytes(const std::vector<std::byte>& value) {
        return {reinterpret_cast<const char*>(value.data()),value.size()};
    }
    explicit TemporalProgramBridge(tp::System system, std::uint64_t selected=0)
        :system_(std::move(system)),selected_(selected){}
public:
    static std::shared_ptr<TemporalProgramBridge> create(std::uint64_t ns, const std::vector<std::size_t>& bounds) {
        return std::shared_ptr<TemporalProgramBridge>(new TemporalProgramBridge(tp::System::create(ns,limits(bounds))));
    }
    static std::shared_ptr<TemporalProgramBridge> decode(const std::string& source, const std::string& endpoint,
            std::uint64_t selected, const std::vector<std::size_t>& bounds) {
        auto registry=tp::Registry::decode({bytes(source),bytes(endpoint)});
        if(selected)(void)registry.object(selected);
        auto out=std::shared_ptr<TemporalProgramBridge>(new TemporalProgramBridge(tp::System(std::move(registry),limits(bounds)),selected));
        out->validate();
        return out;
    }
    void validate() const {
        const auto& registry=system_.registry();
        registry.require_acyclic_references([&]{std::vector<tp::Id> ids;for(const auto& o:registry.objects())ids.push_back(o.id);return ids;}());
        to::validate_all(registry,[&](const auto&,auto id){return system_.get_function(id);});
        to::detail::Validator temporal(registry,[&](const auto&,auto id){return system_.get_function(id);});
        for(const auto& o:registry.objects()) {
            switch(o.kind) {
            case tp::Kind::Function:(void)system_.get_function(o.id);break;
            case tp::Kind::Probability:
                if(o.payload_type==to::detail::tid(17))(void)temporal.law(o.id);
                else (void)system_.get_function(o.id);
                break;
            case tp::Kind::Program:
                if(o.payload_type==tp::codec::id(5))(void)system_.get_program(o.id);
                else if(o.payload_type!=to::detail::tid(19))throw std::invalid_argument("unsupported program evaluator");
                break;
            case tp::Kind::Support:(void)system_.get_support(o.id);break;
            case tp::Kind::Context:(void)system_.context_support(o.id);break;
            case tp::Kind::FunctionSet:(void)system_.memberships(o.id);break;
            case tp::Kind::Coordinate:(void)temporal.coordinate(o.id);break;
            case tp::Kind::Family:
                if(o.payload_type==tp::codec::id(16))(void)system_.read_fit(o.id);
                else if(o.payload_type==tp::codec::id(24))(void)system_.read_evolution(o.id);
                else if(o.payload_type==tp::codec::id(29))(void)system_.read_evolution_selection(o.id);
                else throw std::invalid_argument("unsupported family evaluator");
                break;
            case tp::Kind::Tensor:(void)system_.get_tensor(o.id);break;
            case tp::Kind::TemporalRelation:break; // temporal validate_all replays these producers
            case tp::Kind::Interface:
                if(o.payload_type==rex_temporal::layers::detail::type(3))rex_temporal::layers::verify(system_,o.id);
                break;
            default:throw std::invalid_argument("unsupported relational runtime object kind");
            }
        }
    }
    std::vector<std::string> encode() const {
        const auto encoded=system_.registry().encode();
        return {bytes(encoded.source_bytes),bytes(encoded.endpoint_bytes)};
    }
    bool same_owner(const TemporalProgramBridge& other) const {return system_.registry().same_owner(other.system_.registry());}
    bool extends(const TemporalProgramBridge& parent) const {
        const auto& before=parent.system_.registry();const auto& after=system_.registry();
        if(before.name_space()!=after.name_space() || after.endpoint().sequence<=before.endpoint().sequence ||
           after.objects().size()<before.objects().size() || after.applications().size()<before.applications().size())return false;
        if(!std::equal(before.objects().begin(),before.objects().end(),after.objects().begin()) ||
           !std::equal(before.applications().begin(),before.applications().end(),after.applications().begin()))return false;
        try {rex_temporal::objects::detail::preserve_payloads(before.endpoint(),after.endpoint());}
        catch(const std::exception&){return false;}
        return true;
    }
    std::uint64_t selected() const {return selected_;}
    std::vector<std::uint64_t> identity() const {
        const auto& r=system_.registry();
        return {r.name_space(),r.endpoint().sequence,r.objects().size(),r.applications().size()};
    }
    TemporalInfo inspect(std::size_t maximum) const {
        const auto& r=system_.registry();
        if(r.objects().size()+r.applications().size()>maximum)throw std::length_error("runtime inspection record limit");
        TemporalInfo out;out.name_space=r.name_space();out.sequence=r.endpoint().sequence;out.selected=selected_;
        for(const auto& o:r.objects()) {
            out.objects.push_back({o.id,static_cast<tp::Id>(o.kind),o.coordinate,o.context,o.producer,static_cast<tp::Id>(o.status)});
            out.references.push_back(o.references);out.labels.push_back(o.label);
            if(o.kind==tp::Kind::Interface) {
                if(o.payload_type==rex_temporal::layers::detail::type(3)) {
                    rex_temporal::layers::verify(system_,o.id);out.verified_layers.push_back(o.id);
                } else out.structural_interfaces.push_back(o.id);
            }
        }
        for(const auto& a:r.applications()) {
            out.applications.push_back({a.id,a.operation,a.operation_occurrence,a.support,a.support_occurrence,a.executable?1u:0u});
            out.application_labels.push_back(a.label);
            for(const auto& p:a.inputs)out.ports.push_back({a.id,0,p.object,p.role,p.occurrence,static_cast<tp::Id>(p.expected),p.exists?1u:0u});
            for(const auto& p:a.outputs)out.ports.push_back({a.id,1,p.object,p.role,p.occurrence,static_cast<tp::Id>(p.expected),p.exists?1u:0u});
        }
        if(out.ports.size()>maximum)throw std::length_error("runtime inspection occurrence limit");
        return out;
    }
    TemporalFunction function(std::uint64_t id) const {
        const auto f=system_.get_function(id);TemporalFunction out;
        out.identity=f.identity;out.coordinate=f.parameter_coordinate;out.name=f.name;out.provenance=f.provenance;
        for(const auto& p:f.pieces){TemporalPiece piece;piece.identity=p.identity;piece.domain=domain(p.domain);piece.provenance=p.provenance;
            for(const auto& q:p.numerator)piece.numerator.push_back(q.get_str(16));
            for(const auto& q:p.denominator)piece.denominator.push_back(q.get_str(16));
            out.pieces.push_back(std::move(piece));}
        return out;
    }
    std::string evaluate(std::uint64_t id,const std::string& parameter) const {
        return rex_temporal::evaluate_function(system_.get_function(id),rational(parameter)).get_str(16);
    }
    std::string evaluate_tensor(std::uint64_t id,const std::string& parameter) const {
        return system_.evaluate_tensor(system_.get_tensor(id),rational(parameter)).get_str(16);
    }
    std::vector<TemporalDomain> support(std::uint64_t id) const {
        const auto value=system_.get_support(id);std::vector<TemporalDomain> out;
        for(const auto& d:value.components())out.push_back(domain(d));
        return out;
    }
    std::vector<std::vector<std::uint64_t>> memberships(std::uint64_t id) const {
        std::vector<std::vector<std::uint64_t>> out;
        for(const auto& m:system_.memberships(id))out.push_back({m.identity,m.role,m.function,m.support,m.exists?1u:0u});
        return out;
    }
    TemporalRequest program(std::uint64_t id) const {
        const auto p=system_.get_program(id);TemporalRequest out;
        out.ids={static_cast<tp::Id>(p.opcode),p.input_role,p.constant,p.output_coordinate};out.rows={p.operands};
        out.scalars={p.scale.get_str(16),p.shift.get_str(16)};out.texts={p.meaning};if(p.domain)out.domains.push_back(domain(*p.domain));
        return out;
    }
    TemporalFit fit(std::uint64_t id) const {
        const auto f=system_.read_fit(id);TemporalFit out;
        out.family=f.family;out.context=f.context;out.function_set=f.function_set;out.complete=f.complete;out.forced=f.forced();out.examined=f.examined;
        for(const auto& m:f.matches){out.programs.push_back(m.program);std::vector<std::vector<tp::Id>> inputs;
            for(const auto& p:m.inputs)inputs.push_back({p.role,p.function,p.membership});
            out.inputs.push_back(std::move(inputs));}
        return out;
    }
    TemporalEvolution evolution(std::uint64_t id) const {
        const auto e=system_.read_evolution(id);TemporalEvolution out;
        out.complete=e.complete;out.forced=e.forced();out.status=static_cast<unsigned>(e.status);out.steps=e.search_steps;
        for(const auto& [context,possible]:e.possible_matches){out.contexts.push_back(context);out.possible.push_back(possible);}
        for(const auto& witness:e.witnesses){std::vector<std::vector<tp::Id>> rows;
            for(const auto& [context,choice]:witness)rows.push_back({context,choice});
            out.witnesses.push_back(std::move(rows));}
        return out;
    }
    TemporalSelection selection(std::uint64_t id) const {
        const auto s=system_.read_evolution_selection(id);TemporalSelection out;
        out.premise=s.premise;out.evolution=s.evolution;out.context=s.context;out.local_family=s.local_family;out.program=s.match.program;
        for(const auto& p:s.match.inputs)out.inputs.push_back({p.role,p.function,p.membership});
        return out;
    }
    TemporalRequest layer(std::uint64_t id) const {
        namespace lr=rex_temporal::layers;
        lr::verify(system_,id);
        const auto& value=rex_temporal::records::product(system_.registry().payload(id).value,7);
        TemporalRequest out;
        out.ids={rex_temporal::records::integer(value[0]),rex_temporal::records::integer(value[3]),rex_temporal::records::integer(value[6]),std::get<bool>(value[4].data)};
        out.texts={rex_temporal::records::text(value[1]),rex_temporal::records::text(value[5])};
        for(const auto& entry:rex_temporal::records::repeated(value[2],4096)) {
            const auto& row=rex_temporal::records::product(entry,6);
            out.rows.push_back({rex_temporal::records::integer(row[0]),rex_temporal::records::integer(row[1]),rex_temporal::records::integer(row[2])});
            out.scalars.push_back(rex_temporal::records::rational(row[3]).get_str(16));
            out.scalars.push_back(rex_temporal::records::rational(row[4]).get_str(16));
            out.texts.push_back(rex_temporal::records::text(row[5]));
        }
        return out;
    }
    std::shared_ptr<TemporalProgramBridge> extend(const TemporalRequest& request) const {
        auto next=system_;tp::Id selected=0;const auto& q=request;const auto& op=q.operation;
        if(op=="coordinate")selected=next.add_coordinate(q.texts.at(0),q.texts.at(1),q.texts.at(2),q.texts.at(3));
        else if(op=="function") {
            tp::Function f;f.identity=1;f.parameter_coordinate=q.function.coordinate;f.name=q.function.name;f.provenance=q.function.provenance;
            for(const auto& p:q.function.pieces){rex_temporal::FunctionPiece piece;piece.identity=p.identity;piece.domain=domain(p.domain);piece.provenance=p.provenance;piece.denominator.clear();
                for(const auto& x:p.numerator)piece.numerator.push_back(rational(x));
                for(const auto& x:p.denominator)piece.denominator.push_back(rational(x));
                f.pieces.push_back(std::move(piece));}
            selected=next.add_function(std::move(f));
        } else if(op=="program") {
            tp::ProgramNode node;node.opcode=static_cast<tp::Opcode>(q.ids.at(0));node.input_role=q.ids.at(1);node.constant=q.ids.at(2);node.output_coordinate=q.ids.at(3);
            node.operands=q.rows.at(0);node.scale=rational(q.scalars.at(0));node.shift=rational(q.scalars.at(1));node.meaning=q.texts.at(0);
            if(!q.domains.empty())node.domain=domain(q.domains.at(0));
            selected=next.add_program(std::move(node));
        } else if(op=="support") {
            std::vector<rex_temporal::FunctionDomain> components;for(const auto& d:q.domains)components.push_back(domain(d));
            selected=next.add_support({q.ids.at(0),std::move(components)},q.texts.at(0));
        } else if(op=="context")selected=next.add_context(q.ids.at(0),q.texts.at(0));
        else if(op=="function_set") {
            std::vector<tp::Membership> members;for(const auto& row:q.rows)members.push_back({row.at(0),row.at(1),row.at(2),row.at(3),bool(row.at(4))});
            selected=next.add_function_set(q.ids.at(0),std::move(members),q.texts.at(0));
        } else if(op=="select_evolution")selected=next.select_evolution(q.ids.at(0),q.ids.at(1)).premise;
        else if(op=="apply" || op=="apply_learned" || op=="apply_evolution") {
            tp::Bindings bindings;for(const auto& row:q.rows)if(!bindings.emplace(row.at(0),row.at(1)).second)throw std::invalid_argument("duplicate function input role");
            selected=op=="apply"?next.apply(q.ids.at(0),bindings,q.texts.at(0)):op=="apply_evolution"?
                next.apply_evolution(q.ids.at(0),q.ids.at(1),bindings,q.texts.at(0)):next.apply_learned(q.ids.at(0),bindings,q.texts.at(0));
        } else if(op=="fit") {
            const auto observed=q.domains.empty()?std::optional<rex_temporal::FunctionDomain>{}:domain(q.domains.at(0));
            selected=next.fit_context(q.ids.at(0),q.ids.at(1),q.rows.at(0),q.ids.at(2),observed).family;
        } else if(op=="join") {
            std::vector<tp::Transition> transitions;
            for(const auto& t:q.transitions){tp::Transition transition;transition.left_context=t.left;transition.right_context=t.right;transition.same_program=t.same_program;
                if(t.has_pairs){transition.allowed_program_pairs.emplace();for(const auto& pair:t.pairs)transition.allowed_program_pairs->emplace(pair.at(0),pair.at(1));}
                for(const auto& p:t.participation)transition.participation.push_back({p.at(0),p.at(1),static_cast<tp::ParticipationRule>(p.at(2))});
                transitions.push_back(std::move(transition));}
            selected=next.join_evolution(q.ids,std::move(transitions)).family;
        } else if(op=="glue") {
            std::vector<tp::LocalSection> sections;for(std::size_t i=0;i<q.rows.size();++i)sections.push_back({q.rows[i].at(0),q.rows[i].at(1),rational(q.scalars.at(2*i)),rational(q.scalars.at(2*i+1))});
            const auto glued=next.glue(sections,q.texts.at(0));if(!glued.compatible)throw std::invalid_argument("incompatible local functions: "+glued.reason);selected=glued.result;
        } else if(op=="tensor")selected=next.retain_tensor(next.registry().ref(q.ids.at(0)),q.ids.at(1),q.texts.at(0));
        else {
            auto builder=next.temporal();
            if(op=="interval")selected=builder.interval_from_functions(q.ids.at(0),q.ids.at(1),q.ids.at(2),domain(q.domains.at(0)),q.texts.at(0));
            else if(op=="allen")selected=builder.allen(q.ids.at(0),q.ids.at(1),q.texts.at(0));
            else if(op=="atom_support"){rex_temporal::affine_time::AllenSet atoms;for(auto id:q.rows.at(0)){if(id>=13)throw std::invalid_argument("invalid Allen atom");atoms.insert(static_cast<rex_temporal::affine_time::Allen>(id));}selected=builder.atom_support(q.ids.at(0),std::move(atoms),q.texts.at(0));}
            else if(op=="support_union")selected=builder.support_union(q.ids.at(0),q.ids.at(1),q.texts.at(0));
            else if(op=="support_intersection")selected=builder.support_intersection(q.ids.at(0),q.ids.at(1),q.texts.at(0));
            else if(op=="support_difference")selected=builder.support_difference(q.ids.at(0),q.ids.at(1),q.texts.at(0));
            else throw std::invalid_argument("unsupported relational runtime extension");
            next.adopt(builder.registry());
        }
        return std::shared_ptr<TemporalProgramBridge>(new TemporalProgramBridge(std::move(next),selected));
    }
};

struct TemporalTensorData {
    std::vector<std::size_t> shape, support, indices;
    std::vector<std::string> values;
    std::size_t support_count{};
    bool full_support{};
};
inline TemporalTensorData temporal_tensor_data(const rfc::numeric::SparseRationalTensor& tensor,std::size_t maximum) {
    if(tensor.values().nonzeros()>maximum || (tensor.support().count()!=tensor.domain().logical_size() && tensor.support().count()>maximum))
        throw std::length_error("field tensor inspection limit");
    TemporalTensorData out;out.shape=tensor.domain().extents();out.support_count=tensor.support().count();out.full_support=out.support_count==tensor.domain().logical_size();
    if(!out.full_support)tensor.support().for_each_active([&](auto i){out.support.push_back(i);});
    for(const auto& entry:tensor.values().entries()){out.indices.push_back(entry.index);out.values.push_back(entry.value.get_str(16));}
    return out;
}
class TemporalFieldFunctionBridge;
class TemporalWholeFieldBridge {
    std::shared_ptr<TemporalFieldFunctionBridge> owner_;
    rexlogic::model::GradeMetricField field_;
public:
    TemporalWholeFieldBridge(std::shared_ptr<TemporalFieldFunctionBridge> owner,rexlogic::model::GradeMetricField field)
        :owner_(std::move(owner)),field_(std::move(field)){}
    TemporalTensorData inspect(std::size_t maximum) const;
    bool same_owner(const TemporalWholeFieldBridge& other) const;
    std::string quadrance() const;
};
class TemporalFieldFunctionBridge : public std::enable_shared_from_this<TemporalFieldFunctionBridge> {
    rexlogic::model::DecodedModelEnvelope envelope_;
    std::optional<rex_temporal::BoundPolynomialTensorFunction> polynomial_;
    std::optional<rex_temporal::BoundFieldFunction> field_;
    std::string archive_;
    TemporalFieldFunctionBridge(rexlogic::model::DecodedModelEnvelope envelope,std::string archive,std::uint64_t record,bool polynomial)
        :envelope_(std::move(envelope)),archive_(std::move(archive)) {
        if(!envelope_.local_fields())throw std::invalid_argument("field function requires a complete native local-field model envelope");
        if(polynomial)polynomial_=rex_temporal::BoundPolynomialTensorFunction::bind(*envelope_.local_fields(),record);
        else field_=rex_temporal::BoundFieldFunction::bind(*envelope_.local_fields(),record);
    }
public:
    static std::shared_ptr<TemporalFieldFunctionBridge> decode(const std::string& archive,std::uint64_t record,bool polynomial) {
        const auto raw=std::as_bytes(std::span(archive.data(),archive.size()));
        auto envelope=rexlogic::model::ModelEnvelopeCodec::decode(raw);
        if(!envelope.local_fields())throw std::invalid_argument("field function requires a complete native local-field model envelope");
        const auto encoded=rexlogic::model::ModelEnvelopeCodec::encode(*envelope.local_fields()).bytes;
        if(encoded.size()!=archive.size() || !std::equal(encoded.begin(),encoded.end(),raw.begin()))throw std::invalid_argument("noncanonical field function envelope");
        return std::shared_ptr<TemporalFieldFunctionBridge>(new TemporalFieldFunctionBridge(std::move(envelope),archive,record,polynomial));
    }
    const rexlogic::model::BoundGradePairing& pairing() const {return polynomial_?polynomial_->factors().pairing():field_->pairing();}
    std::string archive() const {return archive_;}
    std::vector<std::vector<std::uint64_t>> ports() const {
        const auto& p=pairing();std::vector<std::vector<std::uint64_t>> out;std::size_t offset=0;
        for(const auto id:p.specification().ports){const auto& port=p.bundle().model().port(id);const auto& stalk=*port.stalk_binding();
            const auto size=stalk.carrier.domain.logical_size();out.push_back({id.name_space(),id.key(),id.version(),port.cell().cell().grade.value,port.cell().stable_key(),offset,size});
            const auto& names=stalk.declaration.axes.at(0).coordinates;out.back().insert(out.back().end(),names.begin(),names.end());offset+=size;}
        return out;
    }
    TemporalTensorData coefficients(std::size_t maximum) const {
        if(!polynomial_)throw std::invalid_argument("coefficient tensor inspection requires a polynomial field function");
        return temporal_tensor_data(polynomial_->reconstruct_coefficients(),maximum);
    }
    std::vector<std::size_t> powers() const {
        if(!polynomial_)throw std::invalid_argument("power axis requires a polynomial field function");
        return polynomial_->specification().powers;
    }
    std::shared_ptr<TemporalWholeFieldBridge> evaluate(const std::string& parameter) {
        const auto q=rfc::numeric::canonical_rational(tp::Q(parameter,16));
        auto result=polynomial_?polynomial_->evaluate(q):field_->evaluate(q);
        return std::make_shared<TemporalWholeFieldBridge>(shared_from_this(),std::move(result));
    }
    TemporalTensorData export_field(const rexlogic::model::GradeMetricField& field,std::size_t maximum) const {
        const auto& p=pairing();
        auto tensor=p.specification().family==rexlogic::model::LocalFieldFamily::Downward?
            p.export_field(field.downward(),1):p.export_field(field.observation(),1);
        return temporal_tensor_data(tensor.tensor,maximum);
    }
    bool same_owner(const TemporalFieldFunctionBridge& other) const {
        return envelope_.model().source().same_owner(other.envelope_.model().source());
    }
    std::string quadrance(const rexlogic::model::GradeMetricField& field) const {
        const auto& p=pairing();const auto q=p.specification().family==rexlogic::model::LocalFieldFamily::Downward?
            p.pair(field.downward(),field.downward()):p.pair(field.observation(),field.observation());
        return q.value().get_str(16);
    }
};
inline TemporalTensorData TemporalWholeFieldBridge::inspect(std::size_t maximum) const {return owner_->export_field(field_,maximum);}
inline bool TemporalWholeFieldBridge::same_owner(const TemporalWholeFieldBridge& other) const {return owner_->same_owner(*other.owner_);}
inline std::string TemporalWholeFieldBridge::quadrance() const {return owner_->quadrance(field_);}
} // namespace rexgraph::python
