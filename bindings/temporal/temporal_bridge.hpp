#pragma once
#include "rex_temporal/layer_product.hpp"
#include <memory>
#include <string>
#include <vector>

namespace rexgraph::temporal_python {
using Id=rex_temporal::programs::Id;
using Q=rex_temporal::programs::Q;
namespace p=rex_temporal::programs;
namespace l=rex_temporal::layers;
namespace t=rex_temporal::temporal_objects;
namespace a=rex_temporal::affine_time;
struct Domain {std::string lower,upper;bool lower_closed{},upper_closed{};};
struct Piece {Id identity{};Domain domain;std::vector<std::string> numerator,denominator;std::string provenance;};
struct Function {Id identity{},coordinate{};std::string name,provenance;std::vector<Piece> pieces;};
struct Node {Id opcode{},role{},constant{},coordinate{};std::vector<Id> operands;std::string scale,shift,meaning;bool has_domain{};Domain domain;};
struct Member {Id identity{},role{},function{},support{};bool exists{};};
struct Coupling {Id left_role{},right_role{},rule{};};
struct Transition {Id left{},right{};bool same{},restricted{};std::vector<Id> pairs;std::vector<Coupling> couplings;};
struct Layer {Id identity{},object{},support{};std::string scale,shift,role;};
struct Fit {Id family{},context{},function_set{};bool complete{},forced{};std::size_t examined{};std::vector<Id> programs;std::vector<std::vector<Id>> bindings;};
struct Joint {Id family{};bool complete{},forced{};std::string status;std::size_t steps{};std::vector<Id> contexts;std::vector<std::vector<Id>> projections,witnesses;};
struct Product {Id diagram{},output{};bool complete{},compatible{};std::string status;};
struct Object {Id identity{},kind{},coordinate{},producer{};std::string label;std::vector<Id> references;};

inline Q rational(const std::string& value) {
    if(value.empty() || value.size()>1048576)throw std::invalid_argument("rational text size");
    Q result(value,10);if(result.get_den()<=0)throw std::invalid_argument("positive denominator required");
    result.canonicalize();return result;
}
inline rex_temporal::FunctionDomain domain(const Domain& value) {
    return {value.lower.empty()?std::nullopt:std::optional<Q>{rational(value.lower)},
        value.upper.empty()?std::nullopt:std::optional<Q>{rational(value.upper)},value.lower_closed,value.upper_closed};
}
inline Domain domain(const rex_temporal::FunctionDomain& value) {
    return {value.lower?value.lower->get_str():"",value.upper?value.upper->get_str():"",value.lower_closed,value.upper_closed};
}
inline p::LearningLimits limits(const std::vector<std::size_t>& values) {
    p::LearningLimits result;
    if(!values.empty()) {
        if(values.size()!=9)throw std::invalid_argument("nine temporal work limits required");
        result.candidate_applications=values[0];result.program_depth=values[1];result.contexts=values[2];
        result.joint_pairs=values[3];result.search.max_steps=values[4];
        result.functions.exact={values[5],values[6],values[7],values[8]};
    }
    return result;
}
inline std::vector<std::byte> bytes(const std::string& value) {
    if(value.size()>(64u<<20))throw std::length_error("temporal archive byte limit");
    std::vector<std::byte> result(value.size());
    std::transform(value.begin(),value.end(),result.begin(),[](unsigned char c){return std::byte(c);});return result;
}
inline std::string bytes(const std::vector<std::byte>& value) {
    return {reinterpret_cast<const char*>(value.data()),value.size()};
}
inline std::vector<p::Transition> transitions(const std::vector<Transition>& values) {
    std::vector<p::Transition> result;
    for(const auto& value:values) {
        p::Transition item;item.left_context=value.left;item.right_context=value.right;item.same_program=value.same;
        if(value.restricted) {
            if(value.pairs.size()%2)throw std::invalid_argument("ordered program pairs required");
            item.allowed_program_pairs.emplace();
            for(std::size_t i=0;i<value.pairs.size();i+=2)item.allowed_program_pairs->insert({value.pairs[i],value.pairs[i+1]});
        }
        for(const auto& coupling:value.couplings)item.participation.push_back({coupling.left_role,coupling.right_role,static_cast<p::ParticipationRule>(coupling.rule)});
        result.push_back(std::move(item));
    }
    return result;
}

class Temporal {
    p::System source_;
    explicit Temporal(p::System source):source_(std::move(source)){}
public:
    static std::shared_ptr<Temporal> create(Id name_space,const std::vector<std::size_t>& work) {
        return std::shared_ptr<Temporal>(new Temporal(p::System::create(name_space,limits(work))));
    }
    static std::shared_ptr<Temporal> load(const std::string& source,const std::string& endpoint,const std::vector<std::size_t>& work) {
        return std::shared_ptr<Temporal>(new Temporal(p::System(rex_temporal::objects::RelationalObjects::decode({bytes(source),bytes(endpoint)}),limits(work))));
    }
    std::shared_ptr<Temporal> fork()const {return std::shared_ptr<Temporal>(new Temporal(source_));}
    std::string source_bytes()const {return bytes(source_.registry().encode().source_bytes);}
    std::string endpoint_bytes()const {return bytes(source_.registry().encode().endpoint_bytes);}
    Id name_space()const {return source_.registry().name_space();}
    std::vector<Object> objects()const {
        std::vector<Object> result;
        for(const auto& value:source_.registry().objects())result.push_back({value.id,static_cast<Id>(value.kind),value.coordinate,value.producer,value.label,value.references});
        return result;
    }
    Id coordinate(const std::string& name,const std::string& clock,const std::string& unit,const std::string& origin) {
        return source_.add_coordinate(name,clock,unit,origin);
    }
    Id function(Id coordinate,const std::vector<Piece>& pieces,const std::string& name) {
        p::Function result{1,coordinate,name,"supplied exact function",{}};
        for(const auto& value:pieces) {
            rex_temporal::FunctionPiece piece;piece.identity=value.identity;piece.domain=domain(value.domain);piece.provenance=value.provenance;piece.numerator.clear();piece.denominator.clear();
            for(const auto& q:value.numerator)piece.numerator.push_back(rational(q));
            for(const auto& q:value.denominator)piece.denominator.push_back(rational(q));
            result.pieces.push_back(std::move(piece));
        }
        return source_.add_function(std::move(result),name);
    }
    Function function_info(Id identity)const {
        const auto value=source_.get_function(identity);Function result{value.identity,value.parameter_coordinate,value.name,value.provenance,{}};
        for(const auto& p:value.pieces) {
            Piece item;item.identity=p.identity;item.domain=domain(p.domain);item.provenance=p.provenance;
            for(const auto& q:p.numerator)item.numerator.push_back(q.get_str());
            for(const auto& q:p.denominator)item.denominator.push_back(q.get_str());
            result.pieces.push_back(std::move(item));
        }
        return result;
    }
    std::string value(Id identity,const std::string& at)const {
        const auto result=rex_temporal::function_ops::evaluate(source_.get_function(identity),rational(at));
        return result?result->get_str():"";
    }
    bool equal(Id left,Id right)const {return rex_temporal::function_ops::compare(source_.get_function(left),source_.get_function(right)).equal;}
    Id program(const Node& value) {
        p::ProgramNode result;result.opcode=static_cast<p::Opcode>(value.opcode);result.input_role=value.role;result.constant=value.constant;
        result.operands=value.operands;result.scale=rational(value.scale);result.shift=rational(value.shift);result.output_coordinate=value.coordinate;result.meaning=value.meaning;
        if(value.has_domain)result.domain=domain(value.domain);
        return source_.add_program(std::move(result));
    }
    Node program_info(Id identity)const {
        const auto value=source_.get_program(identity);
        Node result{static_cast<Id>(value.opcode),value.input_role,value.constant,value.output_coordinate,value.operands,
                    value.scale.get_str(),value.shift.get_str(),value.meaning,value.domain.has_value(),{}};
        if(value.domain)result.domain=domain(*value.domain);return result;
    }
    Id apply(Id program,const std::vector<Id>& bindings,Id learned) {
        if(bindings.size()%2)throw std::invalid_argument("role/function bindings required");
        p::Bindings inputs;
        for(std::size_t i=0;i<bindings.size();i+=2)if(!inputs.emplace(bindings[i],bindings[i+1]).second)throw std::invalid_argument("duplicate input role");
        return learned?source_.apply_learned(learned,inputs):source_.apply(program,inputs);
    }
    Id support(Id coordinate,const std::vector<Domain>& components,const std::string& name) {
        std::vector<rex_temporal::FunctionDomain> result;for(const auto& value:components)result.push_back(domain(value));
        return source_.add_support({coordinate,std::move(result)},name);
    }
    std::vector<Domain> support_info(Id identity)const {
        const auto support=source_.get_support(identity);
        std::vector<Domain> result;for(const auto& value:support.components())result.push_back(domain(value));return result;
    }
    Id context(Id support,const std::string& name) {return source_.add_context(support,name);}
    Id function_set(Id coordinate,const std::vector<Member>& members,const std::string& name) {
        std::vector<p::Membership> result;for(const auto& value:members)result.push_back({value.identity,value.role,value.function,value.support,value.exists});
        return source_.add_function_set(coordinate,std::move(result),name);
    }
    Id fit(Id set,Id context,const std::vector<Id>& candidates,Id expected,bool observed,const Domain& region) {
        return source_.fit_context(set,context,candidates,expected,observed?std::optional<rex_temporal::FunctionDomain>{domain(region)}:std::nullopt).family;
    }
    Fit fit_info(Id identity)const {
        const auto value=source_.read_fit(identity);Fit result{value.family,value.context,value.function_set,value.complete,value.forced(),value.examined,{},{}};
        for(const auto& match:value.matches) {
            result.programs.push_back(match.program);std::vector<Id> bindings;
            for(const auto& input:match.inputs){bindings.push_back(input.role);bindings.push_back(input.function);bindings.push_back(input.membership);}
            result.bindings.push_back(std::move(bindings));
        }
        return result;
    }
    Id joint(const std::vector<Id>& families,const std::vector<Transition>& conditions) {return source_.join_evolution(families,transitions(conditions)).family;}
    Joint joint_info(Id identity)const {
        const auto value=source_.read_evolution(identity);std::string status;
        switch(value.status) {
            case rfc::local::FiniteSearchStatus::Satisfiable:status="satisfiable";break;
            case rfc::local::FiniteSearchStatus::Unsatisfiable:status="inconsistent";break;
            case rfc::local::FiniteSearchStatus::BudgetExceeded:status="incomplete";break;
            case rfc::local::FiniteSearchStatus::Cancelled:status="cancelled";break;
        }
        Joint result{value.family,value.complete,value.forced(),status,value.search_steps,{},{},{}};
        for(const auto& [context,indices]:value.possible_matches){result.contexts.push_back(context);result.projections.emplace_back(indices.begin(),indices.end());}
        for(const auto& witness:value.witnesses){std::vector<Id> row;for(const auto& [context,index]:witness){row.push_back(context);row.push_back(index);}result.witnesses.push_back(std::move(row));}
        return result;
    }
    Product product(bool functions,const std::vector<Layer>& layers,const std::vector<Transition>& conditions,const std::string& name,Id previous) {
        std::vector<l::Layer> declarations;
        for(const auto& value:layers)declarations.push_back({value.identity,value.object,value.support,rational(value.scale),rational(value.shift),value.role});
        auto result=functions?l::function_product(source_,declarations,name,previous):l::family_product(source_,declarations,transitions(conditions),name,previous);
        source_=std::move(result.source);return {result.diagram,result.output,result.complete,result.compatible,result.status};
    }
    Product product_info(Id identity)const {
        l::verify(source_,identity);
        const auto& row=rex_temporal::records::product(source_.registry().payload(identity).value,7);
        const auto output=rex_temporal::records::integer(row[3]);
        const bool complete=rex_temporal::records::integer(row[0])==1 || source_.read_evolution(output).complete;
        return {identity,output,complete,std::get<bool>(row[4].data),rex_temporal::records::text(row[5])};
    }
    Id temporal(Id operation,const std::vector<Id>& inputs,const std::vector<Id>& integers,const std::vector<std::string>& rationals,
                const Domain& region,const std::vector<std::string>& atoms,const std::string& name) {
        auto builder=source_.temporal();Id result{};
        const auto arity=[&](std::size_t n){if(inputs.size()!=n)throw std::invalid_argument("temporal input arity");};
        if(operation==1){arity(3);result=builder.interval_from_functions(inputs[0],inputs[1],inputs[2],domain(region),name);}
        else if(operation==2){arity(2);result=builder.allen(inputs[0],inputs[1],name);}
        else if(operation==3){arity(1);a::AllenSet selected;for(auto atom:integers){if(atom>=13)throw std::invalid_argument("Allen atom");selected.insert(static_cast<a::Allen>(atom));}result=builder.atom_support(inputs[0],selected,name);}
        else if(operation==4){arity(2);result=builder.support_union(inputs[0],inputs[1],name);}
        else if(operation==5){arity(2);result=builder.support_intersection(inputs[0],inputs[1],name);}
        else if(operation==6){arity(2);result=builder.support_difference(inputs[0],inputs[1],name);}
        else if(operation==7){arity(2);result=builder.interval_from_support(inputs[0],inputs[1],domain(region),name);}
        else if(operation==8){arity(1);if(rationals.size()!=1 || atoms.size()%3)throw std::invalid_argument("uniform/atomic law shape");std::vector<a::Atom> values;
            for(std::size_t i=0;i<atoms.size();i+=3)values.push_back({rational(atoms[i]),rational(atoms[i+1]),atoms[i+2]});
            result=builder.law(inputs[0],rational(rationals[0]),std::move(values),name);}
        else if(operation==9){arity(4);result=builder.mixture_probability(inputs[0],inputs[1],inputs[2],inputs[3],domain(region),name);}
        else throw std::invalid_argument("unsupported temporal constructor");
        source_.adopt(builder.registry());return result;
    }
    void verify()const {
        std::vector<Id> roots;for(const auto& value:source_.registry().objects())roots.push_back(value.id);
        source_.registry().require_acyclic_references(roots);
        source_.temporal().validate_all();
        for(const auto& value:source_.registry().objects()) {
            switch(value.kind) {
            case p::Kind::Function:(void)source_.get_function(value.id);break;
            case p::Kind::Probability:
                if(value.payload_type==t::detail::tid(17))(void)t::get_law(source_.registry(),value.id);
                else (void)source_.get_function(value.id);
                break;
            case p::Kind::Program:
                if(value.payload_type==p::codec::id(5))(void)source_.get_program(value.id);
                else if(value.payload_type!=t::detail::tid(19))throw std::invalid_argument("unsupported native program schema");
                break;
            case p::Kind::Support:(void)source_.get_support(value.id);break;
            case p::Kind::Context:(void)source_.context_support(value.id);break;
            case p::Kind::FunctionSet:(void)source_.memberships(value.id);break;
            case p::Kind::Coordinate:(void)t::get_coordinate(source_.registry(),value.id);break;
            case p::Kind::Family:
                if(value.payload_type==p::codec::id(16))(void)source_.read_fit(value.id);
                else if(value.payload_type==p::codec::id(24))(void)source_.read_evolution(value.id);
                else if(value.payload_type==p::codec::id(29))(void)source_.read_evolution_selection(value.id);
                else throw std::invalid_argument("unsupported native family schema");
                break;
            case p::Kind::Tensor:(void)source_.get_tensor(value.id);break;
            case p::Kind::TemporalRelation:break; // validate_all replays its producer.
            case p::Kind::Interface:
                if(value.payload_type==l::detail::type(3))l::verify(source_,value.id);
                else if(value.payload_type!=p::codec::id(20) && value.payload_type!=p::codec::id(23))
                    throw std::invalid_argument("unsupported native interface schema");
                break;
            default:throw std::invalid_argument("unsupported temporal source object kind");
            }
        }
    }
};
inline std::string build_digest() {return REX_TEMPORAL_DIGEST;}
} // namespace rexgraph::temporal_python
