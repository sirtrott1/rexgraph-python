#pragma once
#include "finite_operator_bridge.hpp"
#include "finite_field_admission_bridge.hpp"
#include <rfc/learning/relational_available_finite_operator_codec.hpp>

namespace rexgraph::python {
struct AvailableFiniteOperatorInput {
    std::uint64_t port{},use{};
    std::shared_ptr<FiniteInputSchemaBridge> schema;
    std::string field;
    std::vector<std::uint64_t> origins;
};
struct AvailableFiniteOperatorRuleInfo {
    std::uint64_t key{};std::vector<std::uint64_t> bindings,symbols;
    bool has_target{};std::size_t target{},input_count{};std::string unsupported;
};
struct AvailableFiniteOperatorChildInfo {
    std::uint64_t key{},revision{};std::vector<AvailableFiniteOperatorRuleInfo> rules;
};
namespace available_operator_bridge_detail {
using namespace rfc;
using available_finite_bridge_detail::authority;
using program_bridge_detail::limits;
inline learning::RelationalFiniteOperatorOrigins operator_keys(const std::vector<std::uint64_t>& keys) {
    if(keys.size()!=2)throw std::invalid_argument("operator Maps and relationship keys required");
    return {keys[0],keys[1]};
}
inline learning::RelationalFiniteOperatorInstanceOrigins instance_keys(const std::vector<std::uint64_t>& keys) {
    if(keys.size()!=5)throw std::invalid_argument("transport pair, child pair and positive instance key required");
    return {keys[0],keys[1],keys[2],keys[3],keys[4]};
}
inline std::vector<learning::RelationalAvailableFiniteOperatorInput> inputs(
        const std::vector<AvailableFiniteOperatorInput>& rows,const learning::PreparedRelationalFiniteFieldOperator& op,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
    numeric::ExactActionGuard guard(limits(bounds));guard.entries(rows.size());
    std::vector<learning::RelationalAvailableFiniteOperatorInput> out;
    for(const auto& row:rows) {
        guard.step();if(!row.schema || row.schema->port()!=row.port || !row.schema->native_operator().same_owner(op))
            throw std::invalid_argument("actual native schema for its operator and named port required");
        out.push_back({row.port,{{row.schema->native_asset(),row.schema->declaration(),row.use,
            interface_bridge_detail::field(row.field,bounds,maximum)},admission_bridge_detail::input_keys(row.origins)}});
    }
    return out;
}
inline learning::RelationalFiniteOperatorInstanceDeclaration declaration(std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& d) {
    return {key,revision,finite_operator_bridge_detail::transport(d)};
}
inline void append(std::vector<std::string>& out,const std::vector<std::string>& rows,std::size_t maximum) {
    evidence_bridge_detail::bound(machine::checked_size_add(out.size(),rows.size()),maximum);out.insert(out.end(),rows.begin(),rows.end());
}
template<class Native> class Owner : public finite_operator_bridge_detail::Owner<Native> {
public:
    using finite_operator_bridge_detail::Owner<Native>::Owner;
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {this->owner_->admit(authority(context));}
};
} // namespace available_operator_bridge_detail
class AvailableFiniteOperatorInstanceBridge;
class AvailableFiniteOperatorResultBridge;
class AvailableFiniteOperatorBranchBridge;
class AvailableFiniteOperatorEvaluationBridge;
class AvailableFiniteOperatorBridge : public available_operator_bridge_detail::Owner<rfc::learning::PreparedAvailableRelationalFiniteFieldOperator> {
public:
    using Owner::Owner;
    AvailableFiniteOperatorBridge(std::shared_ptr<AvailableFiniteMapBridge> reference,const FiniteOperatorDeclaration& d,unsigned status,
            const std::vector<std::uint64_t>& keys,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :Owner(rfc::learning::PreparedAvailableRelationalFiniteFieldOperator(reference->native_owner(),finite_operator_bridge_detail::declaration(d),
            available_finite_bridge_detail::status(status),available_operator_bridge_detail::operator_keys(keys),available_operator_bridge_detail::authority(context),program_bridge_detail::limits(bounds)),bounds,maximum) {}
    static std::shared_ptr<AvailableFiniteOperatorBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<AvailableFiniteOperatorBridge>(rfc::learning::decode_available_relational_finite_operator(program_bridge_detail::bytes(image),available_operator_bridge_detail::authority(context),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    static std::string payload(std::shared_ptr<AvailableFiniteMapBridge> reference,const FiniteOperatorDeclaration& d,unsigned status,std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum) {
        return program_bridge_detail::bounded(rfc::learning::relational_finite_operator_payload(reference->native_owner(),finite_operator_bridge_detail::declaration(d),available_finite_bridge_detail::status(status),available_operator_bridge_detail::authority(context)),maximum);
    }
    static std::vector<std::string> input_payloads(std::shared_ptr<FiniteInputSchemaBridge> schema,std::uint64_t use,const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        if(!schema)throw std::invalid_argument("actual native operator input schema required");
        const auto original=rfc::learning::finite_operator_detail::original_input(schema->native_operator(),schema->port());
        const auto accession=original.accession();const auto asset=rfc::inference::encode_relational_input_use_asset(schema->native_asset());
        const auto program=rfc::inference::encode_relational_field_program(accession.program());
        const auto field=interface_bridge_detail::field(image,bounds,maximum);
        const auto value=rfc::inference::relational_input_value_payload(asset,schema->declaration(),use,field);
        const auto interface=rfc::inference::relational_input_accession_payload(program,asset,schema->declaration(),use,field);
        return {program_bridge_detail::bounded(asset,maximum),program_bridge_detail::bounded(value,maximum),
            program_bridge_detail::bounded(interface,maximum),program_bridge_detail::bounded(interface,maximum)};
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_operator(*owner_),maximum_);}
    std::shared_ptr<FiniteOperatorBridge> raw() const {return std::make_shared<FiniteOperatorBridge>(owner_->operator_program(),bounds_,maximum_);}
    std::shared_ptr<AvailableFiniteMapBridge> reference() const {return AvailableFiniteMapBridge::adopt(owner_->reference(),bounds_,maximum_);}
    bool same_owner(const AvailableFiniteOperatorBridge& other) const {return owner_->same_owner(*other.owner_);}
    unsigned status() const {return static_cast<unsigned>(owner_->status());}
    std::vector<std::uint64_t> origins() const {const auto k=owner_->origins();return {k.maps,k.relationship};}
    std::vector<std::string> receipts(std::size_t maximum) const {return available_finite_bridge_detail::receipts(owner_->obligations(),maximum,maximum_);}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out=admission_bridge_detail::dependencies(owner_->obligations(),maximum_);evidence_bridge_detail::bound(out.size(),maximum);
        available_operator_bridge_detail::append(out,reference()->dependency_archives(maximum),maximum);return out;
    }
    FiniteDeclaration source_declaration(std::shared_ptr<InputUseAssetBridge> asset,const FiniteTransportDeclaration& d,std::size_t maximum) const {
        return finite_bridge_detail::info(rfc::learning::relational_finite_operator_source_declaration(*owner_,asset->native_owner(),finite_operator_bridge_detail::transport(d),program_bridge_detail::limits(bounds_)),maximum);
    }
    std::shared_ptr<FiniteInputSchemaBridge> input_schema(std::shared_ptr<InputUseAssetBridge> asset,const FiniteTransportDeclaration& d,std::uint64_t port) const {
        const auto schema=rfc::learning::relational_finite_operator_input_schema(*owner_,asset->native_owner(),finite_operator_bridge_detail::transport(d),port,program_bridge_detail::limits(bounds_));
        return std::make_shared<FiniteInputSchemaBridge>(owner_->operator_program(),asset->native_owner(),port,schema,bounds_,maximum_);
    }
    std::string transport_payload(std::shared_ptr<AvailableFiniteSourceBridge> source,const FiniteTransportDeclaration& d,unsigned status,std::shared_ptr<AvailabilityContextBridge> context) const {
        return program_bridge_detail::bounded(rfc::learning::relational_finite_operator_transport_payload(*owner_,source->native_owner(),finite_operator_bridge_detail::transport(d),available_finite_bridge_detail::status(status),available_operator_bridge_detail::authority(context)),maximum_);
    }
    AvailableFiniteOperatorChildInfo child_declaration(std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& d,const std::vector<AvailableFiniteOperatorInput>& rows,std::size_t maximum) const {
        const auto inputs=available_operator_bridge_detail::inputs(rows,owner_->operator_program(),bounds_,maximum_);
        const auto child=rfc::learning::relational_finite_operator_child_map_declaration(*owner_,available_operator_bridge_detail::declaration(key,revision,d),inputs,program_bridge_detail::limits(bounds_));
        std::size_t count=child.rules.size();evidence_bridge_detail::bound(count,maximum);
        AvailableFiniteOperatorChildInfo out;out.key=child.key;out.revision=child.revision;
        for(const auto& r:child.rules) {
            count=rfc::machine::checked_size_add(count,rfc::machine::checked_size_add(r.bindings.size(),rfc::machine::checked_size_add(r.symbols.size(),rfc::machine::checked_size_add(r.inputs.size(),r.unsupported.size()))));evidence_bridge_detail::bound(count,maximum);
            out.rules.push_back({r.key,r.bindings,r.symbols,r.target.has_value(),r.target.value_or(0),r.inputs.size(),r.unsupported});
        }
        return out;
    }
    std::string child_payload(std::shared_ptr<AvailableFiniteSourceBridge> source,std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& d,
            const std::vector<AvailableFiniteOperatorInput>& rows,unsigned status,std::shared_ptr<AvailabilityContextBridge> context) const {
        return program_bridge_detail::bounded(rfc::learning::relational_finite_operator_child_map_payload(*owner_,source->native_owner(),available_operator_bridge_detail::declaration(key,revision,d),available_operator_bridge_detail::inputs(rows,owner_->operator_program(),bounds_,maximum_),available_finite_bridge_detail::status(status),available_operator_bridge_detail::authority(context)),maximum_);
    }
    std::string instance_payload(std::shared_ptr<AvailableFiniteSourceBridge> source,std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& d,
            const std::vector<AvailableFiniteOperatorInput>& rows,unsigned status,const std::vector<std::uint64_t>& keys,std::shared_ptr<AvailabilityContextBridge> context) const {
        return program_bridge_detail::bounded(rfc::learning::relational_finite_operator_instance_payload(*owner_,source->native_owner(),available_operator_bridge_detail::declaration(key,revision,d),available_operator_bridge_detail::inputs(rows,owner_->operator_program(),bounds_,maximum_),available_finite_bridge_detail::status(status),available_operator_bridge_detail::instance_keys(keys),available_operator_bridge_detail::authority(context)),maximum_);
    }
    std::shared_ptr<AvailableFiniteOperatorInstanceBridge> instantiate(std::shared_ptr<AvailableFiniteSourceBridge> source,std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& d,
        const std::vector<AvailableFiniteOperatorInput>& rows,unsigned status,const std::vector<std::uint64_t>& keys,std::shared_ptr<AvailabilityContextBridge> context) const;
};
class AvailableFiniteOperatorInstanceBridge : public available_operator_bridge_detail::Owner<rfc::learning::AvailableRelationalFiniteOperatorInstance> {
public:
    using Owner::Owner;
    static std::shared_ptr<AvailableFiniteOperatorInstanceBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<AvailableFiniteOperatorInstanceBridge>(rfc::learning::decode_available_relational_finite_operator_instance(program_bridge_detail::bytes(image),available_operator_bridge_detail::authority(context),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_operator_instance(*owner_),maximum_);}
    std::shared_ptr<FiniteOperatorInstanceBridge> raw() const {return std::make_shared<FiniteOperatorInstanceBridge>(owner_->raw(),bounds_,maximum_);}
    std::shared_ptr<AvailableFiniteOperatorBridge> operator_program() const {return std::make_shared<AvailableFiniteOperatorBridge>(owner_->operator_program(),bounds_,maximum_);}
    std::shared_ptr<AvailableFiniteSourceBridge> source() const {return AvailableFiniteSourceBridge::adopt(owner_->source(),bounds_,maximum_);}
    std::shared_ptr<AvailableFiniteMapBridge> field_map() const {return AvailableFiniteMapBridge::adopt(owner_->field_map(),bounds_,maximum_);}
    bool same_owner(const AvailableFiniteOperatorInstanceBridge& other) const {return owner_->same_owner(*other.owner_);}
    unsigned status() const {return static_cast<unsigned>(owner_->status());}
    std::vector<std::uint64_t> origins() const {const auto k=owner_->origins();return {k.transport_maps,k.transport_relationship,k.maps,k.relationship,k.instance};}
    std::vector<std::string> receipts(std::size_t maximum) const {return available_finite_bridge_detail::receipts(owner_->obligations(),maximum,maximum_);}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out=admission_bridge_detail::dependencies(owner_->obligations(),maximum_);evidence_bridge_detail::bound(out.size(),maximum);
        available_operator_bridge_detail::append(out,operator_program()->dependency_archives(maximum),maximum);
        available_operator_bridge_detail::append(out,source()->dependency_archives(maximum),maximum);return out;
    }
    std::shared_ptr<AvailableFiniteOperatorResultBridge> execute(std::shared_ptr<AvailabilityContextBridge> context) const;
};
inline std::shared_ptr<AvailableFiniteOperatorInstanceBridge> AvailableFiniteOperatorBridge::instantiate(std::shared_ptr<AvailableFiniteSourceBridge> source,std::uint64_t key,std::uint64_t revision,
        const FiniteTransportDeclaration& d,const std::vector<AvailableFiniteOperatorInput>& rows,unsigned status,const std::vector<std::uint64_t>& keys,std::shared_ptr<AvailabilityContextBridge> context) const {
    return std::make_shared<AvailableFiniteOperatorInstanceBridge>(owner_->instantiate(source->native_owner(),available_operator_bridge_detail::declaration(key,revision,d),
        available_operator_bridge_detail::inputs(rows,owner_->operator_program(),bounds_,maximum_),available_finite_bridge_detail::status(status),available_operator_bridge_detail::instance_keys(keys),available_operator_bridge_detail::authority(context),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
class AvailableFiniteOperatorResultBridge : public available_operator_bridge_detail::Owner<rfc::learning::AvailableRelationalFiniteOperatorResult> {
public:
    using Owner::Owner;
    static std::shared_ptr<AvailableFiniteOperatorResultBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<AvailableFiniteOperatorResultBridge>(rfc::learning::decode_available_relational_finite_operator_result(program_bridge_detail::bytes(image),available_operator_bridge_detail::authority(context),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_operator_result(*owner_),maximum_);}
    std::shared_ptr<FiniteOperatorResultBridge> raw() const {return std::make_shared<FiniteOperatorResultBridge>(owner_->result(),bounds_,maximum_);}
    std::shared_ptr<AvailableFiniteOperatorInstanceBridge> instance() const {return std::make_shared<AvailableFiniteOperatorInstanceBridge>(owner_->instance(),bounds_,maximum_);}
    std::size_t size() const {return owner_->size();}
    std::vector<std::shared_ptr<AvailableFiniteOperatorBranchBridge>> select(std::size_t begin,std::size_t count,std::shared_ptr<AvailabilityContextBridge> context) const;
    std::string completion_payload(std::size_t ordinal,const std::vector<std::string>& free,std::uint64_t setting,std::shared_ptr<AvailabilityContextBridge> context) const {
        return program_bridge_detail::bounded(rfc::learning::relational_finite_operator_completion_payload(*owner_,ordinal,available_evidence_bridge_detail::free_values(free,bounds_),setting,available_operator_bridge_detail::authority(context)),maximum_);
    }
    std::shared_ptr<AvailableFiniteOperatorEvaluationBridge> evaluate(std::size_t ordinal,const std::vector<std::string>& free,std::uint64_t setting,std::uint64_t choice,std::shared_ptr<AvailabilityContextBridge> context) const;
};
class AvailableFiniteOperatorBranchBridge : public available_operator_bridge_detail::Owner<rfc::learning::AvailableRelationalFiniteOperatorBranch> {
public:
    using Owner::Owner;
    std::shared_ptr<AvailableFiniteOperatorResultBridge> family() const {return std::make_shared<AvailableFiniteOperatorResultBridge>(owner_->parent(),bounds_,maximum_);}
    std::shared_ptr<FiniteOperatorBranchBridge> raw() const {
        const auto parent=owner_->parent().result();return std::make_shared<FiniteOperatorBranchBridge>(parent.select(owner_->ordinal(),1,program_bridge_detail::limits(bounds_)).at(0),bounds_,maximum_);
    }
    std::size_t ordinal() const {return owner_->ordinal();}
};
class AvailableFiniteOperatorEvaluationBridge : public available_operator_bridge_detail::Owner<rfc::learning::AvailableRelationalFiniteOperatorEvaluation> {
public:
    using Owner::Owner;
    static std::shared_ptr<AvailableFiniteOperatorEvaluationBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<AvailableFiniteOperatorEvaluationBridge>(rfc::learning::decode_available_relational_finite_operator_evaluation(program_bridge_detail::bytes(image),available_operator_bridge_detail::authority(context),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_available_relational_finite_operator_evaluation(*owner_),maximum_);}
    std::shared_ptr<AvailableFiniteOperatorResultBridge> family() const {return std::make_shared<AvailableFiniteOperatorResultBridge>(owner_->family(),bounds_,maximum_);}
    std::shared_ptr<FiniteOperatorEvaluationBridge> raw() const {return std::make_shared<FiniteOperatorEvaluationBridge>(owner_->result(),bounds_,maximum_);}
    std::size_t ordinal() const {return owner_->ordinal();}
    std::string setting_receipt() const {const auto o=owner_->parameter_setting();return o?program_bridge_detail::bounded(o->receipt(),maximum_):std::string{};}
    std::string choice_receipt() const {const auto origin=owner_->origin();return program_bridge_detail::bounded(origin.receipt(),maximum_);}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        auto out=family()->instance()->dependency_archives(maximum);std::vector<rfc::inference::OwnedRelationalValueOrigin> own;
        const auto setting=owner_->parameter_setting();if(setting)own.push_back(*setting);own.push_back(owner_->origin());
        available_operator_bridge_detail::append(out,admission_bridge_detail::dependencies(own,maximum_),maximum);return out;
    }
};
inline std::shared_ptr<AvailableFiniteOperatorResultBridge> AvailableFiniteOperatorInstanceBridge::execute(std::shared_ptr<AvailabilityContextBridge> context) const {
    return std::make_shared<AvailableFiniteOperatorResultBridge>(owner_->execute(available_operator_bridge_detail::authority(context),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
inline std::vector<std::shared_ptr<AvailableFiniteOperatorBranchBridge>> AvailableFiniteOperatorResultBridge::select(std::size_t begin,std::size_t count,std::shared_ptr<AvailabilityContextBridge> context) const {
    std::vector<std::shared_ptr<AvailableFiniteOperatorBranchBridge>> out;
    for(const auto& b:owner_->select(begin,count,available_operator_bridge_detail::authority(context),program_bridge_detail::limits(bounds_))) {
        out.push_back(std::make_shared<AvailableFiniteOperatorBranchBridge>(b,bounds_,maximum_));
    }
    return out;
}
inline std::shared_ptr<AvailableFiniteOperatorEvaluationBridge> AvailableFiniteOperatorResultBridge::evaluate(std::size_t ordinal,const std::vector<std::string>& free,std::uint64_t setting,std::uint64_t choice,std::shared_ptr<AvailabilityContextBridge> context) const {
    return std::make_shared<AvailableFiniteOperatorEvaluationBridge>(owner_->evaluate(ordinal,available_evidence_bridge_detail::free_values(free,bounds_),setting,choice,available_operator_bridge_detail::authority(context),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
} // namespace rexgraph::python
