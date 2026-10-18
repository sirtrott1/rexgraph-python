#pragma once
#include "finite_field_bridge.hpp"
#include <rfc/learning/relational_finite_operator_codec.hpp>

namespace rexgraph::python {
struct FiniteOperatorPort { std::uint64_t key{}, rule{}; std::size_t input{}; };
struct FiniteOperatorDeclaration { std::uint64_t key{}, revision{}; std::vector<FiniteOperatorPort> ports; };
struct FiniteUseCorrespondence { std::uint64_t reference{}, input{}; };
struct FiniteTransportDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<std::vector<std::size_t>> cells, occurrences;
    std::vector<FiniteUseCorrespondence> uses;
};
struct FiniteVariableCorrespondence { std::size_t reference{}, input{}, grade{}, reference_cell{}, input_cell{}; };
struct FiniteSchemaInfo {
    std::uint64_t key{}, node{}; std::size_t grade{}; bool isometric{};
    std::vector<ProgramStalkInfo> stalks;
    std::vector<std::vector<std::size_t>> cells, occurrences;
    std::string accession, reconstruction;
};
namespace finite_operator_bridge_detail {
using namespace rfc;
inline learning::RelationalFiniteOperatorDeclaration declaration(const FiniteOperatorDeclaration& d) {
    learning::RelationalFiniteOperatorDeclaration out{d.key,d.revision,{}};
    for (const auto& p:d.ports) out.ports.push_back({p.key,p.rule,p.input});
    return out;
}
inline learning::RelationalFiniteSourceTransportDeclaration transport(const FiniteTransportDeclaration& d) {
    learning::RelationalFiniteSourceTransportDeclaration out;out.key=d.key;out.revision=d.revision;
    for (const auto& row:d.cells) {
        std::vector<std::uint32_t> cells;
        for (const auto i:row) cells.push_back(program_bridge_detail::cell(0,i).index);
        out.source_map.cells.push_back(std::move(cells));
    }
    out.source_map.occurrences=d.occurrences;
    for (const auto& u:d.uses) out.uses.push_back({u.reference,u.input});
    return out;
}
inline FiniteTransportDeclaration transport_info(const learning::RelationalFiniteSourceTransportDeclaration& d,std::size_t maximum) {
    evidence_bridge_detail::bound(machine::checked_size_add(algebra::owned_gluing_detail::words(d.source_map),d.uses.size()),maximum);
    FiniteTransportDeclaration out;out.key=d.key;out.revision=d.revision;
    for (const auto& row:d.source_map.cells) out.cells.emplace_back(row.begin(),row.end());
    out.occurrences=d.source_map.occurrences;
    for (const auto& u:d.uses) out.uses.push_back({u.reference,u.input});
    return out;
}
template<class Native> class Owner {
public:
    Owner(Native value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const Native>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    const Native& native_owner() const & {return *owner_;}
    const Native& native_owner() const && = delete;
protected:
    std::shared_ptr<const Native> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
} // namespace finite_operator_bridge_detail
class FiniteOperatorBridge;
class FiniteOperatorInstanceBridge;
class FiniteOperatorResultBridge;
class FiniteOperatorBranchBridge;
class FiniteOperatorEvaluationBridge;
class FiniteInputSchemaBridge {
public:
    FiniteInputSchemaBridge(rfc::learning::PreparedRelationalFiniteFieldOperator op,
            rfc::inference::OwnedRelationalInputUseAsset asset,std::uint64_t port,
            rfc::inference::RelationalInputAccessionDeclaration declaration,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :op_(std::move(op)),asset_(std::move(asset)),port_(port),declaration_(std::move(declaration)),bounds_(std::move(bounds)),maximum_(maximum) {}
    const rfc::inference::RelationalInputAccessionDeclaration& declaration() const & {return declaration_;}
    const rfc::inference::RelationalInputAccessionDeclaration& declaration() const && = delete;
    rfc::inference::OwnedRelationalInputUseAsset native_asset() const {return asset_;}
    rfc::learning::PreparedRelationalFiniteFieldOperator native_operator() const {return op_;}
    std::uint64_t port() const {return port_;}
    std::shared_ptr<InputUseAssetBridge> asset() const {return InputUseAssetBridge::adopt(asset_,bounds_,maximum_);}
    FiniteSchemaInfo info(std::size_t maximum) const {
        using namespace rfc; const auto& d=declaration_;
        auto count=machine::checked_size_add(algebra::owned_gluing_detail::words(d.instances),d.source_stalks.size());
        count=machine::checked_size_add(count,machine::checked_size_add(d.accession.nonzeros(),d.reconstruction.nonzeros()));
        for (const auto& s:d.source_stalks) count=machine::checked_size_add(count,machine::checked_size_add(s.directions.size(),s.metric.nonzeros()));
        evidence_bridge_detail::bound(count,maximum);
        FiniteSchemaInfo out;out.key=d.key;out.node=d.node;out.grade=d.source_grade.value;out.isometric=d.isometric;
        for (const auto& s:d.source_stalks) out.stalks.push_back(program_bridge_detail::stalk_info(s));
        for (const auto& row:d.instances.cells) out.cells.emplace_back(row.begin(),row.end());
        out.occurrences=d.instances.occurrences;
        out.accession=program_bridge_detail::matrix_bytes(d.accession);out.reconstruction=program_bridge_detail::matrix_bytes(d.reconstruction);
        evidence_bridge_detail::bound(out.accession.size(),maximum_);evidence_bridge_detail::bound(out.reconstruction.size(),maximum_);
        return out;
    }
private:
    rfc::learning::PreparedRelationalFiniteFieldOperator op_;
    rfc::inference::OwnedRelationalInputUseAsset asset_;std::uint64_t port_{};
    rfc::inference::RelationalInputAccessionDeclaration declaration_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
struct FiniteOperatorInput {std::uint64_t port{},use{};std::shared_ptr<FiniteInputSchemaBridge> schema;std::string field;};
class FiniteTransportBridge : public finite_operator_bridge_detail::Owner<rfc::learning::OwnedRelationalFiniteSourceTransport> {
public:
    using Owner::Owner;
    FiniteTransportBridge(std::shared_ptr<FiniteSourceBridge> reference,std::shared_ptr<FiniteSourceBridge> source,
            const FiniteTransportDeclaration& d,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :Owner(rfc::learning::OwnedRelationalFiniteSourceTransport(reference->native_owner(),source->native_owner(),finite_operator_bridge_detail::transport(d),program_bridge_detail::limits(bounds)),bounds,maximum) {}
    std::shared_ptr<FiniteSourceBridge> reference() const {return FiniteSourceBridge::adopt(owner_->reference(),bounds_,maximum_);}
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_->source(),bounds_,maximum_);}
    FiniteTransportDeclaration info(std::size_t maximum) const {return finite_operator_bridge_detail::transport_info(owner_->declaration(),maximum);}
    bool same_owner(const FiniteTransportBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::vector<FiniteVariableCorrespondence> variables(std::size_t begin,std::size_t count) const {
        std::vector<FiniteVariableCorrespondence> out;
        for (const auto& v:owner_->variable_positions(begin,count,program_bridge_detail::limits(bounds_))) out.push_back({v.reference,v.input,v.reference_cell.grade.value,v.reference_cell.index,v.input_cell.index});
        return out;
    }
    std::vector<FiniteUseCorrespondence> uses(std::size_t begin,std::size_t count) const {
        std::vector<FiniteUseCorrespondence> out;
        for (const auto& u:owner_->use_correspondences(begin,count,program_bridge_detail::limits(bounds_))) out.push_back({u.reference,u.input});
        return out;
    }
};
class FiniteOperatorBridge : public finite_operator_bridge_detail::Owner<rfc::learning::PreparedRelationalFiniteFieldOperator> {
public:
    using Owner::Owner;
    FiniteOperatorBridge(std::shared_ptr<FiniteMapBridge> reference,const FiniteOperatorDeclaration& d,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :Owner(rfc::learning::PreparedRelationalFiniteFieldOperator(reference->native_owner(),finite_operator_bridge_detail::declaration(d),program_bridge_detail::limits(bounds)),bounds,maximum) {}
    static std::shared_ptr<FiniteOperatorBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<FiniteOperatorBridge>(rfc::learning::decode_relational_finite_operator(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_operator(*owner_),maximum_);}
    std::shared_ptr<FiniteMapBridge> reference() const {return FiniteMapBridge::adopt(owner_->reference(),bounds_,maximum_);}
    std::uint64_t key() const {return owner_->declaration().key;}
    std::uint64_t revision() const {return owner_->declaration().revision;}
    std::size_t size() const {return owner_->declaration().ports.size();}
    bool same_owner(const FiniteOperatorBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::vector<FiniteOperatorPort> ports(std::size_t begin,std::size_t count) const {
        std::vector<FiniteOperatorPort> out;
        for (const auto& p:owner_->select_ports(begin,count,program_bridge_detail::limits(bounds_))) out.push_back({p.key,p.rule,p.input});
        return out;
    }
    FiniteDeclaration source_declaration(std::shared_ptr<InputUseAssetBridge> asset,const FiniteTransportDeclaration& d,std::size_t maximum) const {
        return finite_bridge_detail::info(rfc::learning::relational_finite_operator_source_declaration(*owner_,asset->native_owner(),finite_operator_bridge_detail::transport(d),program_bridge_detail::limits(bounds_)),maximum);
    }
    std::shared_ptr<FiniteInputSchemaBridge> input_schema(std::shared_ptr<InputUseAssetBridge> asset,const FiniteTransportDeclaration& d,std::uint64_t port) const {
        const auto value=rfc::learning::relational_finite_operator_input_schema(*owner_,asset->native_owner(),finite_operator_bridge_detail::transport(d),port,program_bridge_detail::limits(bounds_));
        return std::make_shared<FiniteInputSchemaBridge>(*owner_,asset->native_owner(),port,value,bounds_,maximum_);
    }
    std::shared_ptr<FiniteOperatorInstanceBridge> instantiate(std::shared_ptr<FiniteSourceBridge> source,std::uint64_t key,std::uint64_t revision,
        const FiniteTransportDeclaration& transport,const std::vector<FiniteOperatorInput>& inputs) const;
};
class FiniteOperatorInstanceBridge : public finite_operator_bridge_detail::Owner<rfc::learning::RelationalFiniteOperatorInstance> {
public:
    using Owner::Owner;
    static std::shared_ptr<FiniteOperatorInstanceBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<FiniteOperatorInstanceBridge>(rfc::learning::decode_relational_finite_operator_instance(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_operator_instance(*owner_),maximum_);}
    std::uint64_t key() const {return owner_->declaration().key;}
    std::uint64_t revision() const {return owner_->declaration().revision;}
    std::size_t size() const {return owner_->inputs().size();}
    bool same_owner(const FiniteOperatorInstanceBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::shared_ptr<FiniteOperatorBridge> operator_program() const {return std::make_shared<FiniteOperatorBridge>(owner_->operator_program(),bounds_,maximum_);}
    std::shared_ptr<FiniteTransportBridge> transport() const {return std::make_shared<FiniteTransportBridge>(owner_->source_transport(),bounds_,maximum_);}
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_->source(),bounds_,maximum_);}
    std::shared_ptr<FiniteMapBridge> model() const {return FiniteMapBridge::adopt(owner_->map(),bounds_,maximum_);}
    std::vector<FiniteOperatorInput> inputs(std::size_t begin,std::size_t count) const {
        rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
        const auto end=rfc::machine::checked_size_add(begin,count);evidence_bridge_detail::bound(end,owner_->inputs().size());guard.entries(count);
        std::vector<FiniteOperatorInput> out;
        for (auto i=begin;i<end;++i) {
            guard.step();const auto& row=owner_->inputs()[i];const auto& in=row.input;
            auto schema=std::make_shared<FiniteInputSchemaBridge>(owner_->operator_program(),in.asset,row.port,in.declaration,bounds_,maximum_);
            auto image=program_bridge_detail::tensor_bytes(in.source_field);
            evidence_bridge_detail::bound(image.size(),maximum_);
            out.push_back({row.port,in.use,std::move(schema),std::move(image)});
        }
        return out;
    }
    std::shared_ptr<FiniteOperatorResultBridge> execute() const;
};
inline std::shared_ptr<FiniteOperatorInstanceBridge> FiniteOperatorBridge::instantiate(std::shared_ptr<FiniteSourceBridge> source,
        std::uint64_t key,std::uint64_t revision,const FiniteTransportDeclaration& transport,const std::vector<FiniteOperatorInput>& inputs) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));guard.entries(inputs.size());
    std::vector<rfc::learning::RelationalFiniteOperatorInput> rows;
    for (const auto& row:inputs) {
        guard.step();
        if (!row.schema || row.schema->port()!=row.port || !row.schema->native_operator().same_owner(*owner_)) throw std::invalid_argument("finite input schema requires its actual operator and named port");
        rows.push_back({row.port,{row.schema->native_asset(),row.schema->declaration(),row.use,interface_bridge_detail::field(row.field,bounds_,maximum_)}});
    }
    return std::make_shared<FiniteOperatorInstanceBridge>(owner_->instantiate(source->native_owner(),{key,revision,finite_operator_bridge_detail::transport(transport)},std::move(rows),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
class FiniteOperatorResultBridge : public finite_operator_bridge_detail::Owner<rfc::learning::RelationalFiniteOperatorResult> {
public:
    using Owner::Owner;
    static std::shared_ptr<FiniteOperatorResultBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<FiniteOperatorResultBridge>(rfc::learning::decode_relational_finite_operator_result(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_operator_result(*owner_),maximum_);}
    std::shared_ptr<FiniteOperatorInstanceBridge> instance() const {return std::make_shared<FiniteOperatorInstanceBridge>(owner_->instance(),bounds_,maximum_);}
    std::shared_ptr<FiniteResultBridge> raw() const {return FiniteResultBridge::adopt(owner_->result(),bounds_,maximum_);}
    std::size_t size() const {return owner_->size();}
    std::vector<std::shared_ptr<FiniteOperatorBranchBridge>> select(std::size_t begin,std::size_t count) const;
    std::shared_ptr<FiniteOperatorEvaluationBridge> evaluate(std::size_t ordinal,const std::vector<std::string>& free) const;
};
class FiniteOperatorBranchBridge : public finite_operator_bridge_detail::Owner<rfc::learning::RelationalFiniteOperatorBranch> {
public:
    using Owner::Owner;
    std::shared_ptr<FiniteOperatorResultBridge> parent() const {return std::make_shared<FiniteOperatorResultBridge>(owner_->parent(),bounds_,maximum_);}
    std::size_t ordinal() const {return owner_->ordinal();}
    std::shared_ptr<FiniteBranchBridge> raw() const {return std::make_shared<FiniteBranchBridge>(owner_->parent().result(),owner_->ordinal(),bounds_,maximum_);}
};
class FiniteOperatorEvaluationBridge : public finite_operator_bridge_detail::Owner<rfc::learning::RelationalFiniteOperatorEvaluation> {
public:
    using Owner::Owner;
    static std::shared_ptr<FiniteOperatorEvaluationBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::make_shared<FiniteOperatorEvaluationBridge>(rfc::learning::decode_relational_finite_operator_evaluation(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_operator_evaluation(*owner_),maximum_);}
    std::shared_ptr<FiniteOperatorResultBridge> family() const {return std::make_shared<FiniteOperatorResultBridge>(owner_->family(),bounds_,maximum_);}
    std::size_t ordinal() const {return owner_->ordinal();}
    std::shared_ptr<FiniteEvaluationBridge> raw() const {return FiniteEvaluationBridge::adopt(owner_->result(),bounds_,maximum_);}
};
inline std::shared_ptr<FiniteOperatorResultBridge> FiniteOperatorInstanceBridge::execute() const {
    return std::make_shared<FiniteOperatorResultBridge>(owner_->execute(program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
inline std::vector<std::shared_ptr<FiniteOperatorBranchBridge>> FiniteOperatorResultBridge::select(std::size_t begin,std::size_t count) const {
    std::vector<std::shared_ptr<FiniteOperatorBranchBridge>> out;
    for (const auto& row:owner_->select(begin,count,program_bridge_detail::limits(bounds_))) out.push_back(std::make_shared<FiniteOperatorBranchBridge>(row,bounds_,maximum_));
    return out;
}
inline std::shared_ptr<FiniteOperatorEvaluationBridge> FiniteOperatorResultBridge::evaluate(std::size_t ordinal,const std::vector<std::string>& free) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));rfc::numeric::RationalVector values;guard.dimension(free.size());
    for (const auto& text:free) {
        rfc::numeric::Rational q;
        if(q.set_str(text,16)!=0 || q.get_den()==0) throw std::invalid_argument("exact complete operator free values required");
        q.canonicalize();guard.value(q);values.push_back(std::move(q));
    }
    return std::make_shared<FiniteOperatorEvaluationBridge>(owner_->evaluate(ordinal,std::move(values),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
} // namespace rexgraph::python
