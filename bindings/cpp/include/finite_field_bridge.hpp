#pragma once
#include "field_evidence_bridge.hpp"
#include <rfc/learning/relational_finite_field_codec.hpp>

namespace rexgraph::python {
struct FiniteSymbol { std::uint64_t key{}; std::string definition; };
struct FiniteVariable { std::size_t grade{}, cell{}; std::vector<std::uint64_t> domain; };
struct FiniteChart {
    std::uint64_t key{}; std::string name;
    std::vector<std::vector<std::size_t>> patch, support;
    std::vector<std::vector<std::uint64_t>> allowed;
};
struct FiniteBinding { std::uint64_t key{}, use{}; std::size_t grade{}, cell{}; };
struct FiniteDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<FiniteSymbol> symbols; std::vector<FiniteVariable> variables;
    std::vector<FiniteChart> charts; std::vector<FiniteBinding> bindings;
};
struct FiniteAssignmentValue { std::size_t grade{}, cell{}; std::uint64_t symbol{}; };
struct FiniteSearchInfo {
    unsigned status{};
    bool has_witness{};
    std::vector<FiniteAssignmentValue> witness;
    std::size_t steps{}, value_trials{}, chart_tests{}, tuple_tests{}, coordinate_tests{};
    std::size_t variables{}, peak_assigned{}, peak_entries{}, domain_reductions{}, search_nodes{};
};
struct FiniteTarget {
    std::shared_ptr<FieldProgramBridge> fixed;
    std::shared_ptr<FieldEvidenceBridge> affine;
};
struct FiniteRule {
    std::uint64_t key{}; std::vector<std::uint64_t> bindings, symbols;
    bool has_target{}; std::size_t target{}; std::string unsupported;
    std::vector<std::shared_ptr<AccessionInputBridge>> inputs;
};
struct FiniteMapDeclaration { std::uint64_t key{}, revision{}; std::vector<FiniteRule> rules; };
struct FiniteBranchInfo {
    std::size_t ordinal{}, section{}, target{}; std::uint64_t rule{};
    unsigned status{}; bool has_rule{}, has_target{}, affine{};
    std::vector<std::uint64_t> assignment;
};
namespace finite_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline std::vector<core::CellId> cells(const std::vector<std::vector<std::size_t>>& rows) {
    std::vector<core::CellId> out;
    for(const auto& row:rows) {
        if(row.size()!=2)throw std::invalid_argument("finite cell requires original grade/index");
        out.push_back(cell(row[0],row[1]));
    }
    return out;
}
inline inference::RelationalFiniteSourceDeclaration declaration(const FiniteDeclaration& value) {
    inference::RelationalFiniteSourceDeclaration out;out.key=value.key;out.revision=value.revision;
    for(const auto& s:value.symbols) {const auto b=bytes(s.definition);out.symbols.push_back({s.key,{b.begin(),b.end()}});}
    for(const auto& v:value.variables)out.variables.push_back({cell(v.grade,v.cell),v.domain});
    for(const auto& c:value.charts)out.charts.push_back({c.key,c.name,cells(c.patch),cells(c.support),c.allowed});
    for(const auto& b:value.bindings)out.bindings.push_back({b.key,b.use,cell(b.grade,b.cell)});
    return out;
}
inline FiniteDeclaration info(const inference::RelationalFiniteSourceDeclaration& value,std::size_t maximum) {
    FiniteDeclaration out;out.key=value.key;out.revision=value.revision;
    std::size_t count=value.symbols.size()+value.variables.size()+value.charts.size()+value.bindings.size();
    evidence_bridge_detail::bound(count,maximum);
    for(const auto& s:value.symbols) {count=machine::checked_size_add(count,s.definition.size());evidence_bridge_detail::bound(count,maximum);out.symbols.push_back({s.key,text(s.definition)});}
    for(const auto& v:value.variables) {count=machine::checked_size_add(count,v.domain.size());evidence_bridge_detail::bound(count,maximum);out.variables.push_back({v.cell.grade.value,v.cell.index,v.domain});}
    for(const auto& c:value.charts) {
        count=machine::checked_size_add(count,c.name.size()+c.patch.size()+c.support.size());
        for(const auto& tuple:c.allowed)count=machine::checked_size_add(count,tuple.size()+1);
        evidence_bridge_detail::bound(count,maximum);
        FiniteChart row;row.key=c.key;row.name=c.name;row.allowed=c.allowed;
        for(const auto p:c.patch)row.patch.push_back({p.grade.value,p.index});
        for(const auto p:c.support)row.support.push_back({p.grade.value,p.index});
        out.charts.push_back(std::move(row));
    }
    for(const auto& b:value.bindings)out.bindings.push_back({b.key,b.use,b.cell.grade.value,b.cell.index});
    return out;
}
inline learning::RelationalFiniteFieldTarget target(const FiniteTarget& value) {
    if(bool(value.fixed)==bool(value.affine))throw std::invalid_argument("one actual finite numerical target owner required");
    if(value.fixed)return value.fixed->native_owner();
    return value.affine->native_owner();
}
inline FiniteTarget target_info(const learning::RelationalFiniteFieldTarget& value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
    if(const auto* fixed=std::get_if<inference::OwnedRelationalFieldProgram>(&value))return {FieldProgramBridge::adopt(*fixed,bounds,maximum),{}};
    return {{},FieldEvidenceBridge::adopt(std::get<learning::PreparedRelationalFieldEvidence>(value),bounds,maximum)};
}
} // namespace finite_bridge_detail

class FiniteSectionsBridge;
class FiniteSourceBridge {
public:
    FiniteSourceBridge(std::shared_ptr<InputUseAssetBridge> asset,const FiniteDeclaration& value,
            const std::vector<std::size_t>& bounds,std::size_t maximum):bounds_(bounds),maximum_(maximum) {
        if(!asset)throw std::invalid_argument("actual finite asset owner required");
        owner_=std::make_shared<const rfc::inference::OwnedRelationalFiniteSource>(asset->native_owner(),finite_bridge_detail::declaration(value),program_bridge_detail::limits(bounds));
    }
    static std::shared_ptr<FiniteSourceBridge> adopt(rfc::inference::OwnedRelationalFiniteSource value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<FiniteSourceBridge>(new FiniteSourceBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<FiniteSourceBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return adopt(rfc::inference::decode_relational_finite_source(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    const rfc::inference::OwnedRelationalFiniteSource& native_owner() const & {return *owner_;}
    const rfc::inference::OwnedRelationalFiniteSource& native_owner() const && = delete;
    std::string archive() const {return program_bridge_detail::bounded(rfc::inference::encode_relational_finite_source(*owner_),maximum_);}
    FiniteDeclaration info(std::size_t maximum) const {return finite_bridge_detail::info(owner_->declaration(),maximum);}
    std::shared_ptr<InputUseAssetBridge> asset() const {return InputUseAssetBridge::adopt(owner_->asset(),bounds_,maximum_);}
    bool same_owner(const FiniteSourceBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::shared_ptr<FiniteSectionsBridge> sections() const;
    FiniteSearchInfo solve(const std::vector<FiniteAssignmentValue>& fixed,
            std::size_t variables, std::size_t steps, std::size_t entries, bool propagate) const {
        rfc::inference::OwnedRelationalFiniteSource::Assignment assignment;
        for (const auto& row : fixed) {
            if (!assignment.emplace(program_bridge_detail::cell(row.grade,row.cell),row.symbol).second)
                throw std::invalid_argument("duplicate fixed finite cell");
        }
        const rfc::local::FiniteSearchLimits limits{variables,steps,entries};
        const auto result = propagate ? owner_->propagated_global_section(assignment,limits)
                                      : owner_->bounded_global_section(assignment,limits);
        const auto& w = result.work;
        FiniteSearchInfo out;
        out.status=static_cast<unsigned>(result.status);out.has_witness=result.witness.has_value();
        if (result.witness) for (const auto& variable : owner_->declaration().variables)
            out.witness.push_back({variable.cell.grade.value,variable.cell.index,result.witness->at(variable.cell)});
        out.steps=w.steps;out.value_trials=w.value_trials;out.chart_tests=w.chart_tests;
        out.tuple_tests=w.tuple_tests;out.coordinate_tests=w.coordinate_tests;
        out.variables=w.variables;out.peak_assigned=w.peak_assigned;out.peak_entries=w.peak_entries;
        out.domain_reductions=w.domain_reductions;out.search_nodes=w.search_nodes;
        return out;
    }
private:
    FiniteSourceBridge(rfc::inference::OwnedRelationalFiniteSource value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::inference::OwnedRelationalFiniteSource>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalFiniteSource> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
class FiniteSectionsBridge {
public:
    FiniteSectionsBridge(rfc::inference::RelationalFiniteSections value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(value)),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_.source(),bounds_,maximum_);}
    std::size_t size() const {return owner_.size();}
    std::size_t candidates() const {return owner_.candidate_count();}
    std::vector<std::vector<std::uint64_t>> select(std::size_t begin,std::size_t count) const {return owner_.select(begin,count,program_bridge_detail::limits(bounds_));}
private:
    rfc::inference::RelationalFiniteSections owner_;std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
inline std::shared_ptr<FiniteSectionsBridge> FiniteSourceBridge::sections() const {
    return std::make_shared<FiniteSectionsBridge>(owner_->sections(program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}

class FiniteResultBridge;
class FiniteEvaluationBridge;
class FiniteMapBridge {
public:
    FiniteMapBridge(std::shared_ptr<FiniteSourceBridge> source,const std::vector<FiniteTarget>& targets,const FiniteMapDeclaration& value,
            const std::vector<std::size_t>& bounds,std::size_t maximum):bounds_(bounds),maximum_(maximum) {
        using namespace finite_bridge_detail;
        if(!source)throw std::invalid_argument("actual finite source required");
        std::vector<learning::RelationalFiniteFieldTarget> native_targets;
        for(const auto& t:targets)native_targets.push_back(target(t));
        learning::RelationalFiniteFieldMapDeclaration d;d.key=value.key;d.revision=value.revision;
        for(const auto& row:value.rules) {
            learning::RelationalFiniteFieldRule r;r.key=row.key;r.bindings=row.bindings;r.symbols=row.symbols;r.unsupported=row.unsupported;
            if(row.has_target)r.target=row.target;
            for(const auto& input:row.inputs) {if(!input)throw std::invalid_argument("actual finite admitted input required");r.inputs.push_back(input->native_owner());}
            d.rules.push_back(std::move(r));
        }
        owner_=std::make_shared<const learning::PreparedRelationalFiniteFieldMap>(source->native_owner(),std::move(native_targets),std::move(d),limits(bounds));
    }
    static std::shared_ptr<FiniteMapBridge> adopt(rfc::learning::PreparedRelationalFiniteFieldMap value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<FiniteMapBridge>(new FiniteMapBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<FiniteMapBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return adopt(rfc::learning::decode_relational_finite_field_map(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    const rfc::learning::PreparedRelationalFiniteFieldMap& native_owner() const & {return *owner_;}
    const rfc::learning::PreparedRelationalFiniteFieldMap& native_owner() const && = delete;
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_field_map(*owner_),maximum_);}
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_->source(),bounds_,maximum_);}
    std::vector<FiniteTarget> targets(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_->targets().size(),maximum);std::vector<FiniteTarget> out;
        for(const auto& value:owner_->targets())out.push_back(finite_bridge_detail::target_info(value,bounds_,maximum_));
        return out;
    }
    FiniteMapDeclaration info(std::size_t maximum) const {
        const auto& d=owner_->declaration();FiniteMapDeclaration out;out.key=d.key;out.revision=d.revision;
        std::size_t count=d.rules.size();evidence_bridge_detail::bound(count,maximum);
        for(const auto& r:d.rules) {
            count=rfc::machine::checked_size_add(count,r.bindings.size()+r.symbols.size()+r.inputs.size()+r.unsupported.size());evidence_bridge_detail::bound(count,maximum);
            FiniteRule row;row.key=r.key;row.bindings=r.bindings;row.symbols=r.symbols;row.has_target=r.target.has_value();row.target=r.target.value_or(0);row.unsupported=r.unsupported;
            for(const auto& input:r.inputs)row.inputs.push_back(AccessionInputBridge::adopt(input,bounds_,maximum_));
            out.rules.push_back(std::move(row));
        }
        return out;
    }
    std::shared_ptr<FiniteResultBridge> execute() const;
private:
    FiniteMapBridge(rfc::learning::PreparedRelationalFiniteFieldMap value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::PreparedRelationalFiniteFieldMap>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::PreparedRelationalFiniteFieldMap> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
class FiniteBranchBridge;
class FiniteResultBridge {
public:
    static std::shared_ptr<FiniteResultBridge> adopt(rfc::learning::RelationalFiniteFieldResult value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<FiniteResultBridge>(new FiniteResultBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<FiniteResultBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return adopt(rfc::learning::decode_relational_finite_field_result(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    const rfc::learning::RelationalFiniteFieldResult& native_owner() const & {return *owner_;}
    const rfc::learning::RelationalFiniteFieldResult& native_owner() const && = delete;
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_field_result(*owner_),maximum_);}
    std::shared_ptr<FiniteMapBridge> model() const {return FiniteMapBridge::adopt(owner_->map(),bounds_,maximum_);}
    std::shared_ptr<FiniteSectionsBridge> sections() const {return std::make_shared<FiniteSectionsBridge>(owner_->sections(),bounds_,maximum_);}
    std::size_t size() const {return owner_->size();}
    std::vector<std::shared_ptr<FiniteBranchBridge>> select(std::size_t begin,std::size_t count) const;
    std::shared_ptr<FiniteEvaluationBridge> evaluate(std::size_t ordinal,const std::vector<std::string>& free) const;
private:
    FiniteResultBridge(rfc::learning::RelationalFiniteFieldResult value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::RelationalFiniteFieldResult>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalFiniteFieldResult> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
inline std::shared_ptr<FiniteResultBridge> FiniteMapBridge::execute() const {
    return FiniteResultBridge::adopt(owner_->execute(program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
class FiniteBranchBridge {
public:
    FiniteBranchBridge(rfc::learning::RelationalFiniteFieldResult family,std::size_t ordinal,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(family)),ordinal_(ordinal),bounds_(std::move(bounds)),maximum_(maximum) {(void)owner_.branch(ordinal);}
    std::shared_ptr<FiniteResultBridge> family() const {return FiniteResultBridge::adopt(owner_,bounds_,maximum_);}
    FiniteBranchInfo info() const {
        const auto& b=owner_.branch(ordinal_);const auto model=owner_.map();FiniteBranchInfo out;
        out.ordinal=b.ordinal;out.section=b.section;out.rule=b.rule.value_or(0);out.has_rule=b.rule.has_value();out.status=static_cast<unsigned>(b.status);
        const auto sections=owner_.sections();out.assignment=sections.assignment(b.section);
        if(b.rule)for(const auto& r:model.declaration().rules)if(r.key==*b.rule){out.has_target=r.target.has_value();out.target=r.target.value_or(0);break;}
        out.affine=b.value && std::holds_alternative<rfc::learning::RelationalAffineProgramResult>(*b.value);return out;
    }
    std::shared_ptr<AccessionResultBridge> fixed() const {
        const auto& b=owner_.branch(ordinal_);if(!b.value)return {};
        const auto* value=std::get_if<rfc::inference::OwnedAccessionFieldResult>(&*b.value);
        return value?AccessionResultBridge::adopt(*value,bounds_,maximum_):nullptr;
    }
    std::shared_ptr<AffineResultBridge> affine() const {
        const auto& b=owner_.branch(ordinal_);if(!b.value)return {};
        const auto* value=std::get_if<rfc::learning::RelationalAffineProgramResult>(&*b.value);
        return value?AffineResultBridge::adopt(*value,bounds_,maximum_):nullptr;
    }
private:
    rfc::learning::RelationalFiniteFieldResult owner_;std::size_t ordinal_{};
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
inline std::vector<std::shared_ptr<FiniteBranchBridge>> FiniteResultBridge::select(std::size_t begin,std::size_t count) const {
    const auto selected=owner_->select(begin,count,program_bridge_detail::limits(bounds_));std::vector<std::shared_ptr<FiniteBranchBridge>> out;
    for(const auto& b:selected)out.push_back(std::make_shared<FiniteBranchBridge>(*owner_,b.ordinal,bounds_,maximum_));
    return out;
}
class FiniteEvaluationBridge {
public:
    static std::shared_ptr<FiniteEvaluationBridge> adopt(rfc::learning::RelationalFiniteFieldEvaluation value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<FiniteEvaluationBridge>(new FiniteEvaluationBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<FiniteEvaluationBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return adopt(rfc::learning::decode_relational_finite_field_evaluation(program_bridge_detail::bytes(image),program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds)),bounds,maximum);
    }
    std::string archive() const {return program_bridge_detail::bounded(rfc::learning::encode_relational_finite_field_evaluation(*owner_),maximum_);}
    std::shared_ptr<FiniteResultBridge> family() const {return FiniteResultBridge::adopt(owner_->family(),bounds_,maximum_);}
    std::size_t ordinal() const {return owner_->ordinal();}
    std::shared_ptr<AccessionResultBridge> result() const {return AccessionResultBridge::adopt(owner_->result(),bounds_,maximum_);}
    std::shared_ptr<AffineEvaluationBridge> affine() const {const auto value=owner_->affine_evaluation();return value?AffineEvaluationBridge::adopt(*value,bounds_,maximum_):nullptr;}
    std::string free_values() const {
        rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
        return program_bridge_detail::tensor_bytes(rfc::inference::field_interface_detail::tensor(owner_->free_values(),guard));
    }
private:
    FiniteEvaluationBridge(rfc::learning::RelationalFiniteFieldEvaluation value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::learning::RelationalFiniteFieldEvaluation>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalFiniteFieldEvaluation> owner_;std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
inline std::shared_ptr<FiniteEvaluationBridge> FiniteResultBridge::evaluate(std::size_t ordinal,const std::vector<std::string>& free) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));rfc::numeric::RationalVector values;
    guard.dimension(free.size());
    for(const auto& text:free) {
        rfc::numeric::Rational q;
        if(q.set_str(text,16)!=0 || q.get_den()==0)throw std::invalid_argument("exact complete finite free values required");
        q.canonicalize();guard.value(q);values.push_back(std::move(q));
    }
    return FiniteEvaluationBridge::adopt(owner_->evaluate(ordinal,std::move(values),program_bridge_detail::limits(bounds_)),bounds_,maximum_);
}
} // namespace rexgraph::python
