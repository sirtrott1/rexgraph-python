#pragma once
#include "finite_field_bridge.hpp"
#include <rfc/inference/relational_finite_family.hpp>
#include <rfc/inference/relational_finite_family_gluing.hpp>
#include <stop_token>
#include <map>
#include <set>
#include <bit>

namespace rexgraph::python {
class FiniteSearchStopBridge {
public:
    void cancel() { source_.request_stop(); }
    std::stop_token token() const { return source_.get_token(); }
private:
    std::stop_source source_;
};
struct FiniteDomainRow {
    std::size_t grade{}, cell{};
    std::vector<std::uint64_t> values;
};
struct FiniteProjectionInfo {
    std::uint64_t key{}, revision{};
    std::vector<std::vector<std::size_t>> cells, relations;
    bool has_window{};
    std::vector<std::vector<std::uint64_t>> window;
    std::vector<FiniteAssignmentValue> fixed;
};
struct FiniteComparisonInfo {
    std::uint64_t key{}, revision{};
    std::vector<std::size_t> positions;
    std::vector<std::vector<std::uint64_t>> symbols;
};
struct FiniteFamilyInfo {
    std::uint64_t key{}, revision{};
    std::size_t leaves{}, retained_entries{};
    std::vector<std::vector<std::size_t>> cells;
    std::vector<std::vector<std::uint64_t>> carriers, window;
};
struct FiniteQuestionInfo {
    unsigned status{}; // unique, alternatives, inconsistent_family, unsupported, budget_exceeded, cancelled
    std::size_t answers{}, observations{}, max_answers{}, checks{}, retained_values{};
    bool has_answer{}, has_obstruction{};
    std::size_t left{}, right{};
    std::vector<std::uint64_t> answer;
};
struct FiniteClaimVerdict {
    unsigned status{}; // entailed, refuted, alternatives, inconsistent_family, unsupported
    std::size_t satisfying{}, refuting{};
};
struct FinitePropagationEvidenceInfo {
    std::size_t leaf{}, variables{};
    unsigned status{};
    bool reused{}, has_snapshot{};
    rfc::local::FiniteSearchReceipt work;
};
struct FinitePruningInfo {
    bool present{}, complete{}, has_unsatisfiable_leaf{};
    std::size_t unsatisfiable_leaf{};
    unsigned status{};
    std::vector<std::size_t> limits;
    std::vector<std::vector<std::uint64_t>> window;
    std::vector<FinitePropagationEvidenceInfo> propagations;
};
class FinitePreparedSearchBridge;
class FinitePropagationBridge;
class FiniteFamilyBridge;
class FiniteExtensionBridge;
class FiniteImageBridge;
class FiniteImageQueriesBridge;
struct FiniteFamilyOrigin {
    bool raw{};
    std::shared_ptr<const FiniteFamilyOrigin> left,right;
};
struct FinitePreparedReply {
    unsigned status{};
    std::shared_ptr<FinitePreparedSearchBridge> prepared;
    rfc::local::FiniteSearchReceipt work;
};
struct FinitePropagationReply {
    unsigned status{};
    std::shared_ptr<FinitePropagationBridge> snapshot;
    rfc::local::FiniteSearchReceipt work;
};
struct FiniteSectionReply {
    unsigned status{};
    bool has_witness{};
    std::vector<FiniteAssignmentValue> witness;
    rfc::local::FiniteSearchReceipt work;
};
struct FiniteExtensionReply {
    unsigned status{};
    std::shared_ptr<FiniteExtensionBridge> extension;
    bool has_decisive_leaf{};
    std::size_t decisive_leaf{};
    rfc::inference::RelationalFiniteFamilyWork work;
};
struct FiniteImageReply {
    unsigned status{};
    std::shared_ptr<FiniteImageBridge> image;
    std::shared_ptr<FiniteImageQueriesBridge> queries;
    rfc::inference::RelationalFiniteFamilyWork work;
};
namespace finite_search_bridge_detail {
using namespace rfc;
inline std::stop_token token(const std::shared_ptr<FiniteSearchStopBridge>& stop) {
    return stop ? stop->token() : std::stop_token{};
}
inline local::FiniteSearchLimits search_limits(const std::vector<std::size_t>& values) {
    if(values.size()!=3) throw std::invalid_argument("three finite search limits required");
    return {values[0],values[1],values[2]};
}
inline inference::RelationalFiniteFamilyLimits family_limits(const std::vector<std::size_t>& values) {
    if(values.size()!=5) throw std::invalid_argument("five finite family limits required");
    return {values[0],values[1],values[2],values[3],values[4]};
}
inline inference::OwnedRelationalFiniteSource::Assignment assignment(const std::vector<FiniteAssignmentValue>& rows) {
    inference::OwnedRelationalFiniteSource::Assignment out;
    for(const auto& v:rows)
        if(!out.emplace(program_bridge_detail::cell(v.grade,v.cell),v.symbol).second)
            throw std::invalid_argument("duplicate fixed finite cell");
    return out;
}
inline std::vector<core::RelationId> relations(const std::vector<std::vector<std::size_t>>& rows) {
    std::vector<core::RelationId> out;
    for(const auto& v:finite_bridge_detail::cells(rows)) out.push_back({v.grade,v.index});
    return out;
}
inline std::vector<FiniteAssignmentValue> field(const inference::OwnedRelationalFiniteSource& source,
        const inference::OwnedRelationalFiniteSource::Assignment& values,
        std::size_t begin,std::size_t count,std::size_t maximum) {
    const auto& variables=source.declaration().variables;
    evidence_bridge_detail::bound(count,maximum);
    if(begin>variables.size() || count>variables.size()-begin)
        throw std::out_of_range("finite field slice outside original coordinate order");
    std::vector<FiniteAssignmentValue> out;
    for(std::size_t i=begin;i<begin+count;++i) {
        const auto c=variables[i].cell;
        out.push_back({c.grade.value,c.index,values.at(c)});
    }
    return out;
}
inline FiniteSectionReply section(const inference::OwnedRelationalFiniteSource& source,
        const inference::OwnedRelationalFiniteSource::BoundedSearchResult& result) {
    FiniteSectionReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
    out.has_witness=result.witness.has_value();
    if(result.witness) {
        const auto n=source.declaration().variables.size();
        out.witness=field(source,*result.witness,0,n,n);
    }
    return out;
}
inline inference::RelationalFiniteProjectionDeclaration projection(const FiniteProjectionInfo& value) {
    inference::RelationalFiniteProjectionDeclaration out;
    out.key=value.key;out.revision=value.revision;out.cells=finite_bridge_detail::cells(value.cells);
    out.distinct_relations=relations(value.relations);out.fixed=assignment(value.fixed);
    if(value.has_window) out.window=value.window;
    return out;
}
inline void charge(std::size_t& used,std::size_t amount,std::size_t maximum) {
    used=machine::checked_size_add(used,amount);evidence_bridge_detail::bound(used,maximum);
}
inline FiniteProjectionInfo projection_info(const inference::RelationalFiniteProjectionDeclaration& value,std::size_t maximum) {
    FiniteProjectionInfo out;out.key=value.key;out.revision=value.revision;
    std::size_t used{};
    charge(used,value.cells.size(),maximum);charge(used,value.distinct_relations.size(),maximum);charge(used,value.fixed.size(),maximum);
    for(const auto c:value.cells) out.cells.push_back({c.grade.value,c.index});
    for(const auto c:value.distinct_relations) out.relations.push_back({c.grade.value,c.index});
    for(const auto& [c,s]:value.fixed) out.fixed.push_back({c.grade.value,c.index,s});
    out.has_window=value.window.has_value();
    if(value.window) for(const auto& row:*value.window) {
        charge(used,machine::checked_size_add(row.size(),1),maximum);out.window.push_back(row);
    }
    return out;
}
} // namespace finite_search_bridge_detail

class FinitePreparedSearchBridge {
public:
    FinitePreparedSearchBridge(rfc::inference::PreparedRelationalFiniteSearch owner,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum) {}
    static FinitePreparedReply prepare(std::shared_ptr<FiniteSourceBridge> source,
            const std::vector<std::vector<std::size_t>>& groups,const std::vector<std::size_t>& limits,
            std::shared_ptr<FiniteSearchStopBridge> stop,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace finite_search_bridge_detail;
        if(!source) throw std::invalid_argument("actual finite source owner required");
        const auto result=source->native_owner().prepare_global_section(relations(groups),search_limits(limits),token(stop));
        FinitePreparedReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
        if(result.prepared) out.prepared=std::make_shared<FinitePreparedSearchBridge>(*result.prepared,bounds,maximum);
        return out;
    }
    const rfc::inference::PreparedRelationalFiniteSearch& native_owner() const & {return owner_;}
    const rfc::inference::PreparedRelationalFiniteSearch& native_owner() const && = delete;
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_.source(),bounds_,maximum_);}
    bool same_owner(const FinitePreparedSearchBridge& other) const {return owner_.same_owner(other.owner_);}
    std::size_t retained_entries() const {return owner_.retained_entries();}
    rfc::local::FiniteSearchReceipt work() const {return owner_.preparation_work();}
    std::vector<std::vector<std::size_t>> groups(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_.relations().size(),maximum);
        std::vector<std::vector<std::size_t>> out;
        for(const auto v:owner_.relations()) out.push_back({v.grade.value,v.index});
        return out;
    }
    FiniteSectionReply solve(const std::vector<FiniteAssignmentValue>& fixed,const std::vector<std::size_t>& limits,
            std::shared_ptr<FiniteSearchStopBridge> stop) const {
        using namespace finite_search_bridge_detail;
        return section(owner_.source(),owner_.global_section(assignment(fixed),search_limits(limits),token(stop)));
    }
    FinitePropagationReply propagate(const std::vector<FiniteAssignmentValue>& fixed,
            const std::vector<std::size_t>& limits,std::shared_ptr<FiniteSearchStopBridge> stop) const;
private:
    rfc::inference::PreparedRelationalFiniteSearch owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
class FinitePropagationBridge {
public:
    FinitePropagationBridge(rfc::inference::OwnedRelationalFinitePropagation owner,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum) {}
    const rfc::inference::OwnedRelationalFinitePropagation& native_owner() const & {return owner_;}
    const rfc::inference::OwnedRelationalFinitePropagation& native_owner() const && = delete;
    std::shared_ptr<FinitePreparedSearchBridge> prepared() const {
        return std::make_shared<FinitePreparedSearchBridge>(owner_.prepared(),bounds_,maximum_);
    }
    std::shared_ptr<FiniteSourceBridge> source() const {return FiniteSourceBridge::adopt(owner_.source(),bounds_,maximum_);}
    bool same_owner(const FinitePropagationBridge& other) const {return owner_.same_owner(other.owner_);}
    std::size_t retained_entries() const {return owner_.retained_entries();}
    std::size_t size() const {return owner_.source().declaration().variables.size();}
    rfc::local::FiniteSearchReceipt work() const {return owner_.propagation_work();}
    std::vector<FiniteAssignmentValue> fixed(std::size_t maximum) const {
        evidence_bridge_detail::bound(owner_.fixed().size(),maximum);
        std::vector<FiniteAssignmentValue> out;
        for(const auto& v:owner_.source().declaration().variables)
            if(owner_.fixed().contains(v.cell)) out.push_back({v.cell.grade.value,v.cell.index,owner_.fixed().at(v.cell)});
        return out;
    }
    std::vector<FiniteDomainRow> domains(std::size_t begin,std::size_t count,std::size_t maximum) const {
        const auto& variables=owner_.source().declaration().variables;
        if(begin>variables.size() || count>variables.size()-begin) throw std::out_of_range("finite domain slice outside coordinates");
        evidence_bridge_detail::bound(count,maximum);
        std::size_t used=count;std::vector<FiniteDomainRow> out;
        for(std::size_t i=begin;i<begin+count;++i) {
            const auto c=variables[i].cell;const auto& values=owner_.domains().at(c);
            used=rfc::machine::checked_size_add(used,values.size());evidence_bridge_detail::bound(used,maximum);
            out.push_back({c.grade.value,c.index,values});
        }
        return out;
    }
    FiniteSectionReply solve(const std::vector<FiniteAssignmentValue>& fixed,const std::vector<std::size_t>& limits,
            std::shared_ptr<FiniteSearchStopBridge> stop) const {
        using namespace finite_search_bridge_detail;
        return section(owner_.source(),owner_.continue_global_section(assignment(fixed),search_limits(limits),token(stop)));
    }
private:
    rfc::inference::OwnedRelationalFinitePropagation owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
inline FinitePropagationReply FinitePreparedSearchBridge::propagate(const std::vector<FiniteAssignmentValue>& fixed,
        const std::vector<std::size_t>& limits,std::shared_ptr<FiniteSearchStopBridge> stop) const {
    using namespace finite_search_bridge_detail;
    const auto result=owner_.propagate(assignment(fixed),search_limits(limits),token(stop));
    FinitePropagationReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
    if(result.snapshot) out.snapshot=std::make_shared<FinitePropagationBridge>(*result.snapshot,bounds_,maximum_);
    return out;
}

class FiniteFamilyBridge {
public:
    FiniteFamilyBridge(rfc::inference::OwnedRelationalFiniteFamily owner,std::vector<std::size_t> bounds,std::size_t maximum,
            bool raw_recipe=false,std::shared_ptr<const FiniteFamilyOrigin> origin={})
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum),
         origin_(origin?std::move(origin):std::make_shared<const FiniteFamilyOrigin>(FiniteFamilyOrigin{raw_recipe,{},{}})) {}
    static std::shared_ptr<FiniteFamilyBridge> projection(std::shared_ptr<FiniteSourceBridge> source,
            std::shared_ptr<FinitePreparedSearchBridge> prepared,std::shared_ptr<FinitePropagationBridge> root,
            const FiniteProjectionInfo& value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        const auto d=finite_search_bridge_detail::projection(value);
        const auto limits=program_bridge_detail::limits(bounds);
        if(root) return std::make_shared<FiniteFamilyBridge>(rfc::inference::OwnedRelationalFiniteFamily::projection(root->native_owner(),d,limits),bounds,maximum);
        if(prepared) return std::make_shared<FiniteFamilyBridge>(rfc::inference::OwnedRelationalFiniteFamily::projection(prepared->native_owner(),d,limits),bounds,maximum);
        if(!source) throw std::invalid_argument("actual finite source, prepared search or propagation owner required");
        return std::make_shared<FiniteFamilyBridge>(rfc::inference::OwnedRelationalFiniteFamily::projection(source->native_owner(),d,limits),bounds,maximum,true);
    }
    static std::shared_ptr<FiniteFamilyBridge> intersect(std::shared_ptr<FiniteFamilyBridge> left,
            std::shared_ptr<FiniteFamilyBridge> right,const FiniteComparisonInfo& value,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        if(!left || !right) throw std::invalid_argument("two actual finite family owners required");
        rfc::inference::RelationalFiniteInterfaceBijection map{value.key,value.revision,value.positions,value.symbols};
        return std::make_shared<FiniteFamilyBridge>(rfc::inference::OwnedRelationalFiniteFamily::intersect_images(
            left->owner_,right->owner_,map,program_bridge_detail::limits(bounds)),bounds,maximum,false,
            std::make_shared<const FiniteFamilyOrigin>(FiniteFamilyOrigin{false,left->origin_,right->origin_}));
    }
    const rfc::inference::OwnedRelationalFiniteFamily& native_owner() const & {return owner_;}
    const rfc::inference::OwnedRelationalFiniteFamily& native_owner() const && = delete;
    bool same_owner(const FiniteFamilyBridge& other) const {return owner_.same_owner(other.owner_);}
    bool raw_recipe() const {return origin_->raw;}
    std::shared_ptr<const FiniteFamilyOrigin> origin() const {return origin_;}
    std::shared_ptr<FiniteSourceBridge> source(std::size_t leaf) const {return FiniteSourceBridge::adopt(owner_.source(leaf),bounds_,maximum_);}
    std::shared_ptr<FinitePropagationBridge> propagation(std::size_t leaf) const {
        const auto& value=owner_.propagation(leaf);
        return value ? std::make_shared<FinitePropagationBridge>(*value,bounds_,maximum_) : nullptr;
    }
    std::shared_ptr<FiniteFamilyBridge> parent(bool right) const {
        return std::make_shared<FiniteFamilyBridge>(owner_.parent(right),bounds_,maximum_,false,
            right?origin_->right:origin_->left);
    }
    FiniteProjectionInfo declaration(std::size_t leaf,std::size_t maximum) const {
        return finite_search_bridge_detail::projection_info(owner_.declaration(leaf),maximum);
    }
    bool has_comparison() const {return owner_.comparison().has_value();}
    FiniteComparisonInfo comparison(std::size_t maximum) const {
        const auto& d=owner_.comparison().value();std::size_t used{};
        finite_search_bridge_detail::charge(used,d.right_positions.size(),maximum);
        FiniteComparisonInfo out;out.key=d.key;out.revision=d.revision;out.positions=d.right_positions;
        for(const auto& row:d.right_symbols) {
            finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);
            out.symbols.push_back(row);
        }
        return out;
    }
    FiniteFamilyInfo source_mapping(std::size_t leaf,std::size_t maximum) const {
        FiniteFamilyInfo out;std::size_t used{};
        finite_search_bridge_detail::charge(used,owner_.source_cells(leaf).size(),maximum);
        for(const auto c:owner_.source_cells(leaf)) out.cells.push_back({c.grade.value,c.index});
        for(const auto& row:owner_.source_symbols(leaf)) {
            finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);
            out.carriers.push_back(row);
        }
        return out;
    }
    FiniteFamilyInfo info(std::size_t maximum) const {
        FiniteFamilyInfo out;out.key=owner_.key();out.revision=owner_.revision();
        out.leaves=owner_.leaf_count();out.retained_entries=owner_.retained_entries();
        std::size_t used=owner_.cells().size();evidence_bridge_detail::bound(used,maximum);
        for(std::size_t i=0;i<owner_.cells().size();++i) {
            used=rfc::machine::checked_size_add(used,rfc::machine::checked_size_add(owner_.carriers()[i].size(),owner_.window()[i].size()));
            evidence_bridge_detail::bound(used,maximum);
            const auto c=owner_.cells()[i];out.cells.push_back({c.grade.value,c.index});
            out.carriers.push_back(owner_.carriers()[i]);out.window.push_back(owner_.window()[i]);
        }
        return out;
    }
    FiniteExtensionReply extend(const std::vector<std::uint64_t>& tuple,const std::vector<std::size_t>& limits,
            std::shared_ptr<FiniteSearchStopBridge> stop) const;
    FiniteImageReply materialize(const std::vector<std::size_t>& limits,std::shared_ptr<FiniteSearchStopBridge> stop) const;
private:
    rfc::inference::OwnedRelationalFiniteFamily owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
    std::shared_ptr<const FiniteFamilyOrigin> origin_;
};
class FiniteExtensionBridge {
public:
    FiniteExtensionBridge(rfc::inference::RelationalFiniteFamilyExtension owner,std::vector<std::size_t> bounds,std::size_t maximum,
            std::shared_ptr<const FiniteFamilyOrigin> origin={})
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum),origin_(std::move(origin)) {}
    const rfc::inference::RelationalFiniteFamilyExtension& native_owner() const & {return owner_;}
    const rfc::inference::RelationalFiniteFamilyExtension& native_owner() const && = delete;
    std::shared_ptr<FiniteFamilyBridge> family() const {return std::make_shared<FiniteFamilyBridge>(owner_.family(),bounds_,maximum_,false,origin_);}
    std::vector<std::uint64_t> tuple(std::size_t maximum) const {evidence_bridge_detail::bound(owner_.tuple().size(),maximum);return owner_.tuple();}
    std::size_t leaf_count() const {return owner_.leaf_count();}
    std::size_t field_size(std::size_t leaf) const {return owner_.family().source(leaf).declaration().variables.size();}
    std::vector<FiniteAssignmentValue> witness(std::size_t leaf,std::size_t begin,std::size_t count,std::size_t maximum) const {
        return finite_search_bridge_detail::field(owner_.family().source(leaf),owner_.witness(leaf),begin,count,maximum);
    }
private:
    rfc::inference::RelationalFiniteFamilyExtension owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
    std::shared_ptr<const FiniteFamilyOrigin> origin_;
};
// This replay packet is a bridge certificate, not an RFC source archive or authority.
namespace finite_search_bridge_detail {
class ProofWriter {
public:
    explicit ProofWriter(std::size_t maximum):maximum_(maximum) {}
    void word(std::uint64_t value) {
        admit(8);
        for(unsigned i=0;i<8;++i) data_.push_back(static_cast<char>((value>>(8*i))&255));
    }
    void bytes(const std::string& value) {
        word(value.size());admit(value.size());data_.append(value);
    }
    void bytes(const std::vector<std::byte>& value) {
        word(value.size());admit(value.size());
        data_.append(reinterpret_cast<const char*>(value.data()),value.size());
    }
    template<class Values> void words(const Values& values) {
        word(values.size());for(const auto value:values) word(value);
    }
    std::string finish() && {return std::move(data_);}
private:
    void admit(std::size_t count) const {
        if(count>maximum_ || data_.size()>maximum_-count)
            throw std::length_error("finite proof exceeds byte admission");
    }
    std::string data_;std::size_t maximum_;
};
inline void proof_family(ProofWriter& out,const rfc::inference::OwnedRelationalFiniteFamily& family) {
    // Prepared roots and intersections need their own explicit replay recipes.
    if(family.leaf_count()!=1 || family.comparison() || family.propagation(0))
        throw std::invalid_argument("finite proof packet requires an unprepared single-source projection");
    out.bytes("rexgraph.finite-projection-proof.v2");
    out.bytes(rfc::inference::encode_relational_finite_source(family.source(0)));
    const auto& d=family.declaration(0);out.word(d.key);out.word(d.revision);
    out.word(d.cells.size());for(const auto c:d.cells) {out.word(c.grade.value);out.word(c.index);}
    out.word(d.window.has_value());
    if(d.window) {out.word(d.window->size());for(const auto& values:*d.window) out.words(values);}
    out.word(d.fixed.size());for(const auto& variable:family.source(0).declaration().variables) {
        const auto found=d.fixed.find(variable.cell);if(found==d.fixed.end()) continue;
        out.word(variable.cell.grade.value);out.word(variable.cell.index);out.word(found->second);
    }
    out.word(d.distinct_relations.size());for(const auto relation:d.distinct_relations) {
        out.word(relation.grade.value);out.word(relation.index);
    }
}
inline void proof_search_work(ProofWriter& out,const rfc::local::FiniteSearchReceipt& w) {
    for(const auto value:{w.steps,w.value_trials,w.chart_tests,w.tuple_tests,w.coordinate_tests,
            w.variables,w.peak_assigned,w.peak_entries,w.domain_reductions,w.search_nodes,
            w.distinct_groups,w.matching_tests}) out.word(value);
}
inline void proof_family_work(ProofWriter& out,const rfc::inference::RelationalFiniteFamilyWork& w) {
    for(const auto value:{w.steps,w.peak_entries,w.candidates,w.completed_queries,w.preparations,
            w.source_queries,w.rows,w.witness_values,w.reused_preparations,w.prior_preparation_steps,
            w.reused_propagations,w.prior_propagation_steps,w.continued_source_queries,
            w.propagations,w.pruned_values}) out.word(value);
    proof_search_work(out,w.native);
}
}
class FiniteImageQueriesBridge {
public:
    FiniteImageQueriesBridge(rfc::inference::OwnedRelationalFiniteFamily family,
            std::vector<rfc::inference::RelationalFiniteImageQuery> queries,std::vector<std::size_t> bounds,std::size_t maximum,
            rfc::inference::RelationalFiniteImageStatus status,rfc::inference::RelationalFiniteFamilyWork work,
            std::vector<std::size_t> limits,std::shared_ptr<const rfc::inference::RelationalFiniteImagePruning> pruning,
            std::shared_ptr<const FiniteFamilyOrigin> origin)
        :family_(std::move(family)),queries_(std::move(queries)),bounds_(std::move(bounds)),maximum_(maximum),
         status_(status),work_(work),limits_(std::move(limits)),pruning_(std::move(pruning)),origin_(std::move(origin)) {}
    std::shared_ptr<FiniteFamilyBridge> family() const {return std::make_shared<FiniteFamilyBridge>(family_,bounds_,maximum_,false,origin_);}
    std::shared_ptr<const FiniteFamilyOrigin> origin() const {return origin_;}
    std::size_t query_count() const {return queries_.size();}
    FinitePruningInfo pruning(std::size_t maximum) const {
        FinitePruningInfo out;out.present=static_cast<bool>(pruning_);out.status=static_cast<unsigned>(status_);out.limits=limits_;
        if(!pruning_) return out;
        out.complete=pruning_->complete;out.has_unsatisfiable_leaf=pruning_->unsatisfiable_leaf.has_value();
        out.unsatisfiable_leaf=pruning_->unsatisfiable_leaf.value_or(0);
        std::size_t used=pruning_->propagations.size();evidence_bridge_detail::bound(used,maximum);
        for(const auto& values:pruning_->window) {
            finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(values.size(),1),maximum);
            out.window.push_back(values);
        }
        for(const auto& p:pruning_->propagations)
            out.propagations.push_back({p.leaf,p.snapshot?p.snapshot->source().declaration().variables.size():0,
                static_cast<unsigned>(p.status),p.reused,p.snapshot.has_value(),p.work});
        return out;
    }
    std::shared_ptr<FinitePropagationBridge> pruning_root(std::size_t index) const {
        if(!pruning_) throw std::out_of_range("finite materialization has no pruning records");
        const auto& p=pruning_->propagations.at(index);
        return p.snapshot?std::make_shared<FinitePropagationBridge>(*p.snapshot,bounds_,maximum_):nullptr;
    }
    void write_proof(finite_search_bridge_detail::ProofWriter& out) const {
        using namespace finite_search_bridge_detail;
        if(!origin_ || !origin_->raw) throw std::invalid_argument("finite proof packet requires an unprepared single-source projection");
        finite_search_bridge_detail::proof_family(out,family_);
        out.words(bounds_);out.word(maximum_);out.words(limits_);out.word(static_cast<unsigned>(status_));
        proof_family_work(out,work_);
        out.word(static_cast<bool>(pruning_));
        if(pruning_) {
            out.word(pruning_->complete);out.word(pruning_->window.size());
            for(const auto& values:pruning_->window) out.words(values);
            out.word(pruning_->unsatisfiable_leaf.has_value());
            if(pruning_->unsatisfiable_leaf) out.word(*pruning_->unsatisfiable_leaf);
            out.word(pruning_->propagations.size());
            for(const auto& p:pruning_->propagations) {
                const auto& source=family_.source(p.leaf);const auto& d=family_.declaration(p.leaf);
                out.word(p.leaf);out.word(static_cast<unsigned>(p.status));out.word(p.reused);proof_search_work(out,p.work);
                out.word(p.snapshot.has_value());
                if(!p.snapshot) continue;
                const auto& root=*p.snapshot;
                if(!root.source().same_owner(source) || root.fixed()!=d.fixed || root.prepared().relations()!=d.distinct_relations)
                    throw std::invalid_argument("pruning root differs from its actual source request");
                proof_search_work(out,root.prepared().preparation_work());proof_search_work(out,root.propagation_work());
                out.word(root.retained_entries());out.word(d.fixed.size());
                for(const auto& variable:source.declaration().variables) {
                    const auto fixed=d.fixed.find(variable.cell);if(fixed==d.fixed.end()) continue;
                    out.word(variable.cell.grade.value);out.word(variable.cell.index);out.word(fixed->second);
                }
                out.word(d.distinct_relations.size());
                for(const auto group:d.distinct_relations) {out.word(group.grade.value);out.word(group.index);}
                out.word(source.declaration().variables.size());
                for(const auto& variable:source.declaration().variables) {
                    out.word(variable.cell.grade.value);out.word(variable.cell.index);out.words(root.domains().at(variable.cell));
                }
            }
        }
        out.word(queries_.size());
        for(const auto& query:queries_) {
            out.words(query.tuple);out.word(static_cast<unsigned>(query.status));
            out.word(query.source_queries);out.word(query.native_steps);
            out.word(query.decisive_leaf.has_value());
            if(query.decisive_leaf) out.word(*query.decisive_leaf);
        }
    }
    std::string proof_archive(std::size_t maximum) const {
        finite_search_bridge_detail::ProofWriter out(maximum);write_proof(out);out.word(0);
        return std::move(out).finish();
    }
    std::vector<rfc::inference::RelationalFiniteImageQuery> queries(std::size_t begin,std::size_t count,std::size_t maximum) const {
        if(begin>queries_.size() || count>queries_.size()-begin) throw std::out_of_range("finite image query slice outside receipt");
        std::size_t used=count;evidence_bridge_detail::bound(used,maximum);
        for(std::size_t i=begin;i<begin+count;++i) {used=rfc::machine::checked_size_add(used,queries_[i].tuple.size());evidence_bridge_detail::bound(used,maximum);}
        return {queries_.begin()+static_cast<std::ptrdiff_t>(begin),queries_.begin()+static_cast<std::ptrdiff_t>(begin+count)};
    }
private:
    rfc::inference::OwnedRelationalFiniteFamily family_;
    std::vector<rfc::inference::RelationalFiniteImageQuery> queries_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
    rfc::inference::RelationalFiniteImageStatus status_;
    rfc::inference::RelationalFiniteFamilyWork work_;
    std::vector<std::size_t> limits_;
    std::shared_ptr<const rfc::inference::RelationalFiniteImagePruning> pruning_;
    std::shared_ptr<const FiniteFamilyOrigin> origin_;
};
class FiniteImageBridge {
public:
    FiniteImageBridge(rfc::inference::OwnedRelationalFiniteImage owner,std::shared_ptr<FiniteImageQueriesBridge> queries,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(owner)),queries_(std::move(queries)),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<FiniteFamilyBridge> family() const {return queries_->family();}
    FinitePruningInfo pruning(std::size_t maximum) const {return queries_->pruning(maximum);}
    std::shared_ptr<FinitePropagationBridge> pruning_root(std::size_t index) const {return queries_->pruning_root(index);}
    std::size_t size() const {return owner_.size();}
    std::vector<FiniteClaimVerdict> classify_equalities(
            const std::vector<std::vector<FiniteAssignmentValue>>& claims,std::size_t maximum) const {
        using finite_search_bridge_detail::charge;
        const auto& family=owner_.family();std::size_t used{};
        charge(used,claims.size(),maximum);
        std::vector<FiniteClaimVerdict> result;result.reserve(claims.size());
        for(const auto& claim:claims) {
            std::vector<std::pair<std::size_t,std::uint64_t>> positions;
            bool supported=true;
            for(const auto& term:claim) {
                std::size_t position=family.cells().size();
                for(std::size_t i=0;i<family.cells().size();++i) {
                    charge(used,1,maximum);const auto cell=family.cells()[i];
                    if(cell.grade.value==term.grade && cell.index==term.cell) {position=i;break;}
                }
                if(position==family.cells().size()) {supported=false;continue;}
                bool admitted=false;
                for(const auto value:family.carriers()[position]) {
                    charge(used,1,maximum);if(value==term.symbol) {admitted=true;break;}
                }
                if(!admitted) supported=false;
                positions.emplace_back(position,term.symbol);
            }
            FiniteClaimVerdict verdict;
            if(!supported) {verdict.status=4;result.push_back(verdict);continue;}
            if(!owner_.size()) {verdict.status=3;result.push_back(verdict);continue;}
            for(std::size_t i=0;i<owner_.size();++i) {
                charge(used,1,maximum);const auto& tuple=owner_.row(i).tuple();bool yes=true;
                for(const auto& [position,value]:positions) {
                    charge(used,1,maximum);if(tuple[position]!=value) yes=false;
                }
                if(yes) ++verdict.satisfying;else ++verdict.refuting;
            }
            verdict.status=verdict.satisfying ? (verdict.refuting ? 2:0):1;
            result.push_back(verdict);
        }
        return result;
    }
    FiniteQuestionInfo analyze_question(const std::vector<std::vector<std::size_t>>& question,
            const std::vector<std::vector<std::size_t>>& observation,std::size_t maximum,
            std::shared_ptr<FiniteSearchStopBridge> stop) const {
        struct Interrupted {unsigned status;};
        struct Work {
            std::size_t used{}, entries{}, maximum{}, entry_maximum{};std::stop_token stop;
            void step(std::size_t count=1) {
                if(stop.stop_requested()) throw Interrupted{5};
                if(count>maximum || used>maximum-count) throw Interrupted{4};
                used+=count;
            }
            void retain(std::size_t count) {
                if(count>entry_maximum || entries>entry_maximum-count) throw Interrupted{4};
                entries+=count;
            }
        } work{0,0,maximum,bounds_.at(1),finite_search_bridge_detail::token(stop)};
        using Key=std::vector<std::uint64_t>;
        struct Less {
            Work* work;
            bool operator()(const Key& a,const Key& b) const {
                for(std::size_t i=0;i<std::min(a.size(),b.size());++i) {
                    work->step();if(a[i]!=b[i]) return a[i]<b[i];
                }
                work->step();return a.size()<b.size();
            }
        };
        struct Fiber {std::size_t first{}, first_answer{};std::set<std::size_t> answers;};
        FiniteQuestionInfo out;
        try {
            work.step();
            if(question.empty()) throw std::invalid_argument("finite question requires declared answer cells");
            if(question.size()>bounds_.at(0) || observation.size()>bounds_.at(0))
                throw std::length_error("finite question exceeds coordinate admission");
            const auto& cells=owner_.family().cells();bool supported=true;
            auto positions=[&](const auto& requested) {
                std::vector<std::size_t> result;std::set<std::pair<std::size_t,std::size_t>> seen;
                for(const auto& c:requested) {
                    work.step();if(c.size()!=2) throw std::invalid_argument("finite question cell arity");
                    if(!seen.emplace(c[0],c[1]).second) throw std::invalid_argument("finite question repeats a cell");
                    auto position=cells.size();
                    for(std::size_t i=0;i<cells.size();++i) {
                        work.step();if(cells[i].grade.value==c[0] && cells[i].index==c[1]) {position=i;break;}
                    }
                    if(position==cells.size()) supported=false;
                    result.push_back(position);
                }
                return result;
            };
            const auto q=positions(question),m=positions(observation);
            if(!supported) {out.status=3;out.checks=work.used;return out;}
            if(!owner_.size()) {out.status=2;out.checks=work.used;return out;}
            std::map<Key,std::size_t,Less> answers{Less{&work}};
            std::map<Key,Fiber,Less> fibers{Less{&work}};
            for(std::size_t i=0;i<owner_.size();++i) {
                work.step();const auto& row=owner_.row(i).tuple();Key a,b;
                for(const auto j:q) {work.step();a.push_back(row.at(j));}
                for(const auto j:m) {work.step();b.push_back(row.at(j));}
                const auto [ai,new_answer]=answers.try_emplace(a,answers.size());
                if(new_answer) work.retain(a.size()+1);
                const auto [mi,new_fiber]=fibers.try_emplace(b,Fiber{i,ai->second,{}});
                if(new_fiber) work.retain(b.size()+3);
                auto& fiber=mi->second;
                work.step(1+std::bit_width(fiber.answers.size()));
                if(fiber.answers.insert(ai->second).second) work.retain(1);
                out.max_answers=std::max(out.max_answers,fiber.answers.size());
                if(!out.has_obstruction && fiber.first_answer!=ai->second) {
                    out.has_obstruction=true;out.left=fiber.first;out.right=i;
                }
            }
            out.answers=answers.size();out.observations=fibers.size();
            out.status=out.answers==1?0:1;
            if(out.answers==1) {out.has_answer=true;out.answer=answers.begin()->first;}
        } catch(const Interrupted& reason) {
            out=FiniteQuestionInfo{};out.status=reason.status;
        }
        out.checks=work.used;out.retained_values=work.entries;
        return out;
    }
    std::string proof_archive(std::size_t maximum) const {
        finite_search_bridge_detail::ProofWriter out(maximum);queries_->write_proof(out);
        out.word(1);out.word(owner_.size());
        for(std::size_t i=0;i<owner_.size();++i) {
            const auto& row=owner_.row(i);out.words(row.tuple());out.word(row.leaf_count());
            for(std::size_t leaf=0;leaf<row.leaf_count();++leaf) {
                const auto& witness=row.witness(leaf);
                const auto& variables=row.family().source(leaf).declaration().variables;
                out.word(variables.size());
                for(const auto& variable:variables) {
                    out.word(variable.cell.grade.value);out.word(variable.cell.index);
                    out.word(witness.at(variable.cell));
                }
            }
        }
        return std::move(out).finish();
    }
    std::shared_ptr<FiniteExtensionBridge> row(std::size_t index) const {return std::make_shared<FiniteExtensionBridge>(owner_.row(index),bounds_,maximum_,queries_->origin());}
    std::size_t query_count() const {return queries_->query_count();}
    auto queries(std::size_t begin,std::size_t count,std::size_t maximum) const {return queries_->queries(begin,count,maximum);}
private:
    rfc::inference::OwnedRelationalFiniteImage owner_;
    std::shared_ptr<FiniteImageQueriesBridge> queries_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
inline FiniteExtensionReply FiniteFamilyBridge::extend(const std::vector<std::uint64_t>& tuple,
        const std::vector<std::size_t>& limits,std::shared_ptr<FiniteSearchStopBridge> stop) const {
    using namespace finite_search_bridge_detail;
    const auto result=owner_.extend(tuple,family_limits(limits),token(stop));
    FiniteExtensionReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
    out.has_decisive_leaf=result.decisive_leaf.has_value();out.decisive_leaf=result.decisive_leaf.value_or(0);
    if(result.extension) out.extension=std::make_shared<FiniteExtensionBridge>(*result.extension,bounds_,maximum_,origin_);
    return out;
}
inline FiniteImageReply FiniteFamilyBridge::materialize(const std::vector<std::size_t>& limits,
        std::shared_ptr<FiniteSearchStopBridge> stop) const {
    using namespace finite_search_bridge_detail;
    auto result=owner_.materialize(family_limits(limits),token(stop));
    FiniteImageReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
    out.queries=std::make_shared<FiniteImageQueriesBridge>(owner_,std::move(result.queries),bounds_,maximum_,
        result.status,result.work,limits,std::move(result.pruning),origin_);
    if(result.image) out.image=std::make_shared<FiniteImageBridge>(*result.image,out.queries,bounds_,maximum_);
    return out;
}

struct FiniteSourceMapInfo {
    std::vector<std::vector<std::size_t>> cells, occurrences;
};
struct FiniteGluingInfo {
    std::uint64_t key{}, revision{}, symbol_key{}, symbol_revision{};
    FiniteSourceMapInfo common_left, common_right, left_glued, right_glued;
    std::vector<std::vector<std::uint64_t>> left_symbols, right_symbols;
};
class FiniteGluingBridge;
class FiniteGluedSectionBridge;
struct FiniteGluingReply {
    unsigned status{};
    std::shared_ptr<FiniteGluedSectionBridge> section;
    rfc::inference::RelationalFiniteFamilyWork work;
};
class FiniteGluingBridge {
public:
    FiniteGluingBridge(const std::string& common,std::shared_ptr<FiniteSourceBridge> left,
            std::shared_ptr<FiniteSourceBridge> right,std::shared_ptr<FiniteSourceBridge> global,
            const FiniteGluingInfo& value,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :bounds_(bounds),maximum_(maximum) {
        if(!left || !right || !global) throw std::invalid_argument("actual finite gluing sources required");
        using namespace rfc;
        storage::NativeSeriesLimits admission;admission.max_archive_bytes=maximum;
        admission.temporal=program_bridge_detail::reader(bounds,maximum);
        const auto publication=storage::decode_native_series(program_bridge_detail::bytes(common),admission).tail();
        auto map=[](const FiniteSourceMapInfo& in) {
            algebra::RetainedSourceMap out;out.occurrences=in.occurrences;
            for(const auto& grade:in.cells) {
                std::vector<std::uint32_t> row;
                for(const auto cell:grade) {
                    if(cell>std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("finite source map cell overflow");
                    row.push_back(static_cast<std::uint32_t>(cell));
                }
                out.cells.push_back(std::move(row));
            }
            return out;
        };
        algebra::RelationalGluingDeclaration d{value.key,value.revision,map(value.common_left),map(value.common_right),map(value.left_glued),map(value.right_glued)};
        inference::RelationalFiniteGluingSymbols symbols{value.symbol_key,value.symbol_revision,value.left_symbols,value.right_symbols};
        owner_=std::make_shared<const inference::OwnedRelationalFiniteGluing>(inference::OwnedRelationalFiniteGluing::prepare(
            publication.endpoint().source,left->native_owner(),right->native_owner(),global->native_owner(),d,symbols,program_bridge_detail::limits(bounds)));
    }
    FiniteGluingBridge(rfc::inference::OwnedRelationalFiniteGluing owner,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::inference::OwnedRelationalFiniteGluing>(std::move(owner))),bounds_(std::move(bounds)),maximum_(maximum) {}
    bool same_owner(const FiniteGluingBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::size_t retained_entries() const {return owner_->retained_entries();}
    std::size_t preparation_steps() const {return owner_->preparation_steps();}
    FiniteGluingInfo maps(std::size_t maximum) const {
        FiniteGluingInfo out;const auto& d=owner_->structural().declaration();const auto& s=owner_->symbols();
        out.key=d.key;out.revision=d.revision;out.symbol_key=s.key;out.symbol_revision=s.revision;
        std::size_t used{};
        auto copy_map=[&](const rfc::algebra::RetainedSourceMap& value) {
            FiniteSourceMapInfo result;
            for(const auto& row:value.cells) {
                finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);
                result.cells.emplace_back(row.begin(),row.end());
            }
            for(const auto& row:value.occurrences) {
                finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);
                result.occurrences.push_back(row);
            }
            return result;
        };
        out.common_left=copy_map(d.common_left);out.common_right=copy_map(d.common_right);
        out.left_glued=copy_map(d.left_glued);out.right_glued=copy_map(d.right_glued);
        for(const auto& row:s.left) {finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);out.left_symbols.push_back(row);}
        for(const auto& row:s.right) {finite_search_bridge_detail::charge(used,rfc::machine::checked_size_add(row.size(),1),maximum);out.right_symbols.push_back(row);}
        return out;
    }
    std::shared_ptr<FiniteSourceBridge> source(unsigned side) const {
        if(side>2) throw std::out_of_range("finite gluing source must be left, right or global");
        return FiniteSourceBridge::adopt(side==0?owner_->left():side==1?owner_->right():owner_->global(),bounds_,maximum_);
    }
    FiniteGluingReply glue(std::shared_ptr<FiniteExtensionBridge> left,std::size_t left_leaf,
        std::shared_ptr<FiniteExtensionBridge> right,std::size_t right_leaf,const std::vector<std::size_t>& limits,
        std::shared_ptr<FiniteSearchStopBridge> stop) const;
private:
    std::shared_ptr<const rfc::inference::OwnedRelationalFiniteGluing> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
class FiniteGluedSectionBridge {
public:
    FiniteGluedSectionBridge(rfc::inference::OwnedRelationalFiniteGluedSection owner,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::move(owner)),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<FiniteGluingBridge> gluing() const {return std::make_shared<FiniteGluingBridge>(owner_.gluing(),bounds_,maximum_);}
    std::shared_ptr<FiniteExtensionBridge> parent(bool right) const {
        return std::make_shared<FiniteExtensionBridge>(right?owner_.right():owner_.left(),bounds_,maximum_);
    }
    std::size_t leaf(bool right) const {return right?owner_.right_leaf():owner_.left_leaf();}
    std::size_t size() const {return owner_.gluing().global().declaration().variables.size();}
    std::vector<FiniteAssignmentValue> field(std::size_t begin,std::size_t count,std::size_t maximum) const {
        return finite_search_bridge_detail::field(owner_.gluing().global(),owner_.field(),begin,count,maximum);
    }
private:
    rfc::inference::OwnedRelationalFiniteGluedSection owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_;
};
inline FiniteGluingReply FiniteGluingBridge::glue(std::shared_ptr<FiniteExtensionBridge> left,std::size_t left_leaf,
        std::shared_ptr<FiniteExtensionBridge> right,std::size_t right_leaf,const std::vector<std::size_t>& limits,
        std::shared_ptr<FiniteSearchStopBridge> stop) const {
    using namespace finite_search_bridge_detail;
    if(!left || !right) throw std::invalid_argument("actual whole finite extensions required");
    const auto result=owner_->glue(left->native_owner(),left_leaf,right->native_owner(),right_leaf,family_limits(limits),token(stop));
    FiniteGluingReply out;out.status=static_cast<unsigned>(result.status);out.work=result.work;
    if(result.section) out.section=std::make_shared<FiniteGluedSectionBridge>(*result.section,bounds_,maximum_);
    return out;
}
} // namespace rexgraph::python
