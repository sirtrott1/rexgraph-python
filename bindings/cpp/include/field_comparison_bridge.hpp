#pragma once

#include "field_interface_bridge.hpp"
#include <rfc/inference/relational_comparison_forms_codec.hpp>

namespace rexgraph::python {
struct ComparisonWeight { std::size_t grade{}, cell{}; std::string value; };
struct ComparisonDeclaration {
    std::uint64_t key{}, revision{};
    std::size_t grade{};
    unsigned participation{};
    std::vector<ComparisonWeight> weights;
};
struct ComparisonInfo {
    std::uint64_t key{}, revision{}, node{};
    std::size_t grade{}, dimension{}, supported{}, nonzeros{}, occurrences{};
    unsigned participation{}, channel{};
    std::string quadrance;
};
struct ComparisonPairInfo {
    std::size_t first{}, second{}, grade{}, cell{};
    unsigned channel{};
    bool admitted{};
    std::string residual, metric, weight, quadrance;
};
namespace comparison_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
inline constexpr std::array<std::byte,8> bundle_magic{
    std::byte{'R'},std::byte{'G'},std::byte{'C'},std::byte{'A'},std::byte{'R'},std::byte{1},std::byte{0},std::byte{0}};
inline inference::RelationalComparisonChannel channel(unsigned value) {
    if (value > 3) throw std::invalid_argument("explicit T/rawG/C/F comparison channel required");
    return static_cast<inference::RelationalComparisonChannel>(value);
}
inline numeric::Rational rational(const std::string& value, numeric::ExactActionGuard& guard) {
    numeric::Rational q;
    if (q.set_str(value,16) != 0 || q.get_den() == 0) throw std::invalid_argument("exact comparison coefficient required");
    q.canonicalize(); guard.value(q); return q;
}
inline void node_grade(const inference::OwnedRelationalFieldProgram& program, std::uint64_t key, core::Grade grade) {
    const auto& nodes = program.declaration().nodes;
    const auto found = std::find_if(nodes.begin(),nodes.end(),[&](const auto& n){return n.key==key;});
    if (found == nodes.end() || found->output_grade != grade)
        throw std::invalid_argument("comparison input needs its actual original node grade");
}
} // namespace comparison_bridge_detail

class ComparisonResultBridge;
class FieldComparisonBridge {
public:
    FieldComparisonBridge(std::shared_ptr<FieldProgramBridge> program, const ComparisonDeclaration& value,
            const std::vector<std::size_t>& bounds, std::size_t maximum) : bounds_(bounds), maximum_(maximum) {
        using namespace comparison_bridge_detail;
        if (!program) throw std::invalid_argument("actual comparison program owner required");
        numeric::ExactActionGuard guard(limits(bounds)); inference::RelationalComparisonDeclaration d;
        d.key=value.key; d.revision=value.revision; d.grade=cell(value.grade,0).grade;
        d.participation=static_cast<inference::RelationalComparisonParticipation>(value.participation);
        for (const auto& w : value.weights) d.weights.push_back({cell(w.grade,w.cell),rational(w.value,guard)});
        owner_=std::make_shared<const inference::PreparedRelationalComparisonForms>(program->native_owner(),std::move(d),limits(bounds));
    }
    static std::shared_ptr<FieldComparisonBridge> restore(const std::string& image,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        using namespace comparison_bridge_detail;
        return std::shared_ptr<FieldComparisonBridge>(new FieldComparisonBridge(
            inference::decode_relational_comparison_forms(bytes(image),reader(bounds,maximum),limits(bounds)),bounds,maximum));
    }
    std::string archive() const { return program_bridge_detail::bounded(rfc::inference::encode_relational_comparison_forms(*owner_),maximum_); }
    std::shared_ptr<FieldProgramBridge> program() const { return FieldProgramBridge::adopt(owner_->program(),bounds_,maximum_); }
    ComparisonInfo identity() const {
        const auto& d=owner_->declaration();const auto program=owner_->program();const auto publication=program.publication();
        return {d.key,d.revision,0,d.grade.value,program.fibers().fibers().stalks().total_dimension(d.grade),0,0,
            publication.endpoint().source.grade(d.grade).incidence_count(),static_cast<unsigned>(d.participation),0,{}};
    }
    void validate_node(std::uint64_t node) const {
        comparison_bridge_detail::node_grade(owner_->program(),node,owner_->declaration().grade);
    }
    ComparisonDeclaration declaration(std::size_t maximum) const {
        const auto& d=owner_->declaration(); ComparisonDeclaration out;
        if (d.weights.size()>maximum) throw std::length_error("comparison weight metadata bound");
        out.key=d.key;out.revision=d.revision;out.grade=d.grade.value;out.participation=static_cast<unsigned>(d.participation);
        for (const auto& w : d.weights) out.weights.push_back({w.cell.grade.value,w.cell.index,w.value.get_str(16)});
        return out;
    }
    std::string operator_matrix(unsigned selected, std::size_t maximum) const {
        return program_bridge_detail::matrix_bytes(owner_->operator_matrix(comparison_bridge_detail::channel(selected),maximum,program_bridge_detail::limits(bounds_)));
    }
    std::shared_ptr<ComparisonResultBridge> apply(std::shared_ptr<AccessionResultBridge> input, std::uint64_t node, unsigned selected) const;
private:
    FieldComparisonBridge(rfc::inference::PreparedRelationalComparisonForms owner,std::vector<std::size_t> bounds,std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::PreparedRelationalComparisonForms>(std::move(owner))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::PreparedRelationalComparisonForms> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class ComparisonPairBridge;
class ComparisonResultBridge {
public:
    static std::shared_ptr<ComparisonResultBridge> restore(const std::string& image,
            const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace comparison_bridge_detail;
        using namespace inference::field_program_codec_detail;
        const auto policy=reader(bounds,maximum); const auto action=limits(bounds);
        Reader r(bytes(image),policy);header(r,bytes(image),bundle_magic);const auto node=r.u64();
        auto input=inference::decode_accession_field_result(r.blob(),policy,action);
        const auto expected=inference::decode_relational_comparison_result(r.blob(),policy,action);
        if (r.remaining()) throw std::invalid_argument("trailing source-backed comparison bundle bytes");
        const auto actual=input.result();const auto program=actual.program(); const auto forms=expected.forms();
        if (inference::encode_relational_field_program(program)!=inference::encode_relational_field_program(forms.program()))
            throw std::invalid_argument("comparison archive references another complete parent program");
        node_grade(program,node,forms.declaration().grade);
        const inference::PreparedRelationalComparisonForms adopted(program,forms.declaration(),action);
        auto result=adopted.apply(expected.channel(),actual.field(node),action);
        if (inference::encode_relational_comparison_result(result)!=inference::encode_relational_comparison_result(expected))
            throw std::invalid_argument("comparison archive input or whole native action differs from its actual parent");
        auto out=std::shared_ptr<ComparisonResultBridge>(new ComparisonResultBridge(std::move(input),std::move(result),node,bounds,maximum));
        if (out->archive()!=image) throw std::invalid_argument("noncanonical source-backed comparison bundle");
        return out;
    }
    std::string archive() const {
        using namespace rfc::inference::field_program_codec_detail;
        auto w=envelope(comparison_bridge_detail::bundle_magic);w.u64(node_);
        blob(w,rfc::inference::encode_accession_field_result(input_));blob(w,rfc::inference::encode_relational_comparison_result(result_));
        return program_bridge_detail::bounded(finish(std::move(w)),maximum_);
    }
    std::string native_archive() const {return program_bridge_detail::bounded(rfc::inference::encode_relational_comparison_result(result_),maximum_);}
    std::string forms_archive() const {return program_bridge_detail::bounded(rfc::inference::encode_relational_comparison_forms(result_.forms()),maximum_);}
    std::shared_ptr<AccessionResultBridge> input() const {return AccessionResultBridge::adopt(input_,bounds_,maximum_);}
    ComparisonInfo info() const {
        const auto forms=result_.forms();const auto& d=forms.declaration();const auto& f=result_.output();
        return {d.key,d.revision,node_,d.grade.value,f.domain().logical_size(),f.support().count(),f.values().nonzeros(),
            result_.occurrence_fields().size(),static_cast<unsigned>(d.participation),static_cast<unsigned>(result_.channel()),result_.quadrance().get_str(16)};
    }
    std::string field(unsigned role) const {
        const auto* value=role==0?&result_.input():role==1?&result_.output():role==2?&result_.signed_boundary_field():role==3?&result_.unsigned_boundary_field():nullptr;
        if (!value) throw std::invalid_argument("explicit comparison input/output/signed/unsigned field role required");
        const auto image=program_bridge_detail::tensor_bytes(*value);return program_bridge_detail::bounded(program_bridge_detail::bytes(image),maximum_);
    }
    std::string occurrence_field(std::size_t occurrence) const {
        if (occurrence>=result_.occurrence_fields().size()) throw std::out_of_range("original comparison flag index");
        const auto image=program_bridge_detail::tensor_bytes(result_.occurrence_fields()[occurrence]);
        return program_bridge_detail::bounded(program_bridge_detail::bytes(image),maximum_);
    }
    std::vector<ProgramSelected> select(const std::vector<std::vector<std::size_t>>& cells,std::size_t maximum) const {
        using namespace comparison_bridge_detail;
        const auto forms=result_.forms();const auto program=forms.program();const auto grade=forms.declaration().grade;
        const auto offsets=program.fibers().fibers().stalks().offsets(grade);numeric::ExactActionGuard guard(limits(bounds_));
        std::size_t used{};std::set<std::size_t> seen;std::vector<ProgramSelected> out;
        for (const auto& c:cells) {
            if (c.size()!=2 || c[0]!=grade.value || offsets.empty() || c[1]>=offsets.size()-1 || !seen.insert(c[1]).second)
                throw std::invalid_argument("distinct actual comparison output stalks required");
            const auto count=offsets[c[1]+1]-offsets[c[1]];
            if (used>maximum || count>maximum-used) throw std::length_error("comparison stalk export coordinate bound");
            used+=count;numeric::RationalVector values;ProgramSelected row;
            row.stalk=stalk_info(inference::field_interface_detail::stalk(program,cell(c[0],c[1])));
            for(std::size_t i=0;i<count;++i){guard.step();const auto q=result_.output().state(offsets[c[1]]+i).value();values.push_back(q);row.values.push_back(q.get_str(16));}
            row.field=tensor_bytes(inference::field_interface_detail::tensor(values,guard));out.push_back(std::move(row));
        }
        return out;
    }
    std::shared_ptr<ComparisonPairBridge> pair(std::size_t first,std::size_t second,unsigned channel) const;
private:
    friend class FieldComparisonBridge;
    ComparisonResultBridge(rfc::inference::OwnedAccessionFieldResult input,rfc::inference::RelationalComparisonResult result,std::uint64_t node,
            std::vector<std::size_t> bounds,std::size_t maximum)
        :input_(std::move(input)),result_(std::move(result)),node_(node),bounds_(std::move(bounds)),maximum_(maximum) {}
    rfc::inference::OwnedAccessionFieldResult input_;
    rfc::inference::RelationalComparisonResult result_;
    std::uint64_t node_{};
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<ComparisonResultBridge> FieldComparisonBridge::apply(std::shared_ptr<AccessionResultBridge> input,std::uint64_t node,unsigned selected) const {
    using namespace comparison_bridge_detail;
    if (!input) throw std::invalid_argument("actual source-backed comparison input owner required");
    const auto parent=input->native_owner();const auto fixed=parent.result();const auto actual=fixed.program();const auto program=owner_->program();
    if (!actual.fibers().same_owner(program.fibers())) throw std::invalid_argument("comparison input uses another actual program owner");
    node_grade(program,node,owner_->declaration().grade);
    auto result=owner_->apply(channel(selected),fixed.field(node),limits(bounds_));
    return std::shared_ptr<ComparisonResultBridge>(new ComparisonResultBridge(parent,std::move(result),node,bounds_,maximum_));
}
class ComparisonPairBridge {
public:
    ComparisonPairInfo info() const {
        const auto c=owner_.lower();return {owner_.first(),owner_.second(),c.grade.value,c.index,static_cast<unsigned>(owner_.channel()),owner_.admitted(),
            program_bridge_detail::tensor_bytes(owner_.residual()),program_bridge_detail::matrix_bytes(owner_.metric()),owner_.weight().get_str(16),owner_.quadrance().get_str(16)};
    }
    std::shared_ptr<ComparisonResultBridge> parent() const {return parent_;}
private:
    friend class ComparisonResultBridge;
    ComparisonPairBridge(rfc::inference::RelationalComparisonPairReading owner,std::shared_ptr<ComparisonResultBridge> parent)
        :owner_(std::move(owner)),parent_(std::move(parent)) {}
    rfc::inference::RelationalComparisonPairReading owner_;
    std::shared_ptr<ComparisonResultBridge> parent_;
};
inline std::shared_ptr<ComparisonPairBridge> ComparisonResultBridge::pair(std::size_t first,std::size_t second,unsigned channel) const {
    auto reading=result_.pair_residual(first,second,comparison_bridge_detail::channel(channel),program_bridge_detail::limits(bounds_));
    auto parent=std::shared_ptr<ComparisonResultBridge>(new ComparisonResultBridge(*this));
    return std::shared_ptr<ComparisonPairBridge>(new ComparisonPairBridge(std::move(reading),std::move(parent)));
}
} // namespace rexgraph::python
