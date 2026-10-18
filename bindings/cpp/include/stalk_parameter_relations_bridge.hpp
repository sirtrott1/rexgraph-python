#pragma once
#include "stalk_transport_bridge.hpp"
#include <rfc/learning/transport_parameter_relations.hpp>

namespace rexgraph::python {
struct ParameterRelationTerm {std::size_t entry{};std::string coefficient;};
struct ParameterRelation {std::string name;std::vector<ParameterRelationTerm> terms;std::string expected;};
struct ParameterRelationsReceipt {
    std::vector<std::string> residual_origin;
    std::vector<BoundaryMapEntry> residual_directions;
    std::vector<std::size_t> statuses;
    std::size_t steps{},retained{};
};
struct ParameterConditionReceipt {
    bool consistent{};
    std::vector<std::string> particular,coordinate_particular;
    std::vector<BoundaryMapEntry> kernel,coordinate_kernel;
    std::vector<std::size_t> free_columns;
    std::size_t rank{},steps{};
};
struct FieldConditionReceipt {
    ParameterConditionReceipt family;
    std::vector<std::vector<std::size_t>> constraints;
    std::vector<std::string> contradiction;
    std::string contradiction_value;
    std::size_t base_rank{},additional_rank{},status{};
    bool base_consistent{};
};
class StalkParameterRelations {
    using Guard=rfc::numeric::ExactActionGuard;
    using M=rfc::numeric::ExactMatrix;
    std::shared_ptr<const StalkTransportFit> transport_;
    std::shared_ptr<const rfc::learning::PreparedTransportParameterRelations> relations_;
    ParameterRelationsReceipt receipt_;
    static std::vector<BoundaryMapEntry> entries(const M& map) {
        std::vector<BoundaryMapEntry> out;
        for(const auto& [key,q]:map.entries()) out.push_back({key.first,key.second,q.get_str(16)});
        return out;
    }
    void record_receipt(std::size_t steps) {
        const auto& owner=*relations_;
        if(!owner.observations().owner().same_owner(transport_->observations_->owner()) ||
                owner.evidence_origin()!=transport_->particular_ || owner.evidence_kernel()!=transport_->kernel_ ||
                owner.free_columns()!=transport_->receipt_.free_columns || owner.rank()!=transport_->receipt_.rank)
            throw std::logic_error("parameter relations differ from complete native observation replay");
        receipt_.residual_origin=StalkTransportFit::strings(owner.residual_origin());
        receipt_.residual_directions=entries(owner.residual_directions());
        for(const auto status:owner.statuses()) receipt_.statuses.push_back(static_cast<std::size_t>(status));
        receipt_.steps=steps;receipt_.retained=owner.retained_entries();
    }
public:
    StalkParameterRelations(std::shared_ptr<StalkTransportFit> transport,const std::vector<ParameterRelation>& relations,
        bool allow_constants=false)
        :transport_(std::move(transport)) {
        using namespace rfc;
        if(!transport_) throw std::invalid_argument("owning original transport required");
        Guard guard(transport_->limits_);guard.dimension(relations.size());
        std::vector<learning::TransportParameterRelation> maps;std::size_t declarations{};
        for(const auto& relation:relations) {
            guard.step();learning::TransportParameterRelation native;
            native.name=relation.name;native.expected=StalkTransportFit::scalar(relation.expected,guard);
            for(const auto& term:relation.terms) {
                declarations=machine::checked_size_add(declarations,2);guard.entries(declarations);
                native.terms.push_back({term.entry,StalkTransportFit::scalar(term.coefficient,guard)});
            }
            maps.push_back(std::move(native));
        }
        relations_=std::make_shared<const learning::PreparedTransportParameterRelations>(*transport_->observations_,
            transport_->evidence_->observations(),std::move(maps),guard,allow_constants?
                learning::TransportParameterSupportPolicy::IncludeConstants:learning::TransportParameterSupportPolicy::VariableEntries);
        record_receipt(guard.steps());
    }
    StalkParameterRelations(std::shared_ptr<StalkTransportFit> transport,const ContextViewRules& acquired)
        :transport_(std::move(transport)),relations_(acquired.proof) {
        if(!transport_ || !relations_) throw std::invalid_argument("owning native context proof required");
        record_receipt(0);
    }
    ParameterRelationsReceipt receipt() const {return receipt_;}
    std::vector<BoundaryMapEntry> sparse_residual() const {
        Guard guard(transport_->limits_);std::vector<BoundaryMapEntry> out;
        const auto append=[&](std::size_t row,std::size_t column,const rfc::numeric::Rational& q) {
            guard.step();guard.entries(rfc::machine::checked_size_add(relations_->retained_entries(),
                rfc::machine::checked_size_mul(rfc::machine::checked_size_add(out.size(),1),3)));
            out.push_back({row,column,q.get_str(16)});
        };
        const auto& origin=relations_->residual_origin();
        for(std::size_t i=0;i<origin.size();++i) {guard.step();if(origin[i]!=0) append(i,0,origin[i]);}
        for(const auto& [key,q]:relations_->residual_directions().entries()) append(key.first,key.second+1,q);
        std::sort(out.begin(),out.end(),[&](const auto& a,const auto& b){guard.step();return std::tie(a.row,a.column)<std::tie(b.row,b.column);});
        return out;
    }
    ParameterConditionReceipt condition(const std::vector<std::size_t>& selected) const {
        Guard guard(transport_->limits_);const auto result=relations_->condition(selected,guard);
        ParameterConditionReceipt out;out.consistent=result.consistent;
        out.particular=StalkTransportFit::strings(result.origin);out.kernel=entries(result.kernel);
        out.coordinate_particular=StalkTransportFit::strings(result.coordinate_origin);
        out.coordinate_kernel=entries(result.coordinate_kernel);
        for(const auto i:result.free_columns) out.free_columns.push_back(transport_->receipt_.free_columns.at(i));
        out.rank=result.rank;out.steps=guard.steps();return out;
    }
    FieldConditionReceipt restrict_fields(const std::vector<std::size_t>& selected,
        const std::vector<TransportObservation>& observations) const {
        using namespace rfc;
        using V=numeric::RationalVector;
        using Q=numeric::Rational;
        Guard guard(transport_->limits_);guard.dimension(observations.size());
        if(observations.empty()) throw std::invalid_argument("nonempty whole-field restriction required");
        std::vector<learning::TransportObservation> samples;std::size_t input_entries{};
        for(const auto& o:observations) {
            guard.step();input_entries=machine::checked_size_add(input_entries,
                machine::checked_size_add(o.input.size(),machine::checked_size_add(o.target.size(),o.observed.size())));
            guard.entries(machine::checked_size_add(relations_->retained_entries(),input_entries));
            samples.push_back({StalkTransportFit::values(o.input,guard),StalkTransportFit::values(o.target,guard),o.observed});
        }
        const auto system=transport_->observations_->assemble(samples,guard);
        const auto base=relations_->condition(selected,guard);
        FieldConditionReceipt out;out.base_consistent=base.consistent;out.base_rank=base.rank;
        for(const auto& c:system.constraints) {guard.step();out.constraints.push_back({c.sample,c.cell,c.axis});}
        out.family.rank=base.rank;
        if(!base.consistent) {out.status=3;out.family.steps=guard.steps();return out;}
        const auto live=machine::checked_size_add(relations_->retained_entries(),machine::checked_size_add(input_entries,
            machine::checked_size_add(system.design.nonzeros(),machine::checked_size_add(system.augmented.nonzeros(),
            machine::checked_size_add(machine::checked_size_mul(system.target.size(),5),
            machine::checked_size_add(machine::checked_size_mul(system.constraints.size(),3),
            machine::checked_size_add(base.origin.size(),machine::checked_size_add(base.coordinate_origin.size(),
            machine::checked_size_add(base.kernel.nonzeros(),base.coordinate_kernel.nonzeros())))))))));
        const auto workspace=[&](std::size_t scratch){guard.entries(machine::checked_size_add(live,scratch));};
        workspace(0);
        const auto map=numeric::checked_multiply(system.design,base.kernel,guard);
        auto rhs=numeric::checked_apply(system.design,base.origin,guard);
        for(std::size_t i=0;i<rhs.size();++i) {guard.step();rhs[i]=guard.sub(system.target[i],rhs[i]);}
        const auto d=map.cols();guard.dimension(machine::checked_size_add(d,1));
        M augmented(map.rows(),machine::checked_size_add(d,1));
        for(const auto& [key,q]:map.entries()) {guard.step();augmented.set(key.first,key.second,q);}
        for(std::size_t i=0;i<rhs.size();++i) {guard.step();augmented.set(i,d,rhs[i]);}
        const auto maps=machine::checked_size_add(map.nonzeros(),augmented.nonzeros());workspace(maps);
        const numeric::PreparedExactFrames frames(map,guard),target(augmented,guard);
        const auto frames_live=machine::checked_size_add(maps,machine::checked_size_add(frames.stored_entries(),target.stored_entries()));
        workspace(frames_live);out.additional_rank=frames.rank();out.family.rank+=frames.rank();
        out.family.consistent=frames.rank()==target.rank();
        if(!out.family.consistent) {
            out.status=1;M dual(map.cols(),map.rows());
            for(const auto& [key,q]:map.entries()) {guard.step();dual.set(key.second,key.first,q);}
            const numeric::PreparedExactFrames left(dual,guard);
            workspace(machine::checked_size_add(frames_live,machine::checked_size_add(dual.nonzeros(),left.stored_entries())));
            for(std::size_t j=0;j<left.kernel().cols();++j) {
                V witness(rhs.size());Q value{};
                for(std::size_t i=0;i<rhs.size();++i) {
                    guard.step();witness[i]=left.kernel().get(i,j);value=guard.add(value,guard.mul(witness[i],rhs[i]));
                }
                if(value==0) continue;
                const auto certificate=numeric::checked_transpose_apply(map,witness,guard);
                if(std::any_of(certificate.begin(),certificate.end(),[](const auto& q){return q!=0;}))
                    throw std::logic_error("whole-field contradiction has nonzero dual residual");
                out.contradiction=StalkTransportFit::strings(witness);out.contradiction_value=value.get_str(16);break;
            }
            if(out.contradiction.empty()) throw std::logic_error("whole-field contradiction certificate missing");
        } else {
            out.status=frames.rank() ? 2 : 0;V coordinate(d);
            for(std::size_t i=0;i<target.rank();++i) {guard.step();coordinate[target.pivots()[i]]=target.reduced().get(i,d);}
            auto point=numeric::checked_apply(base.kernel,coordinate,guard);
            for(std::size_t i=0;i<point.size();++i) {guard.step();point[i]=guard.add(point[i],base.origin[i]);}
            const auto kernel=numeric::checked_multiply(base.kernel,frames.kernel(),guard);
            auto original_coordinate=numeric::checked_apply(base.coordinate_kernel,coordinate,guard);
            for(std::size_t i=0;i<original_coordinate.size();++i) {
                guard.step();original_coordinate[i]=guard.add(original_coordinate[i],base.coordinate_origin[i]);
            }
            const auto coordinate_kernel=numeric::checked_multiply(base.coordinate_kernel,frames.kernel(),guard);
            workspace(machine::checked_size_add(frames_live,machine::checked_size_add(point.size(),
                machine::checked_size_add(original_coordinate.size(),machine::checked_size_add(kernel.nonzeros(),coordinate_kernel.nonzeros())))));
            if(numeric::checked_apply(system.design,point,guard)!=system.target ||
                    numeric::checked_multiply(system.design,kernel,guard).nonzeros())
                throw std::logic_error("whole-field conditional family certificate failed");
            out.family.particular=StalkTransportFit::strings(point);out.family.kernel=entries(kernel);
            out.family.coordinate_particular=StalkTransportFit::strings(original_coordinate);
            out.family.coordinate_kernel=entries(coordinate_kernel);
            for(const auto i:frames.free_columns()) {
                guard.step();out.family.free_columns.push_back(transport_->receipt_.free_columns.at(base.free_columns.at(i)));
            }
        }
        out.family.steps=guard.steps();return out;
    }
    TransportResponse apply(const std::vector<std::size_t>& selected,const std::vector<std::string>& field) const {
        Guard guard(transport_->limits_);const auto family=relations_->condition(selected,guard);
        if(!family.consistent) throw std::invalid_argument("inconsistent conditional transport family");
        std::vector<std::size_t> directions;
        for(const auto i:family.free_columns) directions.push_back(transport_->receipt_.free_columns.at(i));
        auto response=transport_->execute_family(field,family.origin,family.kernel,directions);
        response.steps=rfc::machine::checked_size_add(response.steps,guard.steps());
        if(response.steps>transport_->limits_.steps)
            throw std::length_error("complete conditional field action exceeds work admission");
        return response;
    }
};
} // namespace rexgraph::python
