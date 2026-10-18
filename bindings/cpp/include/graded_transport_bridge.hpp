#pragma once
#include "stalk_boundary_bridge.hpp"
#include <rfc/learning/graded_transport_fit.hpp>

namespace rexgraph::python {
struct GradedTransportObservation {
    std::vector<std::string> input,target;
    std::vector<std::size_t> observed;
};
struct GradedTransportReceipt {
    std::vector<BoundaryMapEntry> chain,design,observation_design,kernel;
    std::vector<std::vector<std::size_t>> chain_coordinates,observation_coordinates;
    std::vector<std::string> particular,target,observation_target;
    std::vector<std::size_t> pivots,free_columns;
    std::size_t chain_rows{},design_rows{},observation_rows{},parameters{},rank{},chain_rank{},observation_rank{},steps{},entries{};
};
struct GradedTransportResponse {
    std::vector<std::string> boundary,occurrences;
    std::vector<BoundaryMapEntry> boundary_directions,occurrence_directions;
    std::size_t free_count{},steps{};
};
struct GradedTransportMaterialization {
    std::vector<std::string> parameters,entries,local_values;
    std::size_t steps{};
};
class GradedTransportFitBridge {
    using Q=rfc::numeric::Rational;
    using V=rfc::numeric::RationalVector;
    using M=rfc::numeric::ExactMatrix;
    using Guard=rfc::numeric::ExactActionGuard;
    rfc::numeric::ExactActionLimits limits_;
    std::size_t guard_steps_{};
    rfc::learning::PreparedGradedTransportFit fit_;
    static Q scalar(const std::string& text,Guard& guard) {
        Q q;if(q.set_str(text,16)!=0 || q.get_den()==0)throw std::invalid_argument("invalid exact graded transport value");
        q.canonicalize();guard.value(q);return q;
    }
    static V values(const std::vector<std::string>& text,Guard& guard) {
        guard.dimension(text.size());V out;for(const auto& q:text){guard.step();out.push_back(scalar(q,guard));}return out;
    }
    static std::vector<std::string> strings(const V& values) {
        std::vector<std::string> out;for(const auto& q:values)out.push_back(q.get_str(16));return out;
    }
    static std::vector<BoundaryMapEntry> entries(const M& matrix) {
        std::vector<BoundaryMapEntry> out;for(const auto& [key,q]:matrix.entries())out.push_back({key.first,key.second,q.get_str(16)});return out;
    }
    static rfc::numeric::ExactActionLimits limits(const std::vector<std::size_t>& bounds) {
        if(bounds.size()!=4 || std::any_of(bounds.begin(),bounds.end(),[](auto n){return !n;}))
            throw std::invalid_argument("four positive graded transport bounds required");
        return {bounds[0],bounds[1],bounds[2],bounds[3]};
    }
    static rfc::learning::PreparedGradedTransportFit prepare(const StalkTransportDeclaration& source,std::size_t k,
        const std::vector<std::vector<std::size_t>>& slots,const std::vector<GradedTransportObservation>& samples,
        bool named,std::size_t parameters,const std::vector<BoundaryMapEntry>& basis,
        rfc::numeric::ExactActionLimits limits,std::size_t& steps) {
        using namespace rfc;Guard guard(limits);auto owner=source.fiber_owner();const auto& fibers=owner.fibers();
        if(!k || k>=fibers.complex().grade_count())throw std::invalid_argument("positive original learned transport grade required");
        const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
        std::vector<learning::TransportCoordinate> fixed,variables;
        guard.dimension(slots.size());guard.dimension(parameters);
        for(const auto& slot:slots) {
            guard.step();if(slot.size()!=3)throw std::invalid_argument("original occurrence, row and column required");
            variables.push_back({slot[0],slot[1],slot[2]});
        }
        for(std::size_t p=0;p<fibers.complex().grade(grade).incidence_count();++p) {
            guard.step();for(const auto& [key,q]:fibers.transport_override(grade,p)->entries()) {
                (void)q;fixed.push_back({p,key.first,key.second});
            }
        }
        std::optional<M> map;
        if(named) {
            map.emplace(slots.size(),parameters);std::set<std::pair<std::size_t,std::size_t>> support;
            for(const auto& entry:basis) {
                guard.step();if(entry.row>=slots.size() || entry.column>=parameters || !support.emplace(entry.row,entry.column).second)
                    throw std::invalid_argument("distinct supported graded parameter basis coordinates required");
                map->set(entry.row,entry.column,scalar(entry.value,guard));
            }
        } else if(parameters!=slots.size() || !basis.empty())throw std::invalid_argument("independent original transport coordinates required");
        auto model=learning::PreparedTransportObservations::at_grade(owner,grade,std::move(fixed),std::move(variables),guard,std::move(map));
        auto observations=parse(samples,guard);
        learning::PreparedGradedTransportFit result(std::move(model),std::move(observations),guard);
        steps=guard.steps();return result;
    }
    static std::vector<rfc::learning::TransportObservation> parse(const std::vector<GradedTransportObservation>& samples,Guard& guard) {
        using namespace rfc;
        std::vector<learning::TransportObservation> observations;guard.dimension(samples.size());std::size_t live{};
        for(const auto& sample:samples) {
            guard.step();live=machine::checked_size_add(live,machine::checked_size_add(sample.input.size(),
                machine::checked_size_add(sample.target.size(),sample.observed.size())));guard.entries(live);
            observations.push_back({values(sample.input,guard),values(sample.target,guard),sample.observed});
        }
        return observations;
    }
    static rfc::learning::PreparedGradedTransportFit acquire(const GradedTransportFitBridge& parent,
        const std::vector<GradedTransportObservation>& samples,std::size_t& steps) {
        Guard guard(parent.limits_);const auto suffix=parse(samples,guard);auto result=parent.fit_.append(suffix,guard);
        steps=guard.steps();return result;
    }
    GradedTransportFitBridge(const GradedTransportFitBridge& parent,const std::vector<GradedTransportObservation>& samples)
        :limits_(parent.limits_),fit_(acquire(parent,samples,guard_steps_)) {}
public:
    GradedTransportFitBridge(const StalkTransportDeclaration& source,std::size_t grade,
        const std::vector<std::vector<std::size_t>>& slots,const std::vector<GradedTransportObservation>& samples,
        bool named,std::size_t parameters,const std::vector<BoundaryMapEntry>& basis,const std::vector<std::size_t>& bounds)
        :limits_(limits(bounds)),fit_(prepare(source,grade,slots,samples,named,parameters,basis,limits_,guard_steps_)) {}
    std::shared_ptr<GradedTransportFitBridge> append(const std::vector<GradedTransportObservation>& suffix) const {
        return std::shared_ptr<GradedTransportFitBridge>(new GradedTransportFitBridge(*this,suffix));
    }
    std::vector<std::size_t> update_stats() const {
        const auto update=fit_.update();return {update.chain_reused?1u:0u,update.reused_observations,update.assembled_observations,guard_steps_};
    }
    GradedTransportReceipt receipt() const {
        GradedTransportReceipt out;
        out.chain=entries(fit_.chain());out.design=entries(fit_.design());out.kernel=entries(fit_.kernel());
        out.observation_design=entries(fit_.evidence().design);out.observation_target=strings(fit_.evidence().target);
        out.particular=strings(fit_.particular());out.target=strings(fit_.target());out.pivots=fit_.pivots();out.free_columns=fit_.free_columns();
        for(const auto& c:fit_.chain_coordinates())out.chain_coordinates.push_back({c.grade,c.row,c.column});
        for(const auto& c:fit_.evidence().constraints)out.observation_coordinates.push_back({c.sample,c.cell,c.axis});
        out.chain_rows=fit_.chain().rows();out.design_rows=fit_.design().rows();out.observation_rows=fit_.evidence().design.rows();
        out.parameters=fit_.model().parameter_count();out.rank=fit_.rank();out.chain_rank=fit_.chain_rank();out.observation_rank=fit_.observation_rank();
        out.steps=fit_.preparation_steps();out.entries=fit_.retained_entries();return out;
    }
    GradedTransportResponse execute(const std::vector<std::string>& input) const {
        using namespace rfc;Guard guard(limits_);const auto field=values(input,guard);const auto& model=fit_.model();
        const auto global=model.observation_map(field,true,guard),local=model.observation_map(field,false,guard);
        const auto fixed=model.fixed_response(field,guard);
        auto boundary=numeric::checked_apply(global,fit_.particular(),guard),occurrences=numeric::checked_apply(local,fit_.particular(),guard);
        for(std::size_t i=0;i<boundary.size();++i)boundary[i]=guard.add(boundary[i],fixed.boundary[i]);
        for(std::size_t i=0;i<occurrences.size();++i)occurrences[i]=guard.add(occurrences[i],fixed.occurrences[i]);
        const auto b=numeric::checked_multiply(global,fit_.kernel(),guard),o=numeric::checked_multiply(local,fit_.kernel(),guard);
        guard.entries(machine::checked_size_add(fit_.retained_entries(),machine::checked_size_add(global.nonzeros(),
            machine::checked_size_add(local.nonzeros(),machine::checked_size_add(b.nonzeros(),machine::checked_size_add(o.nonzeros(),
            machine::checked_size_add(boundary.size(),occurrences.size())))))));
        return {strings(boundary),strings(occurrences),entries(b),entries(o),fit_.kernel().cols(),guard.steps()};
    }
    GradedTransportMaterialization materialize(const std::vector<std::string>& free,const std::vector<std::string>& origins) const {
        Guard guard(limits_);const auto v=values(free,guard);const auto parameters=fit_.parameters(v,guard);
        const auto entries=fit_.entries(parameters,guard);auto local=values(origins,guard);
        if(local.size()!=entries.size())throw std::invalid_argument("complete original local fixed values required");
        const auto& model=fit_.model();const auto& fibers=model.owner().fibers();
        for(std::size_t i=0;i<entries.size();++i) {
            guard.step();const auto& s=model.variable_coordinates()[i];
            const auto c=rfc::algebra::owned_gluing_detail::coefficient(fibers.complex(),model.grade(),s.occurrence,guard);
            if(guard.mul(c,local[i])!=fibers.transport_override(model.grade(),s.occurrence)->get(s.row,s.column))
                throw std::invalid_argument("original local fixed value differs from its realized map");
            local[i]=guard.add(local[i],entries[i]);
        }
        (void)fit_.materialize(v,guard);
        return {strings(parameters),strings(entries),strings(local),guard.steps()};
    }
};
} // namespace rexgraph::python
