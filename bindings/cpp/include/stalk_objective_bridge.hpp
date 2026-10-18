#pragma once
#include "stalk_transport_bridge.hpp"
#include <rfc/learning/transport_field_objective.hpp>

namespace rexgraph::python {
struct ObjectiveViewTerm {std::size_t occurrence{},axis{};std::string coefficient;};
struct ObjectiveView {std::string name;std::vector<ObjectiveViewTerm> terms;};
struct ObjectiveField {std::vector<std::string> origin;std::vector<BoundaryMapEntry> directions;};
struct ObjectiveReceipt {
    std::vector<std::string> particular,coordinate_particular,target;
    std::vector<BoundaryMapEntry> kernel,coordinate_kernel,hessian;
    std::vector<std::size_t> free_columns;
    std::vector<ObjectiveField> fields;
    std::string initial_energy,minimum_energy;
    std::size_t rank{},steps{},retained{};
};
class StalkFieldObjective {
    using Guard=rfc::numeric::ExactActionGuard;
    using V=rfc::numeric::RationalVector;
    using M=rfc::numeric::ExactMatrix;
    std::shared_ptr<const StalkTransportFit> transport_;
    std::unique_ptr<const rfc::learning::PreparedTransportFieldObjective> objective_;
    ObjectiveReceipt receipt_;
    std::vector<std::size_t> free_columns_;
    static std::vector<BoundaryMapEntry> entries(const M& map) {
        std::vector<BoundaryMapEntry> out;
        for(const auto& [key,q]:map.entries()) out.push_back({key.first,key.second,q.get_str(16)});
        return out;
    }
public:
    StalkFieldObjective(std::shared_ptr<StalkTransportFit> transport,
        const std::vector<std::vector<std::string>>& probes,const std::vector<ObjectiveView>& views,
        const std::vector<std::string>& weights,const std::vector<std::string>& coefficients,
        const std::string& mass,bool count,bool trace_normalized):transport_(std::move(transport)) {
        using namespace rfc;
        if(!transport_ || coefficients.size()!=4) throw std::invalid_argument("owning transport and four channel coefficients required");
        Guard guard(transport_->limits_);std::vector<V> inputs;std::vector<learning::TransportFieldView> maps;
        guard.dimension(probes.size());guard.dimension(views.size());std::size_t input_entries{};
        for(const auto& probe:probes) {
            input_entries=machine::checked_size_add(input_entries,probe.size());guard.entries(input_entries);
            inputs.push_back(StalkTransportFit::values(probe,guard));
        }
        for(const auto& view:views) {
            learning::TransportFieldView native;native.name=view.name;
            for(const auto& term:view.terms) {
                input_entries=machine::checked_size_add(input_entries,3);guard.entries(input_entries);
                native.terms.push_back({term.occurrence,term.axis,StalkTransportFit::scalar(term.coefficient,guard)});
            }
            maps.push_back(std::move(native));
        }
        learning::TransportObjectiveSpecification spec;spec.mass=StalkTransportFit::scalar(mass,guard);
        for(std::size_t i=0;i<4;++i) spec.channels[i]=StalkTransportFit::scalar(coefficients[i],guard);
        spec.trace_normalized=trace_normalized;
        objective_=std::make_unique<const learning::PreparedTransportFieldObjective>(*transport_->observations_,
            transport_->particular_,transport_->kernel_,std::move(inputs),std::move(maps),
            StalkTransportFit::values(weights,guard),count ? field::CoparticipationReading::Count : field::CoparticipationReading::Share,spec,guard);
        const auto& o=*objective_;
        for(const auto i:o.free_columns()) free_columns_.push_back(transport_->receipt_.free_columns.at(i));
        receipt_.particular=StalkTransportFit::strings(o.origin());receipt_.kernel=entries(o.kernel());
        receipt_.coordinate_particular=StalkTransportFit::strings(o.coordinate_origin());
        receipt_.coordinate_kernel=entries(o.coordinate_kernel());receipt_.hessian=entries(o.hessian());
        receipt_.target=StalkTransportFit::strings(o.target());receipt_.free_columns=free_columns_;
        for(const auto& f:o.fields()) receipt_.fields.push_back({StalkTransportFit::strings(f.origin),entries(f.directions)});
        receipt_.initial_energy=o.initial_energy().get_str(16);receipt_.minimum_energy=o.minimum_energy().get_str(16);
        receipt_.rank=o.rank();receipt_.steps=guard.steps();receipt_.retained=o.retained_entries();
    }
    ObjectiveReceipt receipt() const {return receipt_;}
    TransportResponse execute(const std::vector<std::string>& field) const {
        return transport_->execute_family(field,objective_->origin(),objective_->kernel(),free_columns_);
    }
};
} // namespace rexgraph::python
