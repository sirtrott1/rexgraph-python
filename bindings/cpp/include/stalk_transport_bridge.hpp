#pragma once
#include <rfc/learning/context_transport_rules.hpp>
#include <rfc/learning/context_transport_view_rules.hpp>
#include "stalk_boundary_bridge.hpp"
#include <rfc/learning/transport_observations.hpp>
#include <rfc/learning/transport_evidence.hpp>
#include <rfc/learning/transport_parameter_basis.hpp>
#include <rfc/learning/context_transport_hypotheses.hpp>
#include <rfc/learning/context_transport_views.hpp>
#include <rfc/learning/context_transport_diagrams.hpp>
#include <rfc/numeric/prepared_exact_frames.hpp>
#include <tuple>

namespace rexgraph::python {
struct TransportSlot { std::size_t occurrence{}, row{}, column{}; };
struct TransportObservation {
    std::vector<std::string> input, target;
    std::vector<std::size_t> observed;
};
struct EncodedColumnField {std::size_t cell{};std::string tensor;};
struct ColumnObservation {std::vector<EncodedColumnField> input,target;};
struct ColumnTensorEntry {std::size_t axis{};std::string value;};
inline std::string encode_column_tensor(std::size_t dimension,const std::vector<ColumnTensorEntry>& entries,
                                      const std::vector<std::size_t>& bounds) {
    using namespace rfc;numeric::ExactActionGuard guard({bounds.at(0),bounds.at(1),bounds.at(2),bounds.at(3)});
    guard.dimension(dimension);guard.entries(machine::checked_size_mul(entries.size(),2));
    std::vector<numeric::SparseRationalEntry> values;std::optional<std::size_t> previous;
    for(const auto& e:entries) {
        guard.step();numeric::Rational q;
        if(e.axis>=dimension || (previous && e.axis<=*previous) || q.set_str(e.value,16)!=0 || q.get_den()==0)
            throw std::invalid_argument("ordered original nonzero tensor coordinates required");
        q.canonicalize();guard.value(q);if(q==0)throw std::invalid_argument("nonzero sparse tensor values required");
        previous=e.axis;values.push_back({e.axis,std::move(q)});
    }
    storage::temporal_codec_detail::Writer writer;
    storage::temporal_codec_detail::tensor(writer,numeric::SparseRationalTensor(numeric::IndexDomain({dimension}),
        numeric::StructuralSupport::full(dimension),numeric::SparseRationalVector(dimension,std::move(values))));
    return {reinterpret_cast<const char*>(writer.bytes().data()),writer.bytes().size()};
}
struct TransportFitReceipt {
    std::vector<std::string> particular, target;
    std::vector<BoundaryMapEntry> kernel;
    std::vector<std::vector<std::size_t>> constraints;
    std::vector<BoundaryMapEntry> design;
    std::vector<std::size_t> pivots, free_columns;
    std::size_t rank{}, steps{}, retained{};
};
struct TransportResponse {
    std::vector<std::vector<std::string>> boundary, occurrences;
    std::vector<std::size_t> directions, zero_directions;
    std::size_t steps{};
};
struct ContextFrame {std::size_t frame{};std::vector<std::size_t> axes;};
struct ContextEquation {std::string name;std::vector<std::size_t> entries;std::vector<std::string> gains;std::string expected;};
struct ContextView {
    std::size_t frame{},dimension{};
    std::vector<BoundaryMapEntry> accession,reconstruction;
};
struct ContextViewWitness {
    std::size_t common_rows{},origin_rows{},target_rows{},columns{};
    std::vector<BoundaryMapEntry> common_input,origin_input,target_input;
    std::vector<std::vector<std::size_t>> response_shapes,equation_coordinates;
};
struct ContextBlock {
    std::size_t origin{},target{};
    std::vector<std::vector<std::size_t>> cells,occurrences;
    std::vector<std::size_t> equations,evidence_samples;
};
struct ContextAcquisition {
    std::vector<std::size_t> seed_rows;
    std::vector<ContextBlock> blocks;
    std::vector<ContextEquation> equations;
    std::vector<std::string> rejection_names;
    std::vector<std::size_t> rejection_counts;
    std::size_t attempts{},steps{},retained{};
};
struct ContextRule {
    std::size_t origin{};
    std::vector<std::size_t> forced,excluded,open;
    bool admitted{};
};
struct ContextRules {
    ContextAcquisition comparisons;
    std::vector<ContextRule> rules;
    std::vector<std::size_t> statuses,selected;
    std::size_t steps{},retained{};
};
struct ContextDiagram {
    std::size_t patch{};
    std::vector<std::size_t> blocks,equations;
    std::vector<std::vector<std::size_t>> rows,cells,occurrences,shared_cells;
};
struct ContextViewAcquisition {
    ContextAcquisition acquisition;std::vector<ContextViewWitness> witnesses;
    std::vector<ContextDiagram> diagrams;
    std::vector<std::string> diagram_rejection_names;
    std::vector<std::size_t> diagram_rejection_counts;
    std::size_t diagram_attempts{};
};
struct ContextViewRules {
    ContextViewAcquisition views;
    std::vector<ContextRule> rules;
    std::vector<std::size_t> statuses, ranks, selected;
    std::vector<std::vector<std::size_t>> covering_peers;
    std::vector<std::string> rejection_names;
    std::vector<std::size_t> rejection_counts;
    std::size_t steps{}, retained{};
    std::shared_ptr<const rfc::learning::PreparedTransportParameterRelations> proof;
};

class StalkTransportFit {
    friend class StalkTransportPattern;
    friend class StalkTransportPatternBank;
    friend class StalkTransportPatternTraining;
    friend class StalkFieldObjective;
    friend class StalkParameterRelations;
    using Q = rfc::numeric::Rational;
    using Vector = rfc::numeric::RationalVector;
    using Matrix = rfc::numeric::ExactMatrix;
    using Guard = rfc::numeric::ExactActionGuard;
    StalkBoundaryTower fixed_;
    std::vector<TransportSlot> slots_;
    std::vector<std::size_t> lower_, upper_, occurrences_;
    Vector particular_;
    std::shared_ptr<const rfc::learning::PreparedTransportObservations> observations_;
    std::shared_ptr<const rfc::learning::PreparedTransportEvidence> evidence_;
    Matrix kernel_, basis_;
    bool independent_{};
    std::size_t template_rank_{};
    rfc::numeric::ExactActionLimits limits_;
    TransportFitReceipt receipt_;
    std::size_t preparation_base_{}, parsed_steps_{}, basis_workspace_{}, guard_steps_{};
    bool source_reused_{};

    static Q scalar(const std::string& text, Guard& guard) {
        Q q;
        if (q.set_str(text, 16) != 0 || q.get_den() == 0)
            throw std::invalid_argument("invalid exact transport coefficient");
        q.canonicalize(); guard.value(q); return q;
    }
    static std::vector<std::string> strings(const Vector& values) {
        std::vector<std::string> out; out.reserve(values.size());
        for (const auto& q : values) out.push_back(q.get_str(16));
        return out;
    }
    static Vector values(const std::vector<std::string>& text, Guard& guard) {
        guard.dimension(text.size()); Vector out; out.reserve(text.size());
        for (const auto& q : text) { guard.step(); out.push_back(scalar(q, guard)); }
        return out;
    }
    static void charge(std::size_t steps, Guard& guard) {
        for(std::size_t i=0;i<steps;++i) guard.step();
    }
    Matrix observation_map(const Vector& input, bool shared, Guard& guard) const {
        return observations_->observation_map(input,shared,guard);
    }
public:
    StalkTransportFit(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& fixed_maps,
        const std::vector<TransportSlot>& slots,
        const std::vector<TransportObservation>& observations,
        const std::vector<std::size_t>& bounds, std::size_t max_bytes)
        : StalkTransportFit(archive, axes, fixed_maps, slots, observations, bounds, max_bytes, slots.size(), {}, true) {}
    StalkTransportFit(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& fixed_maps,
        const std::vector<TransportSlot>& slots,
        const std::vector<TransportObservation>& observations,
        const std::vector<std::size_t>& bounds, std::size_t max_bytes,
        std::size_t parameter_count, const std::vector<BoundaryMapEntry>& basis)
        : StalkTransportFit(archive, axes, fixed_maps, slots, observations, bounds, max_bytes, parameter_count, basis, false) {}
private:
    StalkTransportFit(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& fixed_maps,
        const std::vector<TransportSlot>& slots,
        const std::vector<TransportObservation>& observations,
        const std::vector<std::size_t>& bounds, std::size_t max_bytes,
        std::size_t parameter_count, const std::vector<BoundaryMapEntry>& basis, bool independent)
        : fixed_(archive, axes, fixed_maps, bounds, max_bytes), slots_(slots),
          basis_(slots.size(), parameter_count), independent_(independent),
          limits_{bounds.at(0), bounds.at(1), bounds.at(2), bounds.at(3)} {
        using namespace rfc;
        Guard guard(limits_); charge(fixed_.preparation_steps(),guard); const auto offsets = fixed_.offsets();
        if (axes.size() < 2 || std::any_of(axes.begin()+2, axes.end(), [](const auto& a){return !a.empty();}))
            throw std::invalid_argument("transport identification requires a grade-one source with empty higher grades");
        guard.dimension(machine::checked_size_add(slots.size(), 1));
        guard.dimension(machine::checked_size_add(parameter_count, 1));
        guard.dimension(observations.size()); lower_=offsets[0]; upper_=offsets[1];
        const auto cells=fixed_.occurrence_cells(1);
        occurrences_.push_back(0);
        for (const auto& cell : cells)
            occurrences_.push_back(machine::checked_size_add(occurrences_.back(), axes[0][cell[1]].size()));
        guard.dimension(occurrences_.back());
        if (independent_) template_rank_ = parameter_count;
        else {
            std::set<Matrix::Key> seen;
            for (const auto& entry : basis) {
                guard.step();
                if (entry.row >= slots_.size() || entry.column >= parameter_count ||
                        !seen.emplace(entry.row, entry.column).second)
                    throw std::invalid_argument("distinct original entry and named parameter basis coordinates required");
                const auto value = scalar(entry.value, guard);
                if (value == 0) throw std::invalid_argument("sparse parameter basis entries must be nonzero");
                basis_.set(entry.row, entry.column, value); guard.entries(basis_.nonzeros());
            }
            const auto template_rank=learning::transport_parameter_basis_rank(basis_,guard);
            template_rank_ = template_rank.rank; basis_workspace_ = template_rank.workspace_entries;
        }
        std::vector<learning::TransportCoordinate> fixed_support, variables;
        for(std::size_t i=0;i<fixed_maps[1].size();++i)
            for(const auto& e:fixed_maps[1][i]) fixed_support.push_back({i,e.row,e.column});
        for(const auto& s:slots_) variables.push_back({s.occurrence,s.row,s.column});
        if(independent_) observations_=std::make_shared<const learning::PreparedTransportObservations>(
            fixed_.fiber_owner(),std::move(fixed_support),std::move(variables),guard);
        else observations_=std::make_shared<const learning::PreparedTransportObservations>(
            fixed_.fiber_owner(),std::move(fixed_support),std::move(variables),basis_,guard);
        preparation_base_=guard.steps();
        evidence_=std::make_shared<const learning::PreparedTransportEvidence>(*observations_,guard);
        if(!observations.empty()) {
            const auto native_observations=parse(observations,guard);
            evidence_=std::make_shared<const learning::PreparedTransportEvidence>(evidence_->append(native_observations,guard));
        }
        observations_=std::shared_ptr<const learning::PreparedTransportObservations>(evidence_,&evidence_->model());
        retain(guard);guard_steps_=guard.steps();
    }
    std::vector<rfc::learning::TransportObservation> parse(const std::vector<TransportObservation>& observations,Guard& guard) {
        const auto start=guard.steps();std::size_t input_entries{};
        using namespace rfc;
        std::vector<learning::TransportObservation> native_observations;
        for (const auto& o : observations) {
            guard.step();
            input_entries=machine::checked_size_add(input_entries,machine::checked_size_add(o.input.size(),o.target.size()+o.observed.size()));
            guard.entries(input_entries);
            native_observations.push_back({values(o.input,guard),values(o.target,guard),o.observed});
        }
        parsed_steps_=machine::checked_size_add(parsed_steps_,guard.steps()-start);
        return native_observations;
    }
    std::vector<rfc::learning::TransportColumnObservation> parse_columns(const std::vector<ColumnObservation>& samples,Guard& guard) {
        using namespace rfc;const auto start=guard.steps();std::size_t retained{};
        std::vector<learning::TransportColumnObservation> out;
        storage::TemporalReadLimits policy;policy.max_dimension=limits_.dimension;policy.max_records=limits_.entries;
        policy.max_integer_bytes=machine::ceil_div(limits_.integer_bits,8);policy.retain_encoded_values=true;
        for(const auto& sample:samples) {
            guard.step();learning::TransportColumnObservation o{observations_->owner(),observations_->grade(),{}, {}};
            for(const bool target:{false,true}) {
                const auto& fields=target?sample.target:sample.input;auto& output=target?o.target:o.input;
                for(const auto& field:fields) {
                    guard.step();retained=machine::checked_size_add(retained,field.tensor.size());guard.entries(retained);
                    const auto bytes=std::span<const std::byte>(reinterpret_cast<const std::byte*>(field.tensor.data()),field.tensor.size());
                    storage::temporal_codec_detail::Reader reader(bytes,policy,true);
                    auto tensor=storage::temporal_codec_detail::tensor(reader);
                    if(reader.remaining())throw std::invalid_argument("trailing encoded original stalk tensor bytes");
                    output.push_back({field.cell,std::move(tensor)});
                }
            }
            out.push_back(std::move(o));
        }
        parsed_steps_=machine::checked_size_add(parsed_steps_,guard.steps()-start);return out;
    }
    void retain(Guard& guard) {
        using namespace rfc;
        const auto start=guard.steps();const auto& system=evidence_->system();
        particular_=evidence_->particular();kernel_=evidence_->kernel();
        for(const auto& c:system.constraints) receipt_.constraints.push_back({c.sample,c.cell,c.axis});
        receipt_.particular=strings(particular_); receipt_.target=strings(system.target);
        receipt_.rank=evidence_->rank(); receipt_.pivots=evidence_->pivots(); receipt_.free_columns=evidence_->free_columns();
        for(const auto& [key,q] : kernel_.entries()) { guard.step(); receipt_.kernel.push_back({key.first,key.second,q.get_str(16)}); }
        for (const auto& [key,q] : system.design.entries()) receipt_.design.push_back({key.first,key.second,q.get_str(16)});
        receipt_.retained=machine::checked_size_add(fixed_.retained_entries(),evidence_->retained_entries());
        receipt_.retained=machine::checked_size_add(receipt_.retained,machine::checked_size_add(
            machine::checked_size_mul(slots_.size(),3),machine::checked_size_add(particular_.size(),kernel_.nonzeros())));
        if (!independent_) receipt_.retained=machine::checked_size_add(receipt_.retained,
            machine::checked_size_add(machine::checked_size_mul(basis_.cols(),3),
                machine::checked_size_add(basis_.nonzeros(),basis_workspace_)));
        guard.entries(receipt_.retained);
        receipt_.steps=machine::checked_size_add(preparation_base_,machine::checked_size_add(parsed_steps_,
            machine::checked_size_add(evidence_->preparation_steps(),guard.steps()-start)));
        if(receipt_.steps>limits_.steps)throw std::length_error("complete transport fit replay exceeds step admission");
    }
    StalkTransportFit(const StalkTransportFit& parent,const std::vector<TransportObservation>& suffix)
        :fixed_(parent.fixed_),slots_(parent.slots_),lower_(parent.lower_),upper_(parent.upper_),occurrences_(parent.occurrences_),
         basis_(parent.basis_),independent_(parent.independent_),template_rank_(parent.template_rank_),limits_(parent.limits_),
         preparation_base_(parent.preparation_base_),parsed_steps_(parent.parsed_steps_),basis_workspace_(parent.basis_workspace_),source_reused_(true) {
        Guard guard(limits_);const auto parsed=parse(suffix,guard);
        evidence_=std::make_shared<const rfc::learning::PreparedTransportEvidence>(parent.evidence_->append(parsed,guard));
        observations_=std::shared_ptr<const rfc::learning::PreparedTransportObservations>(evidence_,&evidence_->model());
        retain(guard);guard_steps_=guard.steps();
    }
    StalkTransportFit(const StalkTransportFit& parent,const std::vector<ColumnObservation>& suffix)
        :fixed_(parent.fixed_),slots_(parent.slots_),lower_(parent.lower_),upper_(parent.upper_),occurrences_(parent.occurrences_),
         observations_(parent.observations_),basis_(parent.basis_),independent_(parent.independent_),
         template_rank_(parent.template_rank_),limits_(parent.limits_),preparation_base_(parent.preparation_base_),
         parsed_steps_(parent.parsed_steps_),basis_workspace_(parent.basis_workspace_),source_reused_(true) {
        Guard guard(limits_);const auto parsed=parse_columns(suffix,guard);
        evidence_=std::make_shared<const rfc::learning::PreparedTransportEvidence>(parent.evidence_->append_columns(parsed,guard));
        observations_=std::shared_ptr<const rfc::learning::PreparedTransportObservations>(evidence_,&evidence_->model());
        retain(guard);guard_steps_=guard.steps();
    }
public:
    std::shared_ptr<StalkTransportFit> append_columns(const std::vector<ColumnObservation>& suffix) const {
        return std::shared_ptr<StalkTransportFit>(new StalkTransportFit(*this,suffix));
    }
    std::size_t column_observation_count() const noexcept {return evidence_->column_observation_count();}
    TransportFitReceipt receipt() const { return receipt_; }
    std::shared_ptr<StalkTransportFit> append(const std::vector<TransportObservation>& suffix) const {
        return std::shared_ptr<StalkTransportFit>(new StalkTransportFit(*this,suffix));
    }
    std::vector<std::size_t> update_stats() const {
        const auto update=evidence_->update();return {source_reused_?1u:0u,update.reused_observations,update.assembled_observations,guard_steps_};
    }
    std::size_t materialized_field_entries() const noexcept {return evidence_->materialized_field_entries();}
    ContextAcquisition contextual_hypotheses(const std::vector<std::size_t>& targets,
        const std::vector<std::vector<ContextFrame>>& atlas) const {
        rfc::numeric::ExactActionGuard guard(limits_);
        std::vector<std::vector<rfc::learning::TransportContextFrame>> frames;
        for(const auto& grade:atlas) {
            frames.emplace_back();
            for(const auto& chart:grade) frames.back().push_back({chart.frame,chart.axes});
        }
        const auto result=rfc::learning::acquire_context_transport_hypotheses(*observations_,evidence_->observations(guard),targets,frames,guard);
        return context_result(result,guard.steps());
    }
    ContextViewAcquisition contextual_views(const std::vector<std::size_t>& targets,
        const std::vector<std::vector<ContextView>>& atlas) const {
        return contextual(atlas,targets,{},false);
    }
    ContextViewRules contextual_view_rules(const std::vector<std::size_t>& training,
        const std::vector<std::size_t>& targets,const std::vector<std::vector<ContextView>>& atlas,
        std::size_t minimum_peers,bool observed_inputs=false,bool declared_templates=false) const {
        Guard guard(limits_);const auto views=native_views(atlas,guard);
        const auto result=rfc::learning::acquire_context_transport_view_rules(*evidence_,training,targets,views,minimum_peers,guard,
            observed_inputs?rfc::learning::TransportContextInputScope::ObservedFields:rfc::learning::TransportContextInputScope::FullCarrier,declared_templates);
        ContextViewRules out;out.views=view_result(result.comparisons,guard.steps(),nullptr);
        for(const auto& rule:result.rules) out.rules.push_back({rule.origin,rule.forced,rule.excluded,rule.open,rule.admitted});
        for(const auto status:result.block_statuses) out.statuses.push_back(static_cast<std::size_t>(status));
        out.ranks=result.block_ranks;out.selected=result.selected_blocks;out.covering_peers=result.covering_peers;
        for(const auto& [reason,count]:result.rejections) {out.rejection_names.push_back(reason);out.rejection_counts.push_back(count);}
        out.steps=guard.steps();out.retained=result.retained;out.proof=result.proof;return out;
    }
    ContextRules contextual_rules(const std::vector<std::size_t>& training,
        const std::vector<std::size_t>& targets,const std::vector<std::vector<ContextFrame>>& atlas,
        std::size_t minimum_peers) const {
        Guard guard(limits_);
        std::vector<std::vector<rfc::learning::TransportContextFrame>> frames;
        for(const auto& grade:atlas) {
            frames.emplace_back();
            for(const auto& chart:grade) frames.back().push_back({chart.frame,chart.axes});
        }
        const auto result=rfc::learning::acquire_context_transport_rules(*observations_,
            evidence_->observations(guard),training,targets,frames,minimum_peers,guard);
        ContextRules out;out.comparisons=context_result(result.comparisons,result.comparisons.steps);
        for(const auto& rule:result.rules)
            out.rules.push_back({rule.origin,rule.forced,rule.excluded,rule.open,rule.admitted});
        for(const auto status:result.block_statuses) out.statuses.push_back(static_cast<std::size_t>(status));
        out.selected=result.selected_blocks;out.steps=guard.steps();out.retained=result.retained;return out;
    }
    ContextViewAcquisition contextual_diagrams(const std::vector<std::vector<std::size_t>>& patches,
        const std::vector<std::vector<ContextView>>& atlas) const {
        return contextual(atlas,{},patches,true);
    }
private:
    std::vector<std::vector<rfc::learning::TransportContextView>> native_views(
        const std::vector<std::vector<ContextView>>& atlas,Guard& guard) const {
        const auto& fibers=observations_->owner().fibers();
        if(atlas.size()!=2) throw std::invalid_argument("complete context view atlas required");
        std::vector<std::vector<rfc::learning::TransportContextView>> views;std::size_t entries{};
        for(std::size_t k=0;k<atlas.size();++k) {
            if(atlas[k].size()!=fibers.complex().grade(rfc::core::Grade{static_cast<std::uint16_t>(k)}).cell_count())
                throw std::invalid_argument("context atlas must cover original cells");
            views.emplace_back();
            for(std::size_t i=0;i<atlas[k].size();++i) {
                guard.step();const auto& chart=atlas[k][i];
                const auto n=chart.frame?fibers.stalks().dimension({rfc::core::Grade{static_cast<std::uint16_t>(k)},static_cast<std::uint32_t>(i)}):0;
                rfc::learning::TransportContextView native{chart.frame,Matrix(chart.dimension,n),Matrix(n,chart.dimension)};
                for(const auto side:{0,1}) {
                    auto& map=side?native.reconstruction:native.accession;
                    std::set<Matrix::Key> support;
                    for(const auto& entry:side?chart.reconstruction:chart.accession) {
                        guard.step();if(!support.emplace(entry.row,entry.column).second)
                            throw std::invalid_argument("distinct context view map entries required");
                        entries=rfc::machine::checked_size_add(entries,3);guard.entries(entries);
                        map.set(entry.row,entry.column,scalar(entry.value,guard));
                    }
                }
                views.back().push_back(std::move(native));
            }
        }
        return views;
    }
    ContextViewAcquisition contextual(const std::vector<std::vector<ContextView>>& atlas,
        const std::vector<std::size_t>& targets,const std::vector<std::vector<std::size_t>>& patches,bool diagrams) const {
        Guard guard(limits_);const auto views=native_views(atlas,guard);
        rfc::learning::TransportContextViewAcquisition result;
        rfc::learning::TransportContextDiagramAcquisition groups;
        if(diagrams) {
            groups=rfc::learning::acquire_context_transport_diagrams(*observations_,evidence_->observations(guard),patches,views,guard);
            result=std::move(groups.stars);
        } else result=rfc::learning::acquire_context_transport_views(*evidence_,targets,views,guard);
        return view_result(result,guard.steps(),diagrams?&groups:nullptr);
    }
    static ContextViewAcquisition view_result(const rfc::learning::TransportContextViewAcquisition& result,
        std::size_t steps,const rfc::learning::TransportContextDiagramAcquisition* groups) {
        ContextViewAcquisition out;out.acquisition=context_result(result.acquisition,steps);
        if(groups) {
            out.acquisition.retained=groups->retained;out.diagram_attempts=groups->attempts;
            for(const auto& [reason,count]:groups->rejections) {out.diagram_rejection_names.push_back(reason);out.diagram_rejection_counts.push_back(count);}
            for(const auto& diagram:groups->diagrams) {
                ContextDiagram d;d.patch=diagram.patch;d.blocks=diagram.blocks;d.equations=diagram.equations;
                for(const auto& [a,b]:diagram.rows) d.rows.push_back({a,b});
                for(const auto& [a,b]:diagram.cells) d.cells.push_back({a,b});
                for(const auto& [a,b]:diagram.occurrences) d.occurrences.push_back({a,b});
                for(const auto& [a,b]:diagram.shared_cells) d.shared_cells.push_back({a,b});
                out.diagrams.push_back(std::move(d));
            }
        }
        for(const auto& witness:result.witnesses) {
            ContextViewWitness w;w.common_rows=witness.common_input.rows();w.origin_rows=witness.origin_input.rows();
            w.target_rows=witness.target_input.rows();w.columns=witness.common_input.cols();
            for(const auto& [key,q]:witness.common_input.entries()) w.common_input.push_back({key.first,key.second,q.get_str(16)});
            for(const auto& [key,q]:witness.origin_input.entries()) w.origin_input.push_back({key.first,key.second,q.get_str(16)});
            for(const auto& [key,q]:witness.target_input.entries()) w.target_input.push_back({key.first,key.second,q.get_str(16)});
            for(const auto& [a,b]:witness.response_shapes) w.response_shapes.push_back({a,b});
            for(const auto& c:witness.equation_coordinates) w.equation_coordinates.push_back({c[0],c[1],c[2]});
            out.witnesses.push_back(std::move(w));
        }
        return out;
    }
private:
    static ContextAcquisition context_result(const rfc::learning::TransportContextAcquisition& result,std::size_t steps) {
        ContextAcquisition out;out.seed_rows=result.seed_rows;
        out.attempts=result.attempts;out.steps=steps;out.retained=result.retained;
        for(const auto& [reason,count]:result.rejections) {out.rejection_names.push_back(reason);out.rejection_counts.push_back(count);}
        for(const auto& hypothesis:result.hypotheses) {
            ContextBlock block;block.origin=hypothesis.origin;block.target=hypothesis.target;
            block.equations=hypothesis.equations;block.evidence_samples=hypothesis.evidence_samples;
            for(const auto& [a,b]:hypothesis.cells) block.cells.push_back({a,b});
            for(const auto& [a,b]:hypothesis.occurrences) block.occurrences.push_back({a,b});
            out.blocks.push_back(std::move(block));
        }
        for(const auto& relation:result.relations) {
            ContextEquation equation;equation.name=relation.name;equation.expected=relation.expected.get_str(16);
            for(const auto& term:relation.terms) {equation.entries.push_back(term.entry);equation.gains.push_back(term.coefficient.get_str(16));}
            out.equations.push_back(std::move(equation));
        }
        return out;
    }
public:
    std::size_t template_rank() const { return template_rank_; }
    std::vector<std::vector<std::size_t>> occurrence_cells(std::size_t grade) const { return fixed_.occurrence_cells(grade); }
    std::vector<std::string> occurrence_keys(std::size_t grade) const { return fixed_.occurrence_keys(grade); }
    std::vector<std::string> occurrence_coefficients(std::size_t grade) const { return fixed_.occurrence_coefficients(grade); }
    std::vector<std::size_t> occurrence_activity(std::size_t grade) const { return fixed_.occurrence_activity(grade); }
    std::vector<std::string> parameter_values(const std::vector<std::string>& free_values) const {
        Guard guard(limits_); const auto free=values(free_values,guard);
        auto value=rfc::numeric::checked_apply(kernel_,free,guard);
        for(std::size_t i=0;i<value.size();++i) value[i]=guard.add(value[i],particular_[i]);
        return strings(value);
    }
    std::vector<std::string> entry_values(const std::vector<std::string>& free_values) const {
        if (independent_) return parameter_values(free_values);
        Guard guard(limits_); const auto parameters=values(parameter_values(free_values),guard);
        return strings(rfc::numeric::checked_apply(basis_,parameters,guard));
    }
    TransportResponse execute(const std::vector<std::string>& field) const {
        return execute_family(field,particular_,kernel_,receipt_.free_columns);
    }
private:
    TransportResponse execute_family(const std::vector<std::string>& field,const Vector& point,
        const Matrix& kernel,const std::vector<std::size_t>& free_columns) const {
        using namespace rfc;
        Guard guard(limits_); const auto input=values(field,guard); const auto base=observations_->fixed_response(input,guard);
        const auto shared=observation_map(input,true,guard), local=observation_map(input,false,guard);
        auto boundary=numeric::checked_apply(shared,point,guard), occurrences=numeric::checked_apply(local,point,guard);
        const auto& fixed_boundary=base.boundary; const auto& fixed_occurrences=base.occurrences;
        for (std::size_t i=0; i<boundary.size(); ++i) boundary[i]=guard.add(boundary[i],fixed_boundary[i]);
        for (std::size_t i=0; i<occurrences.size(); ++i) occurrences[i]=guard.add(occurrences[i],fixed_occurrences[i]);
        TransportResponse result; result.boundary.push_back(strings(boundary)); result.occurrences.push_back(strings(occurrences));
        const auto b=numeric::checked_multiply(shared,kernel,guard), o=numeric::checked_multiply(local,kernel,guard);
        std::set<std::size_t> retained;
        for(const auto& [key,q] : b.entries()) { (void)q; guard.step(); retained.insert(key.second); }
        for(const auto& [key,q] : o.entries()) { (void)q; guard.step(); retained.insert(key.second); }
        guard.entries(machine::checked_size_mul(retained.size()+1,machine::checked_size_add(lower_.back(),occurrences_.back())));
        for (std::size_t j=0; j<kernel.cols(); ++j) {
            guard.step();
            if(!retained.contains(j)) { result.zero_directions.push_back(free_columns[j]); continue; }
            result.directions.push_back(free_columns[j]);
            Vector bv(b.rows()), ov(o.rows());
            for (std::size_t i=0; i<bv.size(); ++i) { guard.step(); bv[i]=b.get(i,j); }
            for (std::size_t i=0; i<ov.size(); ++i) { guard.step(); ov[i]=o.get(i,j); }
            result.boundary.push_back(strings(bv)); result.occurrences.push_back(strings(ov));
        }
        result.steps=guard.steps(); return result;
    }
};
} // namespace rexgraph::python
