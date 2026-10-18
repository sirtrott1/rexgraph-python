#pragma once
#include "stalk_pattern_bridge.hpp"
#include <rfc/learning/context_transport_pattern_bank_codec.hpp>
#include <rfc/learning/context_transport_pattern_scope.hpp>

namespace rexgraph::python {
struct PatternBankReceipt {
    PatternReceipt common;
    std::vector<std::vector<BoundaryMapEntry>> images;
    std::vector<std::size_t> rows, columns;
    std::vector<std::string> pattern_keys;
    bool declared_templates{};
};
struct PatternBankInput {std::size_t pattern{};std::vector<std::string> value;};
struct PatternSparseEntry {std::size_t index{};std::string value;};
struct PatternBankSparseInput {
    std::size_t pattern{},frame{},size{};
    std::vector<PatternSparseEntry> entries;
};
struct PatternBankComparison {
    std::string name;
    std::size_t left{},right{},left_frame{},right_frame{},interface_frame{},rows{},left_columns{},right_columns{};
    std::vector<BoundaryMapEntry> left_map,right_map;
};
struct PatternBankResponse {
    PatternResponse field;
    std::vector<std::vector<std::string>> coordinates;
};
struct PatternBankCandidate {
    std::size_t pattern{};
    bool supported{};
    PatternBankResponse response;
};
struct PatternProgramEdge {
    std::size_t node{},occurrence{},source_frame{},target_frame{},rows{},columns{};
    std::vector<BoundaryMapEntry> map;
};
struct PatternProgramNode {
    std::string name;
    std::size_t pattern{};
    bool has_input{};
    PatternBankSparseInput input;
    std::vector<PatternProgramEdge> edges;
};
struct PatternProgramResult {
    std::string status;
    bool has_input{},has_response{};
    std::size_t size{},parameters{},response_size{};
    std::vector<PatternSparseEntry> input;
    std::vector<BoundaryMapEntry> input_directions,parameter_directions;
    PatternBankResponse response;
};
struct PatternBankGluing {
    bool consistent{},forced{};
    std::vector<std::string> defect_origin,parameter_origin,contradiction,comparison_names;
    std::vector<BoundaryMapEntry> defect_directions,parameter_directions;
    std::size_t parameters{},free_parameters{};
    std::vector<std::size_t> offsets,section_cells;
    PatternBankResponse boundary_response,section;
};
struct PatternTrainingReceipt {
    std::vector<ContextRule> rules;
    std::vector<std::size_t> statuses,ranks,origins,targets;
    std::size_t equations{},attempts{},steps{},retained{};
};
class StalkTransportPatternTraining {
    friend class StalkTransportPatternBank;
    using G=rfc::numeric::ExactActionGuard;
    rfc::numeric::ExactActionLimits limits_;
    rfc::learning::PreparedTransportPatternBankTraining prepared_;
    std::vector<PatternFrameLabel> labels_;
    rfc::learning::TransportPatternProvenance provenance_;
public:
    StalkTransportPatternTraining(const StalkTransportFit& fit,const std::vector<std::size_t>& training,
        const std::vector<std::vector<ContextView>>& atlas,std::size_t peers,
        const std::string& source,const std::string& evidence,const std::vector<PatternFrameLabel>& labels,
        bool observed_inputs,bool declared_templates)
        :limits_(fit.limits_),prepared_([&] {
            G guard(limits_);return rfc::learning::PreparedTransportPatternBankTraining(*fit.evidence_,training,
                fit.native_views(atlas,guard),peers,guard,observed_inputs?
                rfc::learning::TransportContextInputScope::ObservedFields:rfc::learning::TransportContextInputScope::FullCarrier,
                declared_templates);
        }()),labels_(labels) {
        provenance_.source_sha256=StalkTransportPattern::digest(source);
        provenance_.evidence_sha256=StalkTransportPattern::digest(evidence);
    }
    PatternTrainingReceipt receipt() const {
        const auto& acquired=prepared_.acquisition();PatternTrainingReceipt out;
        for(const auto& rule:acquired.rules)out.rules.push_back({rule.origin,rule.forced,rule.excluded,rule.open,rule.admitted});
        for(const auto status:acquired.block_statuses)out.statuses.push_back(static_cast<std::size_t>(status));
        out.ranks=acquired.block_ranks;
        for(const auto& block:acquired.comparisons.acquisition.hypotheses){out.origins.push_back(block.origin);out.targets.push_back(block.target);}
        out.equations=acquired.comparisons.acquisition.relations.size();out.attempts=acquired.comparisons.acquisition.attempts;
        out.steps=acquired.steps;out.retained=acquired.retained;return out;
    }
};
class StalkTransportPatternBank {
    friend class StalkTransportPatternScope;
    using G=rfc::numeric::ExactActionGuard;
    using M=rfc::numeric::ExactMatrix;
    rfc::learning::PreparedTransportPatternBank bank_;
    std::uint64_t namespace_;
    rfc::numeric::ExactActionLimits limits_;
    static auto learn(const StalkTransportFit& fit,const std::vector<std::size_t>& training,
        const std::vector<std::size_t>& origins,const std::vector<std::vector<ContextView>>& atlas,std::size_t peers,
        const std::vector<std::size_t>& coordinates,const std::string& source,const std::string& evidence,
        const std::vector<PatternFrameLabel>& labels,bool observed_inputs,bool declared_templates) {
        if(coordinates.size()!=origins.size()+2) throw std::invalid_argument("complete bank coordinate declaration required");
        G guard(fit.limits_);const auto views=fit.native_views(atlas,guard);
        rfc::learning::TransportPatternProvenance p;p.source_sha256=StalkTransportPattern::digest(source);p.evidence_sha256=StalkTransportPattern::digest(evidence);
        rfc::learning::TransportPatternBankCoordinates c{coordinates[0],coordinates[1],{coordinates.begin()+2,coordinates.end()}};
        const auto bank=rfc::learning::PreparedTransportPatternBank::learn(*fit.evidence_,training,origins,views,peers,std::move(c),p,guard,
            observed_inputs?rfc::learning::TransportContextInputScope::ObservedFields:rfc::learning::TransportContextInputScope::FullCarrier,declared_templates);
        std::set<std::uint64_t> wanted;
        for(const auto& carrier:bank.endpoint().carriers) if(carrier.key>=3+origins.size()) wanted.insert(carrier.coordinate_system);
        std::vector<rfc::learning::TransportPatternFrameLabel> selected;
        for(const auto& label:labels)if(wanted.contains(label.frame))selected.push_back({label.frame,label.name,label.axes});
        return bank.with_frame_labels(std::move(selected),guard);
    }
    static auto load(const std::string& image,std::uint64_t ns,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        if(bounds.size()!=4 || image.size()>maximum) throw std::invalid_argument("admitted bank bytes and exact limits required");
        G guard({bounds[0],bounds[1],bounds[2],bounds[3]});
        const auto policy=rfc::learning::transport_pattern_bank_archive_limits(guard.limits(),maximum);
        return rfc::learning::decode_transport_pattern_bank({reinterpret_cast<const std::byte*>(image.data()),image.size()},ns,guard,policy);
    }
    static auto entries(const M& m) {
        std::vector<BoundaryMapEntry> out;for(const auto& [key,q]:m.entries())out.push_back({key.first,key.second,q.get_str(16)});return out;
    }
    static auto values(const rfc::numeric::RationalVector& v) {
        std::vector<std::string> out;for(const auto& q:v)out.push_back(q.get_str(16));return out;
    }
    static auto response(const rfc::learning::TransportPatternBankResponse& v) {
        PatternBankResponse out;out.field.origin=values(v.origin);out.field.directions=entries(v.directions);
        out.field.parameters=v.directions.cols();out.field.offsets=v.offsets;
        for(const auto& c:v.coordinates)out.coordinates.push_back(values(c));
        return out;
    }
    static auto inputs(const std::vector<PatternBankInput>& requests,G& guard) {
        std::vector<rfc::learning::TransportPatternInput> out;
        for(const auto& r:requests){guard.step();out.push_back({r.pattern,StalkTransportFit::values(r.value,guard)});}return out;
    }
    static auto sparse_inputs(const std::vector<PatternBankSparseInput>& requests,G& guard,std::size_t& retained) {
        std::vector<rfc::learning::TransportPatternSparseInput> out;guard.dimension(requests.size());
        for(const auto& r:requests) {
            guard.step();guard.dimension(r.size);retained=rfc::machine::checked_size_add(retained,r.entries.size());guard.entries(retained);
            std::vector<rfc::numeric::SparseRationalEntry> values;values.reserve(r.entries.size());
            for(const auto& e:r.entries) {
                guard.step();auto q=StalkTransportFit::scalar(e.value,guard);
                if(e.index>=r.size || q==0 || (!values.empty() && e.index<=values.back().index))
                    throw std::invalid_argument("ordered distinct nonzero sparse field coefficients required");
                values.push_back({e.index,std::move(q)});
            }
            out.push_back({r.pattern,r.frame,rfc::numeric::SparseRationalTensor(rfc::numeric::IndexDomain::linear(r.size),
                rfc::numeric::StructuralSupport::full(r.size),rfc::numeric::SparseRationalVector(r.size,std::move(values)))});
        }
        return out;
    }
    static M matrix(std::size_t rows,std::size_t columns,const std::vector<BoundaryMapEntry>& entries,G& guard) {
        guard.dimension(rows);guard.dimension(columns);M out(rows,columns);std::set<std::pair<std::size_t,std::size_t>> support;
        for(const auto& e:entries) {
            guard.step();if(e.row>=rows || e.column>=columns || !support.emplace(e.row,e.column).second)
                throw std::invalid_argument("distinct admitted interface map entries required");
            out.set(e.row,e.column,StalkTransportFit::scalar(e.value,guard));guard.entries(out.nonzeros());
        }
        return out;
    }
public:
    static auto observed_scope(const StalkTransportFit& fit,const std::vector<std::vector<ContextView>>& atlas,
        const std::string& source,const std::string& evidence) {
        G guard(fit.limits_);const auto views=fit.native_views(atlas,guard);
        rfc::learning::TransportPatternProvenance p;
        p.source_sha256=StalkTransportPattern::digest(source);p.evidence_sha256=StalkTransportPattern::digest(evidence);
        return rfc::learning::PreparedTransportPatternScope::acquire(rfc::learning::TransportContextSamples(*fit.evidence_),views,p,guard);
    }
    StalkTransportPatternBank(const StalkTransportFit& fit,const std::vector<std::size_t>& training,
        const std::vector<std::size_t>& origins,const std::vector<std::vector<ContextView>>& atlas,std::size_t peers,
        const std::vector<std::size_t>& coordinates,std::uint64_t ns,const std::string& source,const std::string& evidence,
        const std::vector<PatternFrameLabel>& labels,bool observed_inputs=false,bool declared_templates=false)
        :bank_(learn(fit,training,origins,atlas,peers,coordinates,source,evidence,labels,observed_inputs,declared_templates)),namespace_(ns),limits_(fit.limits_) {
        if(!ns)throw std::invalid_argument("nonzero model bank namespace required");
    }
    StalkTransportPatternBank(const StalkTransportPatternTraining& training,const std::vector<std::size_t>& origins,
        const std::vector<std::size_t>& coordinates,std::uint64_t ns)
        :bank_([&] {
            if(!ns || coordinates.size()!=origins.size()+2)
                throw std::invalid_argument("complete bank coordinates and namespace required");
            G guard(training.limits_);
            rfc::learning::TransportPatternBankCoordinates c{coordinates[0],coordinates[1],{coordinates.begin()+2,coordinates.end()}};
            const auto bank=training.prepared_.compile(origins,std::move(c),training.provenance_,guard);
            std::set<std::uint64_t> wanted;
            for(const auto& carrier:bank.endpoint().carriers)if(carrier.key>=3+origins.size())wanted.insert(carrier.coordinate_system);
            std::vector<rfc::learning::TransportPatternFrameLabel> labels;
            for(const auto& label:training.labels_)if(wanted.contains(label.frame))labels.push_back({label.frame,label.name,label.axes});
            return bank.with_frame_labels(std::move(labels),guard);
        }()),namespace_(ns),limits_(training.limits_) {}
    StalkTransportPatternBank(const std::string& image,std::uint64_t ns,const std::vector<std::size_t>& limits,std::size_t maximum)
        :bank_(load(image,ns,limits,maximum)),namespace_(ns),limits_{limits[0],limits[1],limits[2],limits[3]} {}
    StalkTransportPatternBank(const StalkTransportPatternBank& parent,const std::vector<std::size_t>& patterns,
        const std::vector<std::size_t>& coordinates,std::uint64_t ns)
        :bank_([&] {
            if(coordinates.size()!=2 || !ns || ns==parent.namespace_)
                throw std::invalid_argument("new selection namespace and two coordinate identities required");
            G guard(parent.limits_);return parent.bank_.select(patterns,coordinates[0],coordinates[1],guard);
        }()),namespace_(ns),limits_(parent.limits_) {}
    std::string encode() const {
        const auto policy=rfc::learning::transport_pattern_bank_archive_limits(limits_);
        const auto bytes=rfc::learning::encode_transport_pattern_bank(bank_,namespace_,policy);
        return {reinterpret_cast<const char*>(bytes.data()),bytes.size()};
    }
    PatternBankReceipt receipt() const {
        PatternBankReceipt out;out.declared_templates=bank_.provenance().declared_templates;
        auto& c=out.common;c.weights=values(bank_.weights());c.directions=entries(bank_.parameter_directions());
        c.parameters=bank_.parameter_directions().cols();c.retained=bank_.retained_entries();
        for(const auto& coordinate:bank_.weight_coordinates())c.coordinates.push_back({coordinate.occurrence,coordinate.row,coordinate.column});
        for(const auto& image:bank_.input_images()){out.rows.push_back(image.rows());out.columns.push_back(image.cols());out.images.push_back(entries(image));}
        const auto& source=bank_.endpoint().source;
        for(std::size_t i=0;i<bank_.input_images().size();++i)
            out.pattern_keys.push_back(std::to_string(rfc::storage::stable_cell_key(source,{rfc::core::Grade{1},static_cast<std::uint32_t>(i)}).key));
        for(const auto key:source.grade(rfc::core::Grade{1}).occurrence_identities)c.occurrence_keys.push_back(std::to_string(key.value));
        for(const auto& label:bank_.frame_labels())c.frames.push_back({label.frame,label.name,label.axes});
        for(const auto& carrier:bank_.endpoint().carriers){c.carrier_frames.push_back(carrier.coordinate_system);c.carrier_dimensions.push_back(carrier.domain.logical_size());}
        const auto archive=rfc::storage::encode_rfcx(source);c.source_archive.assign(reinterpret_cast<const char*>(archive.bytes.data()),archive.bytes.size());
        const auto& p=bank_.provenance();c.source_sha256=StalkTransportPattern::hex(p.source_sha256);c.evidence_sha256=StalkTransportPattern::hex(p.evidence_sha256);
        c.observations=p.observations;c.peers=p.peers;c.provenance_bytes=StalkTransportPattern::provenance_bytes(bank_.endpoint(),3);return out;
    }
    PatternBankResponse response(const std::vector<PatternBankInput>& requests,bool shared) const {
        G guard(limits_);return response(bank_.response(inputs(requests,guard),guard,shared));
    }
    PatternBankResponse response_sparse(const std::vector<PatternBankSparseInput>& requests,bool shared) const {
        G guard(limits_);auto retained=bank_.retained_entries();
        return response(bank_.response_sparse(sparse_inputs(requests,guard,retained),guard,shared));
    }
    std::vector<PatternBankCandidate> response_sparse_candidates(const std::vector<PatternBankSparseInput>& requests,bool shared) const {
        G guard(limits_);auto retained=bank_.retained_entries();std::vector<PatternBankCandidate> out;
        for(const auto& value:bank_.response_sparse_candidates(sparse_inputs(requests,guard,retained),guard,shared)) {
            PatternBankCandidate candidate;candidate.pattern=value.pattern;candidate.supported=value.response.has_value();
            if(value.response)candidate.response=response(*value.response);
            out.push_back(std::move(candidate));
        }
        return out;
    }
    std::vector<PatternBankResponse> response_sparse_batch(const std::vector<std::vector<PatternBankSparseInput>>& batch,bool shared) const {
        G guard(limits_);guard.dimension(batch.size());auto retained=bank_.retained_entries();
        std::vector<std::vector<rfc::learning::TransportPatternSparseInput>> inputs;
        for(const auto& requests:batch) inputs.push_back(sparse_inputs(requests,guard,retained));
        std::vector<PatternBankResponse> out;
        for(const auto& value:bank_.response_sparse_batch(inputs,guard,shared)) out.push_back(response(value));
        return out;
    }
    PatternBankGluing glue(const std::vector<PatternBankInput>& requests,const std::vector<PatternBankComparison>& comparisons,bool shared) const {
        G guard(limits_);return glue_value(bank_.glue(inputs(requests,guard),native_comparisons(comparisons,guard),guard,shared));
    }
    PatternBankGluing glue_sparse(const std::vector<PatternBankSparseInput>& requests,const std::vector<PatternBankComparison>& comparisons,bool shared) const {
        G guard(limits_);auto retained=bank_.retained_entries();
        return glue_value(bank_.glue_sparse(sparse_inputs(requests,guard,retained),native_comparisons(comparisons,guard),guard,shared));
    }
    std::vector<PatternProgramResult> response_sparse_program(const std::vector<PatternProgramNode>& nodes) const {
        G guard(limits_);auto retained=bank_.retained_entries();
        std::vector<rfc::learning::TransportPatternProgramNode> declarations;guard.dimension(nodes.size());
        for(const auto& node:nodes) {
            guard.step();rfc::learning::TransportPatternProgramNode value;
            value.name=node.name;value.pattern=node.pattern;
            if(node.has_input)value.input=sparse_inputs({node.input},guard,retained).front();
            for(const auto& edge:node.edges) {
                guard.step();auto map=matrix(edge.rows,edge.columns,edge.map,guard);
                retained=rfc::machine::checked_size_add(retained,map.nonzeros());guard.entries(retained);
                value.edges.push_back({edge.node,edge.occurrence,edge.source_frame,edge.target_frame,std::move(map)});
            }
            declarations.push_back(std::move(value));
        }
        std::vector<PatternProgramResult> out;
        for(const auto& result:bank_.response_sparse_program(declarations,guard)) {
            PatternProgramResult value;
            switch(result.status) {
                case rfc::learning::TransportPatternProgramStatus::Supported:value.status="supported";break;
                case rfc::learning::TransportPatternProgramStatus::OutsideDomain:value.status="outside-domain";break;
                case rfc::learning::TransportPatternProgramStatus::OpenInput:value.status="open-input";break;
                case rfc::learning::TransportPatternProgramStatus::Blocked:value.status="blocked";break;
            }
            value.has_input=result.input.has_value();value.has_response=result.response.has_value();
            value.size=result.input_directions.rows();value.parameters=result.input_directions.cols();
            value.response_size=result.parameter_directions.rows();
            value.input_directions=entries(result.input_directions);value.parameter_directions=entries(result.parameter_directions);
            if(result.input)for(const auto& e:result.input->values().entries())value.input.push_back({e.index,e.value.get_str(16)});
            if(result.response)value.response=response(*result.response);
            out.push_back(std::move(value));
        }
        return out;
    }
private:
    static std::vector<rfc::learning::TransportPatternComparison> native_comparisons(const std::vector<PatternBankComparison>& comparisons,G& guard) {
        std::vector<rfc::learning::TransportPatternComparison> declarations;
        for(const auto& c:comparisons) {
            guard.step();declarations.push_back({c.name,c.left,c.right,c.left_frame,c.right_frame,c.interface_frame,
                matrix(c.rows,c.left_columns,c.left_map,guard),matrix(c.rows,c.right_columns,c.right_map,guard)});
        }
        return declarations;
    }
    PatternBankGluing glue_value(const rfc::learning::TransportPatternGluing& value) const {
        PatternBankGluing out;
        out.consistent=value.consistent;out.forced=value.forced;out.parameters=bank_.parameter_directions().cols();
        out.defect_origin=values(value.defect_origin);out.defect_directions=entries(value.defect_directions);out.contradiction=values(value.contradiction);
        out.parameter_origin=values(value.parameter_origin);out.parameter_directions=entries(value.parameter_directions);
        out.free_parameters=value.parameter_directions.cols();out.offsets=value.offsets;out.section_cells=value.section_cells;out.comparison_names=value.comparison_names;
        if(value.boundary_response)out.boundary_response=response(*value.boundary_response);
        if(value.section)out.section=response(*value.section);
        return out;
    }
};
struct PatternScopeReceipt {
    std::string source_archive;
    std::vector<std::string> keys,types;
    std::size_t observations{},retained{};
};
class StalkTransportPatternScope {
    rfc::learning::PreparedTransportPatternScope scope_;
    std::uint64_t namespace_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_;
    auto limits_for(const StalkTransportPatternBank& bank) const {
        auto limits=bank.limits_;
        limits.dimension=std::min(limits.dimension,bounds_[0]);limits.entries=std::min(limits.entries,bounds_[1]);
        limits.steps=std::min(limits.steps,bounds_[2]);limits.integer_bits=std::min(limits.integer_bits,bounds_[3]);return limits;
    }
    static auto load(const std::string& image,std::uint64_t ns,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        if(bounds.size()!=4 || image.size()>maximum)throw std::invalid_argument("admitted scope bytes and exact limits required");
        rfc::numeric::ExactActionGuard guard({bounds[0],bounds[1],bounds[2],bounds[3]});
        const auto policy=rfc::learning::transport_pattern_bank_archive_limits(guard.limits(),maximum);
        return rfc::learning::decode_transport_pattern_scope({reinterpret_cast<const std::byte*>(image.data()),image.size()},ns,guard,policy);
    }
public:
    StalkTransportPatternScope(const StalkTransportFit& fit,const std::vector<std::vector<ContextView>>& atlas,
        std::uint64_t ns,const std::string& source,const std::string& evidence,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :scope_(StalkTransportPatternBank::observed_scope(fit,atlas,source,evidence)),namespace_(ns),bounds_(bounds),maximum_(maximum) {
        if(!ns || bounds.size()!=4 || !maximum)throw std::invalid_argument("nonzero scope namespace and admission required");
    }
    StalkTransportPatternScope(const std::string& image,std::uint64_t ns,const std::vector<std::size_t>& bounds,std::size_t maximum)
        :scope_(load(image,ns,bounds,maximum)),namespace_(ns),bounds_(bounds),maximum_(maximum) {}
    StalkTransportPatternScope(const StalkTransportPatternScope& parent,const StalkTransportPatternBank& bank,std::uint64_t ns)
        :scope_([&]{if(!ns || ns==parent.namespace_ || ns==bank.namespace_)throw std::invalid_argument("distinct local scope namespace required");
            rfc::numeric::ExactActionGuard guard(parent.limits_for(bank));return parent.scope_.for_bank(bank.bank_,guard);}()),
            namespace_(ns),bounds_(parent.bounds_),maximum_(parent.maximum_) {}
    std::string encode() const {
        const auto policy=rfc::learning::transport_pattern_bank_archive_limits(
            {bounds_[0],bounds_[1],bounds_[2],bounds_[3]},maximum_);
        const auto bytes=rfc::learning::encode_transport_pattern_scope(scope_,namespace_,policy);
        return {reinterpret_cast<const char*>(bytes.data()),bytes.size()};
    }
    PatternScopeReceipt receipt() const {
        PatternScopeReceipt out;out.observations=scope_.provenance().observations;out.retained=scope_.retained_entries();
        const auto& source=scope_.endpoint().source;const auto& arena=source.grade(rfc::core::Grade{1});
        for(std::size_t i=0;i<arena.cell_count();++i) {
            out.keys.push_back(std::to_string(rfc::storage::stable_cell_key(source,{rfc::core::Grade{1},static_cast<std::uint32_t>(i)}).key));
            out.types.push_back(source.signature().type(arena.relation_types[i]).name);
        }
        const auto bytes=rfc::storage::encode_rfcx(source).bytes;
        out.source_archive.assign(reinterpret_cast<const char*>(bytes.data()),bytes.size());return out;
    }
    std::vector<std::vector<std::size_t>> coverage(const StalkTransportPatternBank& bank) const {
        rfc::numeric::ExactActionGuard guard(limits_for(bank));return scope_.coverage(bank.bank_,guard).patterns;
    }
};
} // namespace rexgraph::python
