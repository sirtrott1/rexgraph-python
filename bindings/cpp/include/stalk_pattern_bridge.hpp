#pragma once
#include "stalk_transport_bridge.hpp"
#include <rfc/learning/context_transport_pattern_codec.hpp>

namespace rexgraph::python {
struct PatternFrameLabel {std::size_t frame{};std::string name;std::vector<std::string> axes;};
struct PatternReceipt {
    std::vector<std::string> weights;
    std::vector<BoundaryMapEntry> directions, input_image;
    std::vector<std::vector<std::size_t>> coordinates;
    std::vector<std::string> occurrence_keys;
    std::vector<PatternFrameLabel> frames;
    std::vector<std::size_t> carrier_frames,carrier_dimensions;
    std::string source_sha256,evidence_sha256,source_archive;
    std::size_t input_rows{},input_columns{},parameters{},observations{},peers{},retained{},provenance_bytes{};
};
struct PatternResponse {
    std::vector<std::string> origin,coordinates;
    std::vector<BoundaryMapEntry> directions;
    std::vector<std::size_t> offsets;
    std::size_t parameters{};
};
class StalkTransportPattern {
    friend class StalkTransportPatternBank;
    friend class StalkTransportPatternTraining;
    rfc::learning::PreparedTransportPattern pattern_;
    std::uint64_t namespace_;
    rfc::numeric::ExactActionLimits limits_;
    static auto digest(const std::string& text) {
        if(text.size()!=64) throw std::invalid_argument("SHA256 hexadecimal provenance reference required");
        std::array<std::byte,32> result;
        const auto digit=[](char c)->unsigned {
            if(c>='0' && c<='9')return static_cast<unsigned>(c-'0');
            if(c>='a' && c<='f')return static_cast<unsigned>(c-'a'+10);
            throw std::invalid_argument("canonical SHA256 hexadecimal reference required");
        };
        for(std::size_t i=0;i<32;++i) result[i]=static_cast<std::byte>(16*digit(text[2*i])+digit(text[2*i+1]));
        return result;
    }
    static std::string hex(const std::array<std::byte,32>& bytes) {
        std::string result;const char* digits="0123456789abcdef";
        for(const auto b:bytes) {const auto n=std::to_integer<unsigned>(b);result.push_back(digits[n>>4]);result.push_back(digits[n&15]);}
        return result;
    }
    static std::size_t provenance_bytes(const rfc::storage::TemporalEndpoint& endpoint,std::uint64_t key) {
        const auto record=std::find_if(endpoint.tensors.begin(),endpoint.tensors.end(),[&](const auto& t){return t.key==key;});
        if(record==endpoint.tensors.end())throw std::logic_error("pattern provenance record missing");
        rfc::storage::temporal_codec_detail::Writer writer;rfc::storage::temporal_codec_detail::tensor_record(writer,*record);
        return writer.size();
    }
    static auto learn(const StalkTransportFit& fit,const std::vector<std::size_t>& training,std::size_t origin,
        const std::vector<std::vector<ContextView>>& atlas,std::size_t minimum_peers,
        const std::vector<std::size_t>& coordinates,const std::string& source,const std::string& evidence,
        const std::vector<PatternFrameLabel>& labels,bool observed_inputs) {
        if(coordinates.size()!=3 || std::any_of(coordinates.begin(),coordinates.end(),[](auto x){return !x;}))
            throw std::invalid_argument("three nonzero pattern coordinate identities required");
        rfc::numeric::ExactActionGuard guard(fit.limits_);
        const auto views=fit.native_views(atlas,guard);
        rfc::learning::TransportPatternProvenance provenance;provenance.source_sha256=digest(source);provenance.evidence_sha256=digest(evidence);
        const auto pattern=rfc::learning::PreparedTransportPattern::learn(*fit.evidence_,training,origin,views,
            minimum_peers,{coordinates[0],coordinates[1],coordinates[2]},provenance,guard,
            observed_inputs?rfc::learning::TransportContextInputScope::ObservedFields:rfc::learning::TransportContextInputScope::FullCarrier);
        std::set<std::uint64_t> wanted;
        for(const auto& carrier:pattern.endpoint().carriers) if(carrier.key==2 || carrier.key>=5) wanted.insert(carrier.coordinate_system);
        std::vector<rfc::learning::TransportPatternFrameLabel> selected;
        for(const auto& label:labels) if(wanted.contains(label.frame)) selected.push_back({label.frame,label.name,label.axes});
        return pattern.with_frame_labels(std::move(selected),guard);
    }
    static auto load(const std::string& bytes,std::uint64_t ns,const std::vector<std::size_t>& limits,std::size_t maximum) {
        if(limits.size()!=4 || bytes.size()>maximum) throw std::invalid_argument("admitted pattern image and exact action limits required");
        rfc::numeric::ExactActionGuard guard({limits[0],limits[1],limits[2],limits[3]});
        rfc::storage::NativeSeriesLimits policy;policy.max_archive_bytes=maximum;policy.temporal.max_file_bytes=maximum;
        policy.temporal.max_dimension=std::max(limits[0],limits[1]);policy.temporal.max_records=limits[1];
        policy.temporal.max_integer_bytes=rfc::machine::checked_size_add(limits[3]/8,1);
        return rfc::learning::decode_transport_pattern({reinterpret_cast<const std::byte*>(bytes.data()),bytes.size()},ns,guard,policy);
    }
public:
    StalkTransportPattern(const StalkTransportFit& fit,const std::vector<std::size_t>& training,std::size_t origin,
        const std::vector<std::vector<ContextView>>& atlas,std::size_t minimum_peers,const std::vector<std::size_t>& coordinates,
        std::uint64_t ns,const std::string& source,const std::string& evidence,const std::vector<PatternFrameLabel>& labels,bool observed_inputs=false)
        :pattern_(learn(fit,training,origin,atlas,minimum_peers,coordinates,source,evidence,labels,observed_inputs)),namespace_(ns),limits_(fit.limits_) {
        if(!ns) throw std::invalid_argument("nonzero model namespace required");
    }
    StalkTransportPattern(const std::string& bytes,std::uint64_t ns,const std::vector<std::size_t>& limits,std::size_t maximum)
        :pattern_(load(bytes,ns,limits,maximum)),namespace_(ns),limits_{limits[0],limits[1],limits[2],limits[3]} {}
    std::string encode() const {
        const auto bytes=rfc::learning::encode_transport_pattern(pattern_,namespace_);
        return {reinterpret_cast<const char*>(bytes.data()),bytes.size()};
    }
    PatternReceipt receipt() const {
        PatternReceipt result;result.input_rows=pattern_.input_image().rows();result.input_columns=pattern_.input_image().cols();
        result.parameters=pattern_.parameter_directions().cols();result.retained=pattern_.retained_entries();
        for(const auto& q:pattern_.weights()) result.weights.push_back(q.get_str(16));
        for(const auto& [key,q]:pattern_.input_image().entries()) result.input_image.push_back({key.first,key.second,q.get_str(16)});
        for(const auto& [key,q]:pattern_.parameter_directions().entries()) result.directions.push_back({key.first,key.second,q.get_str(16)});
        for(const auto& c:pattern_.weight_coordinates()) result.coordinates.push_back({c.occurrence,c.row,c.column});
        const auto& source=pattern_.endpoint().source;
        for(const auto& label:pattern_.frame_labels()) result.frames.push_back({label.frame,label.name,label.axes});
        for(const auto& carrier:pattern_.endpoint().carriers) {
            result.carrier_frames.push_back(carrier.coordinate_system);result.carrier_dimensions.push_back(carrier.domain.logical_size());
        }
        for(const auto key:source.grade(rfc::core::Grade{1}).occurrence_identities) result.occurrence_keys.push_back(std::to_string(key.value));
        const auto archive=rfc::storage::encode_rfcx(source);result.source_archive.assign(reinterpret_cast<const char*>(archive.bytes.data()),archive.bytes.size());
        const auto& p=pattern_.provenance();result.source_sha256=hex(p.source_sha256);result.evidence_sha256=hex(p.evidence_sha256);
        result.observations=p.observations;result.peers=p.peers;result.provenance_bytes=provenance_bytes(pattern_.endpoint(),4);return result;
    }
    PatternResponse response(const std::vector<std::string>& values,bool shared) const {
        rfc::numeric::ExactActionGuard guard(limits_);const auto input=StalkTransportFit::values(values,guard);
        const auto value=pattern_.response(input,guard,shared);PatternResponse out;
        for(const auto& q:value.origin) out.origin.push_back(q.get_str(16));
        for(const auto& q:value.coordinates) out.coordinates.push_back(q.get_str(16));
        for(const auto& [key,q]:value.directions.entries()) out.directions.push_back({key.first,key.second,q.get_str(16)});
        out.offsets=value.offsets;out.parameters=value.directions.cols();return out;
    }
};
} // namespace rexgraph::python
