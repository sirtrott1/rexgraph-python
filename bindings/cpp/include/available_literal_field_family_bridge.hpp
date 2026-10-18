#pragma once

#include "field_evidence_admission_bridge.hpp"
#include "literal_field_family_bridge.hpp"
#include <rfc/learning/relational_available_literal_field_family_codec.hpp>

namespace rexgraph::python {
struct JointReaderInput {
    LiteralFamilyInput input;
    std::vector<std::uint64_t> origins;
};
struct JointObservationReaders {
    std::uint64_t observation{};
    std::vector<JointReaderInput> inputs;
};
struct JointMemberOrigins {
    std::vector<std::uint64_t> program;
    std::uint64_t root{};
    std::vector<std::vector<std::uint64_t>> targets;
};
struct JointSink {
    int (*write)(void*,const char*,std::size_t)=nullptr;
    void* context{};
};
namespace available_joint_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
using Context=inference::RelationalAvailabilityContext;
using Model=learning::PreparedAvailableRelationalLiteralFieldFamily;
using Result=learning::AvailableRelationalLiteralJointResult;
using Evaluation=learning::AvailableRelationalLiteralJointEvaluation;
inline const Context& authority(const std::shared_ptr<AvailabilityContextBridge>& context) {
    if(!context)throw std::invalid_argument("actual independent availability context required");
    return context->native_owner();
}
inline learning::RelationalLiteralFamilyInput input(const LiteralFamilyInput& value) {
    if(bool(value.literal)==bool(value.graph))throw std::invalid_argument("one actual literal or known-graph input required");
    if(value.literal)return value.literal->native_owner();
    return value.graph->native_owner();
}
inline learning::RelationalAvailableJointInput reader(const JointReaderInput& row) {
    const auto& k=row.origins;
    if(k.size()!=12)throw std::invalid_argument("twelve complete joint reader origin keys required");
    return {input(row.input),{{k[0],k[1],k[2],k[3]},{k[4],k[5],k[6]},{k[7],k[8],k[9],k[10]},k[11]}};
}
inline std::vector<learning::RelationalAvailableJointInput> inputs(const std::vector<JointReaderInput>& rows,
        const std::vector<std::size_t>& bounds) {
    evidence_bridge_detail::bound(rows.size(),bounds.at(0));
    std::vector<learning::RelationalAvailableJointInput> out;
    for(const auto& row:rows)out.push_back(reader(row));
    return out;
}
inline std::vector<std::vector<learning::RelationalJointObservationReaders>> readers(
        const std::vector<std::vector<JointObservationReaders>>& rows,const std::vector<std::size_t>& bounds) {
    evidence_bridge_detail::bound(rows.size(),bounds.at(0));
    std::vector<std::vector<learning::RelationalJointObservationReaders>> out;
    std::size_t count=rows.size();
    for(const auto& member:rows) {
        count=machine::checked_size_add(count,member.size());evidence_bridge_detail::bound(count,bounds.at(1));
        std::vector<learning::RelationalJointObservationReaders> next;
        for(const auto& row:member) {
            count=machine::checked_size_add(count,row.inputs.size());evidence_bridge_detail::bound(count,bounds.at(1));
            next.push_back({row.observation,inputs(row.inputs,bounds)});
        }
        out.push_back(std::move(next));
    }
    return out;
}
inline JointReaderInput wrap(const learning::RelationalAvailableJointInput& row,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
    JointReaderInput out;
    if(const auto* literal=std::get_if<inference::OwnedRelationalLiteralInput>(&row.input))
        out.input.literal=LiteralInputBridge::adopt(*literal,bounds,maximum);
    else out.input.graph=KnownGraphInputBridge::adopt(std::get<inference::RelationalKnownTensorGraphInput>(row.input),bounds,maximum);
    const auto& k=row.origins;
    out.origins={k.input.asset,k.input.value,k.input.accession,k.input.reconstruction,k.original_program.program,k.original_program.maps,
        k.original_program.metrics,k.original_input.asset,k.original_input.value,k.original_input.accession,k.original_input.reconstruction,k.reader};
    return out;
}
inline std::vector<std::shared_ptr<ValueOriginBridge>> origins(const std::vector<inference::OwnedRelationalValueOrigin>& rows,
        std::size_t maximum,std::size_t count) {
    evidence_bridge_detail::bound(rows.size(),count);std::vector<std::shared_ptr<ValueOriginBridge>> out;
    for(const auto& row:rows)out.push_back(ValueOriginBridge::adopt(row,maximum));
    return out;
}
inline numeric::RationalVector free_values(const std::vector<std::string>& values,
        const std::vector<std::size_t>& bounds,GraphCheck check) {
    numeric::ExactActionGuard guard(limits(bounds),check.callback());guard.dimension(values.size());numeric::RationalVector out;
    for(const auto& value:values) {
        guard.step();numeric::Rational q;
        if(q.set_str(value,16)!=0||q.get_den()==0)throw std::invalid_argument("exact complete joint free values required");
        q.canonicalize();guard.value(q);out.push_back(std::move(q));
    }
    return out;
}
} // namespace available_joint_bridge_detail

inline LiteralFamilyInput replay_joint_reader_input(const LiteralFamilyInput& input,
        std::shared_ptr<FieldProgramBridge> program,const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
    using namespace rfc;using namespace available_joint_bridge_detail;
    if(!program)throw std::invalid_argument("actual selected member program required");
    const auto value=available_joint_bridge_detail::input(input);const auto original=learning::available_joint_detail::raw(value);
    const auto frozen_bounds=bounds;const auto accession=original.accession();const auto target=program->native_owner();const auto policy=limits(frozen_bounds);
    if(inference::encode_relational_field_program(accession.program())!=inference::encode_relational_field_program(target))
        throw std::invalid_argument("reader replay requires the same complete source/program/map/metric image");
    const auto original_packet=std::visit([&](const auto& row) {
        using Value=std::decay_t<decltype(row)>;
        if constexpr(std::is_same_v<Value,inference::OwnedRelationalLiteralInput>)return inference::encode_relational_literal_input(row,maximum);
        else return inference::encode_relational_known_tensor_graph_input(row,maximum);
    },value);
    if(check.run)check.callback()();
    const inference::OwnedRelationalInputAccession replayed(accession.asset(),target,accession.declaration(),policy,check.callback());
    if(const auto* literal=std::get_if<inference::OwnedRelationalLiteralInput>(&value)) {
        const inference::OwnedRelationalLiteralInput result(replayed,literal->declaration(),policy,check.callback());
        if(!result.input().accession().program().same_owner(target)||inference::encode_relational_literal_input(result,maximum)!=original_packet)
            throw std::invalid_argument("literal replay changes the complete retained reader image");
        return {LiteralInputBridge::adopt(result,frozen_bounds,maximum),{}};
    }
    const auto result=std::get<inference::RelationalKnownTensorGraphInput>(value).graph().admit(replayed,policy,check.callback());
    if(!result.input().accession().program().same_owner(target)||inference::encode_relational_known_tensor_graph_input(result,maximum)!=original_packet)
        throw std::invalid_argument("graph replay changes the complete retained reader image");
    return {{},KnownGraphInputBridge::adopt(result,frozen_bounds,maximum)};
}

class AvailableJointPacketBridge {
public:
    using Context=available_joint_bridge_detail::Context;
    using Admit=std::function<void(const Context&,const std::function<void()>&)>;
    AvailableJointPacketBridge(rfc::storage::PreparedBytePacket packet,Admit admit)
        :packet_(std::move(packet)),admit_(std::move(admit)){}
    std::size_t size() const {return packet_.size();}
    std::string bytes(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        const auto& actual=available_joint_bridge_detail::authority(context);admit_(actual,check.callback());
        return program_bridge_detail::text(packet_.bytes(check.callback()));
    }
    std::size_t stream(JointSink sink,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        if(!sink.write)throw std::invalid_argument("synchronous packet sink required");
        const auto& actual=available_joint_bridge_detail::authority(context);admit_(actual,check.callback());std::size_t count{};
        packet_.stream([&](std::span<const std::byte> chunk) {
            if(!sink.write(sink.context,reinterpret_cast<const char*>(chunk.data()),chunk.size()))throw std::runtime_error("joint packet sink failed");
            count=rfc::machine::checked_size_add(count,chunk.size());
        },check.callback());return count;
    }
    std::shared_ptr<ValueOriginBridge> publish(std::uint64_t key,const std::string& history,std::uint64_t cutoff,unsigned role,
            std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum,
            const std::vector<std::size_t>& admission,GraphCheck check) const {
        using namespace rfc;const auto source_image=history;const auto frozen_bounds=bounds;const auto frozen_admission=admission;
        const auto& actual=available_joint_bridge_detail::authority(context);admit_(actual,check.callback());
        const auto cap=ValueOriginBridge::admission_limits(frozen_admission);
        if(size()>maximum||size()>cap.payload_bytes)throw std::length_error("joint origin payload byte bound");
        const auto policy=program_bridge_detail::reader(frozen_bounds,maximum);storage::NativeSeriesLimits limits;limits.temporal=policy;limits.max_archive_bytes=maximum;
        auto source=storage::decode_native_series(program_bridge_detail::bytes(source_image),limits);
        inference::OwnedRelationalValueOrigin origin(key,storage::NativeSeriesPrefix::capture(source,cutoff),
            static_cast<inference::RelationalDependencyRole>(role),packet_.bytes(check.callback()),cap);
        return ValueOriginBridge::adopt(std::move(origin),maximum);
    }
private:
    rfc::storage::PreparedBytePacket packet_;
    Admit admit_;
};
class AvailableJointResultBridge;
class AvailableJointEvaluationBridge;
class AvailableJointFamilyBridge {
public:
    using Model=available_joint_bridge_detail::Model;
    AvailableJointFamilyBridge(std::shared_ptr<FieldFamilyBridge> family,
            const std::vector<std::shared_ptr<AvailableEvidenceBridge>>& members,
            const std::vector<std::vector<JointObservationReaders>>& ledger,const std::vector<std::uint64_t>& interfaces,
            std::uint64_t origin,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,
            std::size_t maximum,GraphCheck check):bounds_(bounds),maximum_(maximum) {
        using namespace available_joint_bridge_detail;if(!family)throw std::invalid_argument("actual raw family owner required");
        rfc::learning::relational_evidence_detail::same_limits(family->native_owner().limits(),limits(bounds));
        std::vector<rfc::learning::PreparedAvailableRelationalEvidence> values;
        for(const auto& member:members){if(!member)throw std::invalid_argument("actual available evidence owner required");values.push_back(member->native_owner());}
        owner_=std::make_shared<const Model>(family->native_owner(),std::move(values),readers(ledger,bounds),
            rfc::learning::RelationalJointFamilyOrigins{interfaces,origin},authority(context),maximum,check.callback());
    }
    static std::shared_ptr<AvailableJointFamilyBridge> adopt(Model value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<AvailableJointFamilyBridge>(new AvailableJointFamilyBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<AvailableJointFamilyBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace available_joint_bridge_detail;
        auto value=rfc::learning::decode_available_relational_literal_field_family(bytes(image),authority(context),program_bridge_detail::reader(bounds,maximum),limits(bounds),check.callback());
        rfc::learning::relational_evidence_detail::same_limits(value.family().limits(),limits(bounds));
        return adopt(std::move(value),bounds,maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {owner_->admit(available_joint_bridge_detail::authority(context),check.callback());}
    bool same_owner(const AvailableJointFamilyBridge& other) const {return owner_->same_owner(*other.owner_);}
    std::shared_ptr<FieldFamilyBridge> family(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return FieldFamilyBridge::adopt(owner_->family(),bounds_,maximum_);
    }
    std::vector<std::shared_ptr<AvailableEvidenceBridge>> members(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);std::vector<std::shared_ptr<AvailableEvidenceBridge>> out;
        for(const auto& member:owner_->members())out.push_back(AvailableEvidenceBridge::adopt(member,bounds_,maximum_));
        return out;
    }
    std::vector<std::vector<JointObservationReaders>> training_readers(std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);std::vector<std::vector<JointObservationReaders>> out;std::size_t count=owner_->training_readers().size();
        evidence_bridge_detail::bound(count,maximum);
        for(const auto& member:owner_->training_readers()) {
            count=rfc::machine::checked_size_add(count,member.size());evidence_bridge_detail::bound(count,maximum);std::vector<JointObservationReaders> rows;
            for(const auto& row:member) {
                count=rfc::machine::checked_size_add(count,row.inputs.size());evidence_bridge_detail::bound(count,maximum);JointObservationReaders value;value.observation=row.observation;
                for(const auto& input:row.inputs)value.inputs.push_back(available_joint_bridge_detail::wrap(input,bounds_,maximum_));
                rows.push_back(std::move(value));
            }
            out.push_back(std::move(rows));
        }
        return out;
    }
    std::vector<std::uint64_t> origin_keys(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);auto out=owner_->origins().interfaces;out.push_back(owner_->origins().family);return out;
    }
    std::vector<std::shared_ptr<ValueOriginBridge>> obligations(std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);return available_joint_bridge_detail::origins(owner_->obligations(),maximum_,maximum);
    }
    std::shared_ptr<AvailableJointPacketBridge> prepare(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto pin=*owner_;return std::make_shared<AvailableJointPacketBridge>(
            rfc::learning::prepare_available_relational_literal_field_family(pin,maximum_,check.callback()),
            [pin](const auto& actual,const auto& callback){pin.admit(actual,callback);});
    }
    std::shared_ptr<AvailableJointPacketBridge> setting(const std::vector<std::string>& free,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        using namespace available_joint_bridge_detail;const auto pin=*owner_;const auto values=free_values(free,bounds_,check);
        return std::make_shared<AvailableJointPacketBridge>(rfc::learning::prepare_relational_joint_setting_payload(pin,values,authority(context),maximum_,check.callback()),
            [pin](const auto& actual,const auto& callback){pin.admit(actual,callback);});
    }
    static std::shared_ptr<AvailableJointPacketBridge> interface_payload(std::shared_ptr<FieldFamilyBridge> family,std::size_t index,
            std::size_t maximum,GraphCheck check) {
        if(!family)throw std::invalid_argument("actual complete family owner required");
        const auto pin=family->native_owner();
        return std::make_shared<AvailableJointPacketBridge>(rfc::learning::prepare_relational_joint_family_interface_payload(pin,index,maximum,check.callback()),
            [pin](const auto&,const auto& callback){if(callback)callback();(void)pin;});
    }
    static std::shared_ptr<AvailableJointPacketBridge> model_payload(std::shared_ptr<FieldFamilyBridge> family,
            const std::vector<std::shared_ptr<AvailableEvidenceBridge>>& members,const std::vector<std::vector<JointObservationReaders>>& ledger,
            const std::vector<std::uint64_t>& interfaces,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace available_joint_bridge_detail;if(!family)throw std::invalid_argument("actual raw family owner required");
        std::vector<rfc::learning::PreparedAvailableRelationalEvidence> actual;
        for(const auto& member:members){if(!member)throw std::invalid_argument("actual available member required");actual.push_back(member->native_owner());}
        const auto prepared=rfc::learning::available_joint_detail::content(family->native_owner(),actual,readers(ledger,bounds),{interfaces,0},authority(context),maximum,check.callback());
        return std::make_shared<AvailableJointPacketBridge>(prepared.packet,[required=prepared.required](const auto& ctx,const auto& callback){
            rfc::learning::available_joint_detail::recheck(required,ctx,callback);});
    }
    std::shared_ptr<AvailableJointResultBridge> execute(std::size_t,const std::vector<JointReaderInput>&,std::shared_ptr<AvailabilityContextBridge>,GraphCheck) const;
private:
    AvailableJointFamilyBridge(Model value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const Model>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum){}
    std::shared_ptr<const Model> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableJointResultBridge {
public:
    using Result=available_joint_bridge_detail::Result;
    static std::shared_ptr<AvailableJointResultBridge> adopt(Result value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<AvailableJointResultBridge>(new AvailableJointResultBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<AvailableJointResultBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace available_joint_bridge_detail;
        auto value=rfc::learning::decode_available_relational_literal_joint_result(bytes(image),authority(context),program_bridge_detail::reader(bounds,maximum),limits(bounds),check.callback());
        rfc::learning::relational_evidence_detail::same_limits(value.model().family().limits(),limits(bounds));
        return adopt(std::move(value),bounds,maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {owner_->admit(available_joint_bridge_detail::authority(context),check.callback());}
    std::shared_ptr<AvailableJointFamilyBridge> model(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return AvailableJointFamilyBridge::adopt(owner_->model(),bounds_,maximum_);
    }
    std::shared_ptr<LiteralJointResultBridge> raw(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return LiteralJointResultBridge::adopt(owner_->raw_result(),bounds_,maximum_);
    }
    std::vector<JointReaderInput> inputs(std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);evidence_bridge_detail::bound(owner_->inputs().size(),maximum);std::vector<JointReaderInput> out;
        for(const auto& row:owner_->inputs())out.push_back(available_joint_bridge_detail::wrap(row,bounds_,maximum_));
        return out;
    }
    std::vector<std::shared_ptr<ValueOriginBridge>> obligations(std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum,GraphCheck check) const {
        admit(context,check);return available_joint_bridge_detail::origins(owner_->obligations(),maximum_,maximum);
    }
    std::shared_ptr<AvailableJointPacketBridge> prepare(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto pin=*owner_;return std::make_shared<AvailableJointPacketBridge>(rfc::learning::prepare_available_relational_literal_joint_result(pin,maximum_,check.callback()),
            [pin](const auto& actual,const auto& callback){pin.admit(actual,callback);});
    }
    std::shared_ptr<AvailableJointPacketBridge> completion(const std::vector<std::string>& free,std::uint64_t setting,
            std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        using namespace available_joint_bridge_detail;const auto pin=*owner_;const auto values=free_values(free,bounds_,check);const auto& ctx=authority(context);
        const auto payload=rfc::learning::prepare_relational_joint_completion_payload(pin,values,setting,ctx,maximum_,check.callback());
        const auto expected=ctx.require_packet(setting,rfc::inference::RelationalDependencyRole::Prospective,
            rfc::learning::prepare_relational_joint_setting_payload(pin.model(),values,ctx,maximum_,check.callback()),check.callback());
        return std::make_shared<AvailableJointPacketBridge>(payload,[pin,expected](const auto& actual,const auto& callback){pin.admit(actual,callback);actual.recheck(expected);});
    }
    std::shared_ptr<AvailableJointEvaluationBridge> evaluate(const std::vector<std::string>&,std::uint64_t,std::uint64_t,
            std::shared_ptr<AvailabilityContextBridge>,GraphCheck) const;
private:
    AvailableJointResultBridge(Result value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const Result>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum){}
    std::shared_ptr<const Result> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableJointEvaluationBridge {
public:
    using Evaluation=available_joint_bridge_detail::Evaluation;
    static std::shared_ptr<AvailableJointEvaluationBridge> adopt(Evaluation value,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<AvailableJointEvaluationBridge>(new AvailableJointEvaluationBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<AvailableJointEvaluationBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
            const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
        using namespace available_joint_bridge_detail;
        auto value=rfc::learning::decode_available_relational_literal_joint_evaluation(bytes(image),authority(context),program_bridge_detail::reader(bounds,maximum),limits(bounds),check.callback());
        rfc::learning::relational_evidence_detail::same_limits(value.parent().model().family().limits(),limits(bounds));
        return adopt(std::move(value),bounds,maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {owner_->admit(available_joint_bridge_detail::authority(context),check.callback());}
    std::shared_ptr<AvailableJointResultBridge> parent(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return AvailableJointResultBridge::adopt(owner_->parent(),bounds_,maximum_);
    }
    std::shared_ptr<LiteralJointEvaluationBridge> raw(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);return LiteralJointEvaluationBridge::adopt(owner_->raw_evaluation(),bounds_,maximum_);
    }
    std::shared_ptr<ValueOriginBridge> origin(unsigned kind,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);if(kind>1)throw std::invalid_argument("setting or completion origin required");return ValueOriginBridge::adopt(kind?owner_->completion():owner_->setting(),maximum_);
    }
    std::shared_ptr<AvailableJointPacketBridge> prepare(std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
        admit(context,check);const auto pin=*owner_;return std::make_shared<AvailableJointPacketBridge>(rfc::learning::prepare_available_relational_literal_joint_evaluation(pin,maximum_,check.callback()),
            [pin](const auto& actual,const auto& callback){pin.admit(actual,callback);});
    }
private:
    AvailableJointEvaluationBridge(Evaluation value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const Evaluation>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum){}
    std::shared_ptr<const Evaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AvailableJointResultBridge> AvailableJointFamilyBridge::execute(std::size_t member,
        const std::vector<JointReaderInput>& rows,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
    using namespace available_joint_bridge_detail;return AvailableJointResultBridge::adopt(owner_->execute(member,inputs(rows,bounds_),authority(context),maximum_,limits(bounds_),check.callback()),bounds_,maximum_);
}
inline std::shared_ptr<AvailableJointEvaluationBridge> AvailableJointResultBridge::evaluate(const std::vector<std::string>& values,
        std::uint64_t setting,std::uint64_t completion,std::shared_ptr<AvailabilityContextBridge> context,GraphCheck check) const {
    using namespace available_joint_bridge_detail;return AvailableJointEvaluationBridge::adopt(owner_->evaluate(free_values(values,bounds_,check),setting,completion,authority(context),maximum_,limits(bounds_),check.callback()),bounds_,maximum_);
}

struct JointAvailableReplay {
    std::shared_ptr<FieldFamilyBridge> family;
    std::vector<std::shared_ptr<AvailableEvidenceBridge>> members;
};
inline JointAvailableReplay rebuild_available_field_family(std::shared_ptr<FieldFamilyBridge> blueprint,
        const std::vector<JointMemberOrigins>& origins,const std::vector<std::vector<JointObservationReaders>>& ledger,
        std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check) {
    using namespace rfc;using namespace available_joint_bridge_detail;
    if(!blueprint)throw std::invalid_argument("actual raw family blueprint required");
    const auto raw=blueprint->native_owner();
    const auto recipes=origins;const auto frozen_bounds=bounds;const auto rows=readers(ledger,frozen_bounds);const auto& ctx=authority(context);const auto policy=limits(frozen_bounds);
    learning::relational_evidence_detail::same_limits(raw.limits(),policy);
    if(recipes.size()!=raw.members().size()||rows.size()!=recipes.size())throw std::invalid_argument("complete available member and reader recipe required");
    std::vector<learning::PreparedRelationalFieldEvidence> actual;JointAvailableReplay out;
    for(std::size_t m=0;m<recipes.size();++m) {
        if(check.run)check.callback()();
        const auto original=raw.members()[m];const auto program=original.program();
        if(recipes[m].targets.size()!=original.observations().size()||rows[m].size()!=original.observations().size())throw std::invalid_argument("complete original observation ledger required");
        std::vector<learning::PreparedRelationalFieldEvidence> lineage{original};
        while(const auto parent=lineage.back().parent())lineage.push_back(*parent);
        const inference::PreparedAvailableRelationalProgram available(program,admission_bridge_detail::program_keys(recipes[m].program),ctx);
        learning::PreparedAvailableRelationalEvidence rebuilt(available,lineage.back().declaration(),recipes[m].root,ctx,policy,check.callback());
        for(std::size_t stage=lineage.size()-1;stage>0;--stage) {
            const auto& child=lineage[stage-1];const auto first=lineage[stage].observations().size();std::vector<learning::RelationalAvailableTrainingObservation> suffix;
            for(std::size_t j=first;j<child.observations().size();++j) {
                const auto& observation=child.observations()[j];const auto& row=rows[m][j];
                if(row.observation!=observation.key||row.inputs.size()!=observation.input.inputs().size())throw std::invalid_argument("original observation/input order changed");
                std::vector<inference::RelationalAvailableInputSubmission> submitted;
                for(std::size_t k=0;k<row.inputs.size();++k) {
                    const auto value=learning::available_joint_detail::raw(row.inputs[k].input);const auto a=value.accession();
                    if(!a.program().same_owner(program)||inference::relational_input_accession_payload(value)!=inference::relational_input_accession_payload(observation.input.inputs()[k]))throw std::invalid_argument("original actual input program or full accession changed");
                    (void)learning::available_joint_detail::input_origins(row.inputs[k],ctx,maximum,check.callback());
                    submitted.push_back({{a.asset(),a.declaration(),value.use(),value.source_field()},row.inputs[k].origins.input});
                }
                const auto input=available.execute_inputs(submitted,ctx,policy,check.callback());const auto& keys=recipes[m].targets[j];
                if(keys.size()!=3)throw std::invalid_argument("typed target asset/value/port origin keys required");
                suffix.push_back({observation.key,input,observation.target_asset,observation.target_use,observation.target,observation.observed,
                    {keys[0],keys[1],keys[2]},observation.port,observation.observed_coordinates});
            }
            rebuilt=rebuilt.append(std::move(suffix),ctx,check.callback());
        }
        if(learning::encode_relational_field_evidence(rebuilt.evidence())!=learning::encode_relational_field_evidence(original))throw std::invalid_argument("available replay changes original evidence or parent lineage");
        actual.push_back(rebuilt.evidence());out.members.push_back(AvailableEvidenceBridge::adopt(rebuilt,bounds,maximum));
    }
    learning::PreparedRelationalFieldFamily family(raw.declaration(),std::move(actual),policy,check.callback());
    learning::prepare_relational_field_family(family,maximum,check.callback()).require_equal(learning::encode_relational_field_family(raw,maximum),check.callback());
    out.family=FieldFamilyBridge::adopt(std::move(family),bounds,maximum);return out;
}
} // namespace rexgraph::python
