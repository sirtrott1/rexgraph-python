#pragma once
#include "known_product_learning_bridge.hpp"

namespace rexgraph::python {
namespace four_use_role_application_detail {
using namespace rfc;
using namespace rfc::inference;
using Matrix=numeric::ExactMatrix;
inline OwnedRelationalFieldProgram parent(const OwnedRelationalInputUseAsset& asset,numeric::ExactActionLimits limits) {
    RelationalFieldProgramDeclaration d;d.key=20000;d.revision=1;d.differential=true;
    for(std::uint32_t word=0;word<4;++word){
        std::vector<std::uint64_t> directions;for(std::uint64_t i=1;i<=17;++i)directions.push_back(i);
        d.stalks.push_back({{{0},word},701,std::move(directions),Matrix::identity(17)});
        Matrix map(17,68);for(std::size_t i=0;i<17;++i)map.set(i,word*17+i,1);
        d.occurrences.push_back({{1},word,storage::TemporalTensorRole::CoefficientIndependentMap,std::move(map)});
    }
    std::vector<std::uint64_t> directions;for(std::uint64_t i=1000;i<1068;++i)directions.push_back(i);
    d.stalks.push_back({{{1},0},8000,std::move(directions),Matrix::identity(68)});
    d.nodes={{1,FieldProgramNodeKind::Input,{1},{},{}},{2,FieldProgramNodeKind::Boundary,{0},1,{}}};
    return {asset.publication(),std::move(d),limits};
}
inline std::vector<std::uint64_t> word_directions(std::uint64_t word) {
    std::vector<std::uint64_t> directions;for(std::uint64_t i=1;i<=17;++i)directions.push_back(i);
    for(std::uint64_t head=0;head<4;++head)if(head!=word)for(std::uint64_t role=0;role<2;++role)
        directions.push_back(10000+100*head+10*word+role);
    return directions;
}
inline OwnedRelationalFieldProgram model(const storage::NativeTemporalSeries& source,numeric::ExactActionLimits limits) {
    RelationalFieldProgramDeclaration d;d.key=20001;d.revision=1;d.differential=true;
    for(std::uint32_t word=0;word<4;++word){
        d.stalks.push_back({{{0},word},90000+word,word_directions(word),Matrix::identity(23)});
        Matrix map(23,4692);for(std::size_t i=0;i<17;++i)map.set(i,word*17+i,1);
        d.occurrences.push_back({{1},word,storage::TemporalTensorRole::CoefficientIndependentMap,std::move(map)});
    }
    std::vector<std::uint64_t> directions;for(std::uint64_t i=1000;i<1068;++i)directions.push_back(i);
    for(std::uint64_t i=2000;i<6624;++i)directions.push_back(i);
    for(std::uint32_t relation=0;relation<13;++relation)
        d.stalks.push_back({{{1},relation},9000,directions,Matrix::identity(4692)});
    for(std::size_t occurrence=4;occurrence<28;++occurrence)
        d.occurrences.push_back({{1},occurrence,storage::TemporalTensorRole::CoefficientIndependentMap,Matrix(23,4692)});
    d.nodes={{1,FieldProgramNodeKind::Input,{1},{},{}},{2,FieldProgramNodeKind::Boundary,{0},1,{}}};
    return {source.at(source.last_sequence()),std::move(d),limits};
}
inline RelationalInputAccessionDeclaration accession(const OwnedRelationalFieldProgram& program,bool lifted) {
    RelationalInputAccessionDeclaration d;d.key=lifted?40001:40000;d.node=1;
    d.source_grade={static_cast<std::uint16_t>(lifted?1:0)};d.isometric=!lifted;
    d.instances.cells={{0,1,2,3},{0}};d.instances.occurrences={{},{0,1,2,3}};
    if(lifted)d.source_stalks={program.declaration().stalks.at(4)};
    else for(std::size_t i=0;i<4;++i)d.source_stalks.push_back(program.declaration().stalks.at(i));
    if(lifted){
        d.accession=Matrix(60996,4692);d.reconstruction=Matrix(4692,60996);
        for(std::size_t i=0;i<4692;++i){
            d.reconstruction.set(i,i,1);
            for(std::size_t relation=0;relation<13;++relation)d.accession.set(4692*relation+i,i,1);
        }
    }else{d.accession=Matrix::identity(68);d.reconstruction=d.accession;}
    return d;
}
inline RelationalKnownTensorSquareDeclaration square() {
    RelationalKnownTensorSquareDeclaration d;d.key=50000;d.revision=1;d.lifted_basis=9000;d.context={{1},0};
    for(std::uint64_t i=2000;i<6624;++i)d.pair_directions.push_back(i);
    return d;
}
inline std::vector<std::string> payloads(const OwnedRelationalFieldProgram& program,
    const RelationalInputSubmission& submission,std::size_t maximum) {
    const auto p=encode_relational_field_program(program),a=encode_relational_input_use_asset(submission.asset);
    const auto access=relational_input_accession_payload(p,a,submission.declaration,submission.use,submission.source_field);
    return {program_bridge_detail::bounded(a,maximum),program_bridge_detail::bounded(
        relational_input_value_payload(a,submission.declaration,submission.use,submission.source_field),maximum),
        program_bridge_detail::bounded(access,maximum),program_bridge_detail::bounded(access,maximum)};
}
inline RelationalInputOrigins origins(const std::vector<std::uint64_t>& keys) {
    if(keys.size()!=4)throw std::invalid_argument("four actual input origins required");
    return {keys[0],keys[1],keys[2],keys[3]};
}
} // namespace four_use_role_application_detail

inline std::vector<std::shared_ptr<FieldProgramBridge>> four_use_role_programs(
    std::shared_ptr<InputUseAssetBridge> initial,const std::string& model_source,const std::vector<std::size_t>& bounds,std::size_t maximum) {
    using namespace four_use_role_application_detail;
    if(!initial||model_source.size()>maximum)throw std::invalid_argument("actual initial asset and bounded model source required");
    storage::NativeSeriesLimits policy;policy.temporal=program_bridge_detail::reader(bounds,maximum);
    const auto source=storage::decode_native_series(program_bridge_detail::bytes(model_source),policy);
    const auto limits=program_bridge_detail::limits(bounds);
    return {FieldProgramBridge::adopt(parent(initial->native_owner(),limits),bounds,maximum),
            FieldProgramBridge::adopt(model(source,limits),bounds,maximum)};
}
inline std::string four_use_role_observed_tensor(const std::string& image,const std::vector<std::size_t>& bounds,std::size_t maximum) {
    using namespace four_use_role_application_detail;
    const auto value=interface_bridge_detail::field(image,bounds,maximum);
    if(value.domain().extents()!=std::vector<std::size_t>{92}||value.support().count()!=92)
        throw std::invalid_argument("complete Gamma68 and role24 observation support required");
    numeric::RationalVector result(92);
    for(std::size_t word=0;word<4;++word)for(std::size_t tag=0;tag<17;++tag)result[23*word+tag]=value.values().coefficient(17*word+tag);
    std::size_t pair{};
    for(std::size_t head=0;head<4;++head)for(std::size_t dependent=0;dependent<4;++dependent)if(head!=dependent){
        const auto rank=head<dependent?head:head-1;
        for(std::size_t role=0;role<2;++role)result[23*dependent+17+2*rank+role]=value.values().coefficient(68+2*pair+role);
        ++pair;
    }
    numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds));
    const auto packet=program_bridge_detail::tensor_bytes(field_interface_detail::tensor(result,guard));
    if(packet.size()>maximum)throw std::length_error("complete observation byte bound");
    return packet;
}
inline std::string four_use_role_publisher_history(const std::string& image,std::uint64_t name_space,std::uint64_t stages,
    const std::vector<std::size_t>& bounds,std::size_t maximum) {
    using namespace rfc;
    if(!name_space||!stages||stages>5||image.size()>maximum)throw std::invalid_argument("bounded actual publisher stages required");
    storage::NativeSeriesLimits policy;policy.temporal=program_bridge_detail::reader(bounds,maximum);
    const auto original=storage::decode_native_series(program_bridge_detail::bytes(image),policy);
    if(original.source_namespace()==name_space)throw std::invalid_argument("separate publisher namespace required");
    const auto publication=original.at(original.last_sequence());
    storage::TemporalEndpoint endpoint(publication.endpoint().source,1);
    storage::NativeTemporalSeries publisher(endpoint,name_space);
    for(std::uint64_t cut=2;cut<=stages;++cut){endpoint.sequence=cut;publisher.append_target(endpoint);}
    return program_bridge_detail::bounded(storage::encode_native_series(publisher),maximum);
}

class FourUseRoleLiftBridge {
public:
    FourUseRoleLiftBridge(rfc::inference::AvailableRelationalTensorSquare value,rfc::inference::PreparedAvailableRelationalProgram target,
        rfc::inference::RelationalInputAccessionDeclaration accession,std::vector<std::size_t> bounds,std::size_t maximum)
        :square_(std::move(value)),target_(std::move(target)),accession_(std::move(accession)),bounds_(std::move(bounds)),maximum_(maximum){}
    std::vector<std::string> input_payloads() const {
        const auto square=square_.square();const auto parent=square.parent();const auto input=parent.accession();
        return four_use_role_application_detail::payloads(target_.program(),{input.asset(),accession_,parent.use(),square.lifted_field()},maximum_);
    }
    std::shared_ptr<AvailableKnownSquareResultBridge> admit_input(const std::vector<std::uint64_t>& keys,
        std::shared_ptr<AvailabilityContextBridge> context) const {
        const auto authority=known_product_bridge_detail::authority(context);
        const auto value=square_.admit_input(target_,accession_,four_use_role_application_detail::origins(keys),authority,
            program_bridge_detail::limits(bounds_)).execute(authority,program_bridge_detail::limits(bounds_));
        return AvailableKnownSquareResultBridge::adopt(value,bounds_,maximum_);
    }
private:
    rfc::inference::AvailableRelationalTensorSquare square_;
    rfc::inference::PreparedAvailableRelationalProgram target_;
    rfc::inference::RelationalInputAccessionDeclaration accession_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
class FourUseRoleInputBridge {
public:
    FourUseRoleInputBridge(std::shared_ptr<AvailableProgramBridge> parent,std::shared_ptr<AvailableProgramBridge> target,
        std::uint64_t interface_origin,std::shared_ptr<AvailabilityContextBridge> context,std::vector<std::size_t> bounds,std::size_t maximum)
        :parent_(checked(parent)),target_(checked(target)),producer_(parent_,four_use_role_application_detail::square(),
            interface_origin,known_product_bridge_detail::authority(context)),original_(four_use_role_application_detail::accession(parent_.program(),false)),
          derived_(four_use_role_application_detail::accession(target_.program(),true)),bounds_(std::move(bounds)),maximum_(maximum){}
    static std::string interface_payload(std::shared_ptr<FieldProgramBridge> parent,std::size_t maximum) {
        if(!parent)throw std::invalid_argument("actual original program required");
        return program_bridge_detail::bounded(rfc::inference::relational_tensor_square_dependency_payload(
            parent->native_owner(),four_use_role_application_detail::square()),maximum);
    }
    std::vector<std::string> input_payloads(std::shared_ptr<InputUseAssetBridge> asset,const std::string& field,std::uint64_t use) const {
        return four_use_role_application_detail::payloads(parent_.program(),submission(asset,field,use),maximum_);
    }
    std::shared_ptr<FourUseRoleLiftBridge> lift(std::shared_ptr<InputUseAssetBridge> asset,const std::string& field,std::uint64_t use,
        const std::vector<std::uint64_t>& keys,std::shared_ptr<AvailabilityContextBridge> context) const {
        const auto input=submission(asset,field,use);const auto origins=four_use_role_application_detail::origins(keys);
        const auto authority=known_product_bridge_detail::authority(context);
        rfc::learning::require_relational_four_use_grammar_image(parent_,input,origins,authority);
        return std::make_shared<FourUseRoleLiftBridge>(producer_.lift(input,origins,authority,program_bridge_detail::limits(bounds_)),
            target_,derived_,bounds_,maximum_);
    }
private:
    static rfc::inference::PreparedAvailableRelationalProgram checked(const std::shared_ptr<AvailableProgramBridge>& program) {
        if(!program)throw std::invalid_argument("actual four-use available program required");
        return program->native_owner();
    }
    rfc::inference::RelationalInputSubmission submission(std::shared_ptr<InputUseAssetBridge> asset,const std::string& field,std::uint64_t use) const {
        if(!asset)throw std::invalid_argument("actual original input asset required");
        return {asset->native_owner(),original_,use,interface_bridge_detail::field(field,bounds_,maximum_)};
    }
    rfc::inference::PreparedAvailableRelationalProgram parent_,target_;
    rfc::inference::PreparedAvailableRelationalTensorSquare producer_;
    rfc::inference::RelationalInputAccessionDeclaration original_,derived_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
} // namespace rexgraph::python
