#pragma once
#include <rfc/learning/relational_four_use_grammar_codec.hpp>

#include "known_tensor_square_bridge.hpp"
#include "field_evidence_admission_bridge.hpp"
#include <rfc/learning/relational_available_tensor_square_evidence_codec.hpp>
#include <rfc/learning/relational_four_use_moment_basis.hpp>

namespace rexgraph::python {
struct MomentPair {
    std::size_t relation{}, head{}, dependent{}, row0{}, row1{};
    std::uint64_t head_flag{}, dependent_flag{};
};
namespace known_product_bridge_detail {
using namespace rfc;
using namespace program_bridge_detail;
using available_evidence_bridge_detail::authority;
inline storage::SharedPacketLimits sharing(const std::vector<std::size_t>& value) {
    if (value.size()!=4 || std::any_of(value.begin(),value.end(),[](auto n){return !n;}))
        throw std::invalid_argument("four positive native shared byte bounds required");
    return {value[0],value[1],value[2],value[3]};
}
inline storage::TemporalReadLimits policy(const std::vector<std::size_t>& bounds,std::size_t expanded) {
    (void)limits(bounds); storage::TemporalReadLimits out;
    out.max_dimension=bounds.at(0); out.max_file_bytes=expanded; return out;
}
inline inference::RelationalInitialInputLineageDeclaration lineage_declaration(std::uint64_t key,std::uint64_t revision,
    const std::vector<std::vector<std::uint64_t>>& uses) {
    inference::RelationalInitialInputLineageDeclaration out{key,revision,{}};
    for(const auto& pair:uses){if(pair.size()!=2)throw std::invalid_argument("complete ordered original and initial use keys required");out.uses.push_back({pair[0],pair[1]});}
    return out;
}
inline inference::OwnedRelationalInputUseAsset initial_asset(const AvailableKnownSquareResultBridge& input) {
    const auto compound=input.native_owner().input(); const auto native=compound.native_input();
    const auto square=native.square();return square.parent().accession().asset();
}
inline void grammar(const inference::AvailableRelationalTensorSquareResult& input,
    const inference::RelationalAvailabilityContext& context) {
    input.admit(context); const auto compound=input.input(); const auto native=compound.native_input();
    const auto square=native.square(); const auto parent=square.parent(); const auto a=parent.accession();
    inference::PreparedAvailableRelationalProgram program(a.program(),compound.origins().parent_program,context);
    learning::require_relational_four_use_grammar_image(program,{a.asset(),a.declaration(),parent.use(),parent.source_field()},compound.origins().parent_input,context);
}
inline std::vector<std::string> sources(const std::vector<inference::OwnedRelationalValueOrigin>& origins,
    std::size_t count,std::size_t maximum) {
    evidence_bridge_detail::bound(origins.size(),count);return admission_bridge_detail::dependencies(origins,maximum);
}
} // namespace known_product_bridge_detail

class InitialInputLineageBridge {
public:
    const rfc::inference::AvailableRelationalInitialInputLineage& native_owner() const & {return *owner_;}
    const rfc::inference::AvailableRelationalInitialInputLineage& native_owner() const && = delete;
    static std::string payload(std::shared_ptr<InputUseAssetBridge> original,std::shared_ptr<AvailableKnownSquareResultBridge> input,
        std::uint64_t key,std::uint64_t revision,const std::vector<std::vector<std::uint64_t>>& uses,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace known_product_bridge_detail;
        if(!original||!input)throw std::invalid_argument("actual original asset and product input required");
        const inference::OwnedRelationalInitialInputLineage owner(original->native_owner(),initial_asset(*input),lineage_declaration(key,revision,uses),limits(bounds));
        return bounded(inference::relational_initial_input_lineage_payload(owner),maximum);
    }
    InitialInputLineageBridge(std::shared_ptr<InputUseAssetBridge> original,std::shared_ptr<AvailableKnownSquareResultBridge> input,
        std::uint64_t key,std::uint64_t revision,const std::vector<std::vector<std::uint64_t>>& uses,
        const std::vector<std::uint64_t>& origins,std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds,std::size_t maximum):bounds_(bounds),maximum_(maximum) {
        using namespace known_product_bridge_detail;
        if(!original||!input||origins.size()!=3)throw std::invalid_argument("actual assets and three original lineage origins required");
        const auto actual=authority(context);input->native_owner().admit(actual);
        const auto old_bytes=inference::encode_relational_input_use_asset(original->native_owner());
        (void)actual.require(origins[0],inference::RelationalDependencyRole::InputAsset,old_bytes);
        const inference::OwnedRelationalInitialInputLineage owner(original->native_owner(),initial_asset(*input),lineage_declaration(key,revision,uses),limits(bounds));
        owner_=std::make_shared<const inference::AvailableRelationalInitialInputLineage>(owner,
            inference::RelationalInitialInputLineageOrigins{origins[0],origins[1],origins[2]},actual);
    }
    static std::shared_ptr<InitialInputLineageBridge> adopt(rfc::inference::AvailableRelationalInitialInputLineage value,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<InitialInputLineageBridge>(new InitialInputLineageBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<InitialInputLineageBridge> restore(const std::string& image,std::shared_ptr<AvailableKnownSquareResultBridge> input,
        std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace known_product_bridge_detail;
        if(!input)throw std::invalid_argument("actual checked product input required for lineage restoration");
        if(image.size()>maximum)throw std::length_error("initial lineage byte cap");
        const auto actual=authority(context);input->native_owner().admit(actual);
        return adopt(inference::decode_available_relational_initial_input_lineage(bytes(image),actual,policy(bounds,maximum),limits(bounds),{},initial_asset(*input)),bounds,maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(known_product_bridge_detail::authority(context));}
    std::string archive() const {return program_bridge_detail::bounded(rfc::inference::encode_available_relational_initial_input_lineage(*owner_),maximum_);}
    std::string asset(bool original) const {
        const auto value=owner_->lineage();return program_bridge_detail::bounded(rfc::inference::encode_relational_input_use_asset(original?value.original():value.initial()),maximum_);
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        return known_product_bridge_detail::sources(owner_->obligations(),maximum,maximum_);
    }
private:
    InitialInputLineageBridge(rfc::inference::AvailableRelationalInitialInputLineage value,std::vector<std::size_t> bounds,std::size_t maximum)
        :owner_(std::make_shared<const rfc::inference::AvailableRelationalInitialInputLineage>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum){}
    std::shared_ptr<const rfc::inference::AvailableRelationalInitialInputLineage> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
class FourUseMomentBasisBridge {
public:
    FourUseMomentBasisBridge(std::shared_ptr<FieldProgramBridge> program,std::shared_ptr<AvailableKnownSquareResultBridge> reference,
        std::uint64_t key,std::uint64_t revision,std::uint64_t node,const std::vector<std::uint64_t>& parameters,
        const std::vector<MomentPair>& pairs,const std::vector<std::size_t>& bounds,std::size_t maximum):bounds_(bounds),maximum_(maximum) {
        using namespace known_product_bridge_detail;
        if(!program||!reference)throw std::invalid_argument("actual program and complete known reference required");
        std::vector<learning::RelationalMomentRolePair> native;
        for(const auto& p:pairs)native.push_back({cell(1,p.relation),cell(0,p.head).index,
            cell(0,p.dependent).index,p.head_flag,p.dependent_flag,{p.row0,p.row1}});
        const auto compound=reference->native_owner().input();const auto input=compound.native_input();
        owner_=std::make_shared<const learning::RelationalFourUseMomentBasis>(learning::make_relational_four_use_moment_basis(
            program->native_owner(),input.square(),key,revision,node,parameters,std::move(native),limits(bounds)));
    }
    const rfc::learning::RelationalFourUseMomentBasis& native_owner() const & {return *owner_;}
    const rfc::learning::RelationalFourUseMomentBasis& native_owner() const && = delete;
    std::vector<std::size_t> counts() const {return {owner_->parameters.variables.size(),owner_->parameters.parameters.size(),owner_->parameters.basis.nonzeros(),owner_->readings.size()};}
    std::string matrix(std::size_t pair,unsigned kind) const {
        const auto& list=kind==0?owner_->readings:kind==1?owner_->base_reconstructions:owner_->base_metrics;
        if(kind>2||pair>=list.size())throw std::out_of_range("actual moment comparison matrix required");
        const auto value=program_bridge_detail::matrix_bytes(list[pair]);
        if(value.size()>maximum_)throw std::length_error("moment inspection byte bound");
        return value;
    }
private:
    std::shared_ptr<const rfc::learning::RelationalFourUseMomentBasis> owner_;
    std::vector<std::size_t> bounds_;std::size_t maximum_{};
};
struct KnownProductObservation {
    std::uint64_t key{},use{},asset_origin{},value_origin{},interface_origin{},evidence_origin{};
    std::shared_ptr<AvailableKnownSquareResultBridge> input;
    std::shared_ptr<InitialInputLineageBridge> lineage;
    std::shared_ptr<InputUseAssetBridge> target_asset;
    std::string target;
    std::vector<std::vector<std::size_t>> observed;
};
struct KnownProductBatchInfo {std::size_t origin_count{};std::vector<std::string> dependencies;};
class KnownProductFamilyBridge;
struct KnownProductQueryOwners {
    std::shared_ptr<AvailableKnownSquareResultBridge> input;
    std::shared_ptr<InitialInputLineageBridge> lineage;
};
class KnownProductEvidenceBridge {
public:
    using Native=rfc::learning::PreparedAvailableRelationalTensorSquareEvidence;
    KnownProductEvidenceBridge(std::shared_ptr<AvailableProgramBridge> program,const EvidenceDeclaration& d,
        std::uint64_t root,std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,
        std::size_t maximum,const std::vector<std::size_t>& shared):bounds_(bounds),maximum_(maximum),shared_(shared) {
        using namespace known_product_bridge_detail;
        if(!program)throw std::invalid_argument("actual available program required");
        (void)sharing(shared_);
        rfc::numeric::ExactActionGuard guard(limits(bounds));owner_=std::make_shared<const Native>(program->native_owner(),evidence_bridge_detail::declaration(d,guard),root,authority(context),limits(bounds));
    }
    static std::shared_ptr<KnownProductEvidenceBridge> from_basis(std::shared_ptr<AvailableProgramBridge> program,
        std::shared_ptr<FourUseMomentBasisBridge> basis,std::uint64_t root,std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds,std::size_t maximum,const std::vector<std::size_t>& shared) {
        using namespace known_product_bridge_detail;
        if(!program||!basis)throw std::invalid_argument("actual available program and native moment basis required");
        const auto p=program->native_owner();const auto actual_program=p.program();const auto& b=basis->native_owner();
        if(!actual_program.fibers().same_owner(b.program.fibers()))throw std::invalid_argument("moment basis belongs to another actual program");
        return adopt(Native(p,b.parameters,root,authority(context),limits(bounds)),bounds,maximum,shared);
    }
    static std::string basis_root_payload(std::shared_ptr<FourUseMomentBasisBridge> basis,const std::vector<std::size_t>& bounds,std::size_t maximum) {
        using namespace known_product_bridge_detail;if(!basis)throw std::invalid_argument("actual native basis required");const auto& b=basis->native_owner();
        return bounded(learning::relational_evidence_root_payload(inference::encode_relational_field_program(b.program),b.parameters,limits(bounds)),maximum);
    }
    static std::shared_ptr<KnownProductEvidenceBridge> adopt(Native value,const std::vector<std::size_t>& bounds,
        std::size_t maximum,const std::vector<std::size_t>& shared) {
        (void)known_product_bridge_detail::sharing(shared);
        return std::shared_ptr<KnownProductEvidenceBridge>(new KnownProductEvidenceBridge(std::move(value),bounds,maximum,shared));
    }
    static std::shared_ptr<KnownProductEvidenceBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds,std::size_t maximum,const std::vector<std::size_t>& shared,bool teacher=false,const std::string& expected={}) {
        using namespace known_product_bridge_detail;if(image.size()>shared.at(0))throw std::length_error("stored complete evidence byte cap");
        if(teacher&&(expected.empty()||expected.size()>maximum))throw std::invalid_argument("teacher replay requires complete factory-derived root payload");
        auto value=teacher?learning::decode_shared_available_relational_four_use_tensor_square_evidence(bytes(image),authority(context),policy(bounds,shared.at(1)),limits(bounds),sharing(shared),{},bytes(expected)):
            learning::decode_shared_available_relational_tensor_square_evidence(bytes(image),authority(context),policy(bounds,shared.at(1)),limits(bounds),sharing(shared));
        return adopt(std::move(value),bounds,maximum,shared);
    }
    const Native& native_owner() const & {return *owner_;} const Native& native_owner() const &&=delete;
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(known_product_bridge_detail::authority(context));}
    std::string archive() const {
        using namespace known_product_bridge_detail;return bounded(learning::encode_shared_available_relational_tensor_square_evidence(*owner_,sharing(shared_)),shared_.at(0));
    }
    std::string parameter_schema() const {rfc::storage::temporal_codec_detail::Writer writer;const auto base=owner_->base();rfc::learning::relational_affine_codec_detail::parameters(writer,base.root_declaration());return program_bridge_detail::bounded(writer.bytes(),maximum_);}
    std::shared_ptr<FieldEvidenceBridge> evidence() const {return FieldEvidenceBridge::adopt(owner_->evidence(),bounds_,maximum_);}
    std::shared_ptr<AvailableProgramBridge> program() const {return AvailableProgramBridge::adopt(owner_->program(),bounds_,maximum_);}
    std::shared_ptr<KnownProductEvidenceBridge> parent() const {const auto value=owner_->parent();return value?adopt(*value,bounds_,maximum_,shared_):nullptr;}
    std::size_t origin_count() const {return owner_->obligations().size();}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        using namespace known_product_bridge_detail;auto out=sources(owner_->obligations(),maximum,maximum_);
        const auto old=evidence()->dependency_archives(maximum);out.insert(out.end(),old.begin(),old.end());
        auto current=owner_;while(current){for(const auto& o:current->suffix())if(o.lineage){const auto lineage=o.lineage->lineage();
            out.push_back(bounded(storage::encode_native_series(lineage.original().history()),maximum_));
            out.push_back(bounded(storage::encode_native_series(lineage.initial().history()),maximum_));}
            const auto parent=current->parent();current=parent?std::make_shared<const Native>(*parent):nullptr;}
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
    std::string descriptor(const KnownProductObservation& row,std::shared_ptr<AvailabilityContextBridge> context) const {
        using namespace known_product_bridge_detail;const auto actual=authority(context);owner_->admit(actual);
        if(!row.input||!row.target_asset)throw std::invalid_argument("actual complete input and target owners required");
        row.input->native_owner().admit(actual);const auto input=row.input->native_owner().input();auto dependencies=input.obligations();
        if(row.lineage){row.lineage->native_owner().admit(actual);const auto& extra=row.lineage->native_owner().obligations();dependencies.insert(dependencies.end(),extra.begin(),extra.end());}
        const auto asset=inference::encode_relational_input_use_asset(row.target_asset->native_owner());
        const auto base=owner_->base();const auto program=inference::encode_relational_field_program(base.program().program());
        const auto value=interface_bridge_detail::field(row.target,bounds_,maximum_);
        const auto a=actual.require(row.asset_origin,inference::RelationalDependencyRole::TargetAsset,asset);
        const auto b=actual.require(row.value_origin,inference::RelationalDependencyRole::TargetValue,learning::relational_target_value_payload(program,base.root_declaration().node,asset,row.use,value));
        std::vector<core::CellId> mask;for(const auto& c:row.observed){if(c.size()!=2)throw std::invalid_argument("actual observed cell required");mask.push_back(cell(c[0],c[1]));}
        return bounded(learning::relational_tensor_square_observation_payload(row.key,base.root_origin(),dependencies,a,b,row.use,mask),maximum_);
    }
    KnownProductBatchInfo batch_info(const std::vector<KnownProductObservation>& rows,std::shared_ptr<AvailabilityContextBridge> context,std::size_t maximum) const {
        using namespace known_product_bridge_detail;const auto actual=authority(context);owner_->admit(actual);
        std::vector<inference::OwnedRelationalValueOrigin> obligations;std::vector<std::string> assets;
        for(const auto& row:rows){const auto payload=descriptor(row,context);const auto input=row.input->native_owner().input();
            obligations.insert(obligations.end(),input.obligations().begin(),input.obligations().end());
            if(row.lineage){const auto& origins=row.lineage->native_owner().obligations();obligations.insert(obligations.end(),origins.begin(),origins.end());
                const auto lineage=row.lineage->native_owner().lineage();assets.push_back(bounded(storage::encode_native_series(lineage.original().history()),maximum_));
                assets.push_back(bounded(storage::encode_native_series(lineage.initial().history()),maximum_));}
            const auto asset=inference::encode_relational_input_use_asset(row.target_asset->native_owner());
            const auto base=owner_->base();const auto program=inference::encode_relational_field_program(base.program().program());
            const auto target=interface_bridge_detail::field(row.target,bounds_,maximum_);
            obligations.push_back(actual.require(row.asset_origin,inference::RelationalDependencyRole::TargetAsset,asset));
            obligations.push_back(actual.require(row.value_origin,inference::RelationalDependencyRole::TargetValue,learning::relational_target_value_payload(program,base.root_declaration().node,asset,row.use,target)));
            obligations.push_back(actual.require(row.interface_origin,inference::RelationalDependencyRole::Interface,{bytes(payload).begin(),bytes(payload).end()}));
            obligations.push_back(actual.require(row.evidence_origin,inference::RelationalDependencyRole::Evidence,{bytes(payload).begin(),bytes(payload).end()}));
            const auto native=input.native_input();const auto square=native.square();
            assets.push_back(bounded(storage::encode_native_series(square.parent().accession().asset().history()),maximum_));
            assets.push_back(bounded(storage::encode_native_series(row.target_asset->native_owner().history()),maximum_));}
        auto dependencies=sources(obligations,maximum,maximum_);dependencies.insert(dependencies.end(),assets.begin(),assets.end());
        evidence_bridge_detail::bound(dependencies.size(),maximum);return {obligations.size(),std::move(dependencies)};
    }
    std::shared_ptr<KnownProductEvidenceBridge> append(const std::vector<KnownProductObservation>& rows,std::shared_ptr<AvailabilityContextBridge> context,bool teacher) const {
        using namespace known_product_bridge_detail;const auto actual=authority(context);owner_->admit(actual);std::vector<learning::RelationalAvailableTensorSquareObservation> suffix;
        evidence_bridge_detail::bound(rows.size(),bounds_.at(0));
        for(const auto& row:rows){if(!row.input||!row.target_asset)throw std::invalid_argument("actual complete input and target owners required");
            if(teacher){if(!row.lineage)throw std::invalid_argument("teacher role evidence requires actual original source lineage");grammar(row.input->native_owner(),actual);}
            std::vector<core::CellId> mask;for(const auto& c:row.observed){if(c.size()!=2)throw std::invalid_argument("actual observed cell required");mask.push_back(cell(c[0],c[1]));}
            std::optional<inference::AvailableRelationalInitialInputLineage> lineage;if(row.lineage)lineage=row.lineage->native_owner();
            suffix.push_back({row.key,row.input->native_owner(),row.target_asset->native_owner(),row.use,interface_bridge_detail::field(row.target,bounds_,maximum_),mask,
                {row.asset_origin,row.value_origin},row.interface_origin,row.evidence_origin,std::move(lineage)});}
        return adopt(owner_->append(std::move(suffix),actual),bounds_,maximum_,shared_);
    }
    std::shared_ptr<AvailableKnownSquareResultBridge> restore_input(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context) const {
        using namespace known_product_bridge_detail;if(image.size()>shared_.at(0))throw std::length_error("stored query byte cap");
        const auto raw=storage::decode_shared_packet(bytes(image),sharing(shared_));
        const auto value=inference::decode_available_relational_tensor_square_result(raw,authority(context),policy(bounds_,shared_.at(1)),limits(bounds_),{},owner_->program().program());
        return AvailableKnownSquareResultBridge::adopt(value,bounds_,maximum_);
    }
    KnownProductQueryOwners restore_query(const std::string& image,const std::string& lineage,
        std::shared_ptr<AvailabilityContextBridge> context,bool teacher) const {
        using namespace known_product_bridge_detail;
        if(image.size()>shared_.at(0)||lineage.size()>maximum_)throw std::length_error("complete checked query byte cap");
        if(teacher){
            if(lineage.empty())throw std::invalid_argument("teacher query requires complete original lineage");
            const auto value=learning::decode_shared_available_relational_four_use_query(bytes(image),bytes(lineage),authority(context),
                policy(bounds_,shared_.at(1)),limits(bounds_),sharing(shared_),{},owner_->program().program());
            return {AvailableKnownSquareResultBridge::adopt(value.input,bounds_,maximum_),InitialInputLineageBridge::adopt(value.lineage,bounds_,maximum_)};
        }
        if(!lineage.empty())(void)inference::decode_available_relational_initial_input_lineage(bytes(lineage),authority(context),
            policy(bounds_,maximum_),limits(bounds_));
        const auto input=restore_input(image,context);
        return {input,lineage.empty()?nullptr:InitialInputLineageBridge::restore(lineage,input,context,bounds_,maximum_)};
    }
    std::shared_ptr<KnownProductFamilyBridge> execute(std::shared_ptr<AvailableKnownSquareResultBridge> input,
        std::shared_ptr<InitialInputLineageBridge> lineage,std::shared_ptr<AvailabilityContextBridge> context,bool teacher) const;
private:
    KnownProductEvidenceBridge(Native value,std::vector<std::size_t> bounds,std::size_t maximum,std::vector<std::size_t> shared)
        :owner_(std::make_shared<const Native>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum),shared_(std::move(shared)){}
    std::shared_ptr<const Native> owner_;std::vector<std::size_t> bounds_;std::size_t maximum_{};std::vector<std::size_t> shared_;
};
class KnownProductFamilyBridge {
public:
    using Native=rfc::learning::AvailableRelationalTensorSquareAffineResult;
    static std::shared_ptr<KnownProductFamilyBridge> adopt(Native value,const std::vector<std::size_t>& bounds,std::size_t maximum,const std::vector<std::size_t>& shared) {
        return std::shared_ptr<KnownProductFamilyBridge>(new KnownProductFamilyBridge(std::move(value),bounds,maximum,shared));
    }
    static std::shared_ptr<KnownProductFamilyBridge> restore(const std::string& image,std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds,std::size_t maximum,const std::vector<std::size_t>& shared,bool teacher=false,const std::string& expected={}) {
        using namespace known_product_bridge_detail;if(image.size()>shared.at(0))throw std::length_error("stored complete family byte cap");
        if(teacher&&(expected.empty()||expected.size()>maximum))throw std::invalid_argument("teacher replay requires complete factory-derived root payload");
        auto value=teacher?learning::decode_shared_available_relational_four_use_tensor_square_affine_result(bytes(image),authority(context),policy(bounds,shared.at(1)),limits(bounds),sharing(shared),{},bytes(expected)):
            learning::decode_shared_available_relational_tensor_square_affine_result(bytes(image),authority(context),policy(bounds,shared.at(1)),limits(bounds),sharing(shared));
        return adopt(std::move(value),bounds,maximum,shared);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {owner_->admit(known_product_bridge_detail::authority(context));}
    std::string archive() const {using namespace known_product_bridge_detail;return bounded(rfc::learning::encode_shared_available_relational_tensor_square_affine_result(*owner_,sharing(shared_)),shared_.at(0));}
    std::shared_ptr<KnownProductEvidenceBridge> evidence() const {return KnownProductEvidenceBridge::adopt(owner_->evidence(),bounds_,maximum_,shared_);}
    std::shared_ptr<AvailableKnownSquareResultBridge> input() const {return AvailableKnownSquareResultBridge::adopt(owner_->input(),bounds_,maximum_);}
    std::shared_ptr<InitialInputLineageBridge> lineage() const {const auto value=owner_->lineage();return value?InitialInputLineageBridge::adopt(*value,bounds_,maximum_):nullptr;}
    std::shared_ptr<AffineResultBridge> result() const {return AffineResultBridge::adopt(owner_->result(),bounds_,maximum_);}
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        using namespace known_product_bridge_detail;auto out=evidence()->dependency_archives(maximum);const auto compound=owner_->input().input();
        auto extra=sources(compound.obligations(),maximum,maximum_);out.insert(out.end(),extra.begin(),extra.end());
        const auto initial=owner_->lineage();if(initial){auto values=sources(initial->obligations(),maximum,maximum_);out.insert(out.end(),values.begin(),values.end());
            const auto l=initial->lineage();out.push_back(bounded(storage::encode_native_series(l.original().history()),maximum_));out.push_back(bounded(storage::encode_native_series(l.initial().history()),maximum_));}
        evidence_bridge_detail::bound(out.size(),maximum);return out;
    }
private:
    KnownProductFamilyBridge(Native value,std::vector<std::size_t> bounds,std::size_t maximum,std::vector<std::size_t> shared)
        :owner_(std::make_shared<const Native>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum),shared_(std::move(shared)){}
    std::shared_ptr<const Native> owner_;std::vector<std::size_t> bounds_;std::size_t maximum_{};std::vector<std::size_t> shared_;
};
inline std::shared_ptr<KnownProductFamilyBridge> KnownProductEvidenceBridge::execute(std::shared_ptr<AvailableKnownSquareResultBridge> input,
    std::shared_ptr<InitialInputLineageBridge> lineage,std::shared_ptr<AvailabilityContextBridge> context,bool teacher) const {
    using namespace known_product_bridge_detail;const auto actual=authority(context);owner_->admit(actual);
    if(!input)throw std::invalid_argument("actual complete product input required");
    if(teacher){if(!lineage)throw std::invalid_argument("teacher role query requires actual original lineage");grammar(input->native_owner(),actual);}
    std::optional<inference::AvailableRelationalInitialInputLineage> original;if(lineage)original=lineage->native_owner();
    return KnownProductFamilyBridge::adopt(owner_->execute(input->native_owner(),actual,limits(bounds_),{},std::move(original)),bounds_,maximum_,shared_);
}
inline std::string shared_product_input(std::shared_ptr<AvailableKnownSquareResultBridge> input,const std::vector<std::size_t>& shared) {
    using namespace known_product_bridge_detail;if(!input)throw std::invalid_argument("actual complete product required");
    return bounded(rfc::storage::encode_shared_packet(rfc::inference::encode_available_relational_tensor_square_result(input->native_owner()),sharing(shared)),shared.at(0));
}
inline void require_four_use_grammar(std::shared_ptr<AvailableProgramBridge> program,const AdmissionInput& row,
    std::shared_ptr<AvailabilityContextBridge> context,const std::vector<std::size_t>& bounds,std::size_t maximum) {
    using namespace known_product_bridge_detail;if(!program)throw std::invalid_argument("actual parent program required");
    const auto actual=authority(context);const auto submissions=available_evidence_bridge_detail::submissions({row},actual,bounds,maximum);
    rfc::learning::require_relational_four_use_grammar_image(program->native_owner(),submissions.front().input,submissions.front().origins,actual);
}
} // namespace rexgraph::python
