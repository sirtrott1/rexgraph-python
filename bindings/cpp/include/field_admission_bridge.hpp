#pragma once

#include "field_interface_bridge.hpp"
#include <rfc/inference/relational_value_admission_codec.hpp>

namespace rexgraph::python {
struct AdmissionOriginInfo {
    std::uint64_t key{}, name_space{}, first_sequence{}, sequence{};
    unsigned role{};
    std::size_t payload_bytes{};
};
class ValueOriginBridge {
public:
    ValueOriginBridge(std::uint64_t key, const std::string& history, std::uint64_t cutoff,
            unsigned role, const std::string& payload, const std::vector<std::size_t>& bounds,
            std::size_t maximum, const std::vector<std::size_t>& admission)
        : maximum_(maximum) {
        using namespace program_bridge_detail;
        const auto policy = reader(bounds, maximum);
        rfc::storage::NativeSeriesLimits source_limits; source_limits.temporal = policy;
        source_limits.max_archive_bytes = maximum;
        auto source = rfc::storage::decode_native_series(bytes(history), source_limits);
        const auto content = bytes(payload);
        owner_ = std::make_shared<const rfc::inference::OwnedRelationalValueOrigin>(key,
            rfc::storage::NativeSeriesPrefix::capture(source, cutoff),
            static_cast<rfc::inference::RelationalDependencyRole>(role),
            std::vector<std::byte>(content.begin(), content.end()), admission_limits(admission));
    }
    static rfc::inference::RelationalAdmissionLimits admission_limits(const std::vector<std::size_t>& bounds) {
        if (bounds.size() != 3 || std::any_of(bounds.begin(), bounds.end(), [](auto n) { return !n; }))
            throw std::invalid_argument("three positive native origin admission bounds required");
        return {bounds[0], bounds[1], bounds[2]};
    }
    const rfc::inference::OwnedRelationalValueOrigin& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalValueOrigin& native_owner() const && = delete;
    AdmissionOriginInfo info() const {
        const auto source = owner_->publication();
        return {owner_->key(), source.source_namespace(), source.first_sequence(), source.cutoff(),
            static_cast<unsigned>(owner_->role()), owner_->payload_bytes().size()};
    }
    std::string payload() const { return program_bridge_detail::bounded(owner_->payload(), maximum_); }
    std::string receipt() const { return program_bridge_detail::bounded(owner_->receipt(), maximum_); }
    std::string source_archive() const {
        return program_bridge_detail::bounded(rfc::storage::encode_native_series(owner_->publication().history()), maximum_);
    }
    bool same_owner(const ValueOriginBridge& other) const { return owner_->same_owner(*other.owner_); }
    static std::shared_ptr<ValueOriginBridge> adopt(rfc::inference::OwnedRelationalValueOrigin value,
            std::size_t maximum) {
        return std::shared_ptr<ValueOriginBridge>(new ValueOriginBridge(std::move(value), maximum));
    }
private:
    ValueOriginBridge(rfc::inference::OwnedRelationalValueOrigin value, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalValueOrigin>(std::move(value))), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalValueOrigin> owner_;
    std::size_t maximum_{};
};
struct AdmissionGrant {
    std::shared_ptr<ValueOriginBridge> origin;
    std::uint64_t name_space{}, sequence{};
};
class AvailabilityContextBridge {
public:
    AvailabilityContextBridge(const std::vector<AdmissionGrant>& grants, const std::vector<std::size_t>& bounds) {
        std::vector<rfc::inference::RelationalOriginGrant> native;
        for (const auto& grant : grants) {
            if (!grant.origin) throw std::invalid_argument("actual value origin owner required");
            native.push_back({grant.origin->native_owner(), {grant.name_space, grant.sequence}});
        }
        owner_ = std::make_shared<const rfc::inference::RelationalAvailabilityContext>(
            std::move(native), ValueOriginBridge::admission_limits(bounds));
    }
    const rfc::inference::RelationalAvailabilityContext& native_owner() const & { return *owner_; }
    const rfc::inference::RelationalAvailabilityContext& native_owner() const && = delete;
    std::shared_ptr<ValueOriginBridge> require(std::uint64_t key, unsigned role,
            const std::string& payload, std::size_t maximum) const {
        if (payload.size() > maximum) throw std::length_error("native origin payload byte bound");
        return ValueOriginBridge::adopt(owner_->require_bytes(key,
            static_cast<rfc::inference::RelationalDependencyRole>(role), program_bridge_detail::bytes(payload)), maximum);
    }
    void recheck(std::shared_ptr<ValueOriginBridge> origin) const {
        if (!origin) throw std::invalid_argument("actual native expected origin required");
        owner_->recheck(origin->native_owner());
    }
private:
    std::shared_ptr<const rfc::inference::RelationalAvailabilityContext> owner_;
};
struct AdmissionInput {
    std::string asset, field;
    InterfaceAccession accession;
    std::uint64_t use{};
    std::vector<std::uint64_t> origins;
};
namespace admission_bridge_detail {
inline rfc::inference::RelationalProgramOrigins program_keys(const std::vector<std::uint64_t>& keys) {
    if (keys.size() != 3) throw std::invalid_argument("complete program, maps and metrics origin keys required");
    return {keys[0], keys[1], keys[2]};
}
inline rfc::inference::RelationalInputOrigins input_keys(const std::vector<std::uint64_t>& keys) {
    if (keys.size() != 4) throw std::invalid_argument("complete asset, value, accession and reconstruction origin keys required");
    return {keys[0], keys[1], keys[2], keys[3]};
}
inline std::vector<std::string> dependencies(const std::vector<rfc::inference::OwnedRelationalValueOrigin>& values,
        std::size_t maximum) {
    std::vector<std::string> out;
    for (const auto& origin : values)
        out.push_back(program_bridge_detail::bounded(rfc::storage::encode_native_series(origin.publication().history()), maximum));
    return out;
}
} // namespace admission_bridge_detail
class AvailableResultBridge;
class AvailableProgramBridge {
public:
    static std::shared_ptr<AvailableProgramBridge> adopt(rfc::inference::PreparedAvailableRelationalProgram owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableProgramBridge>(new AvailableProgramBridge(std::move(owner), bounds, maximum));
    }
    const rfc::inference::PreparedAvailableRelationalProgram& native_owner() const & { return *owner_; }
    const rfc::inference::PreparedAvailableRelationalProgram& native_owner() const && = delete;
    AvailableProgramBridge(std::shared_ptr<FieldProgramBridge> program, const std::vector<std::uint64_t>& origins,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum)
        : bounds_(bounds), maximum_(maximum) {
        if (!program || !context) throw std::invalid_argument("actual program and independent context owners required");
        owner_ = std::make_shared<const rfc::inference::PreparedAvailableRelationalProgram>(program->native_owner(),
            admission_bridge_detail::program_keys(origins), context->native_owner());
    }
    static std::shared_ptr<AvailableProgramBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!context) throw std::invalid_argument("independent context owner required");
        using namespace program_bridge_detail;
        auto owner = rfc::inference::decode_available_relational_program(bytes(image), context->native_owner(),
            reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<AvailableProgramBridge>(new AvailableProgramBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_available_relational_program(*owner_), maximum_);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent context owner required");
        owner_->admit(context->native_owner());
    }
    std::shared_ptr<FieldProgramBridge> program() const {
        return FieldProgramBridge::adopt(owner_->program(), bounds_, maximum_);
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        if (owner_->obligations().size() > maximum) throw std::length_error("bounded native program origin dependencies");
        return admission_bridge_detail::dependencies(owner_->obligations(), maximum_);
    }
    static std::vector<std::string> input_payloads(std::shared_ptr<FieldProgramBridge> program,
            const AdmissionInput& input, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!program) throw std::invalid_argument("actual native program owner required");
        using namespace program_bridge_detail;
        rfc::numeric::ExactActionGuard guard(limits(bounds));
        const auto d = interface_bridge_detail::accession_declaration(input.accession, guard);
        const auto field = interface_bridge_detail::field(input.field, bounds, maximum);
        const auto asset = bytes(input.asset); const auto parent = rfc::inference::encode_relational_field_program(program->native_owner());
        const auto value = rfc::inference::relational_input_value_payload(asset, d, input.use, field);
        const auto accession = rfc::inference::relational_input_accession_payload(parent, asset, d, input.use, field);
        return {bounded(asset, maximum), bounded(value, maximum), bounded(accession, maximum), bounded(accession, maximum)};
    }
    std::shared_ptr<AvailableResultBridge> execute(const std::vector<AdmissionInput>& inputs,
        std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableProgramBridge(rfc::inference::PreparedAvailableRelationalProgram owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::PreparedAvailableRelationalProgram>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::PreparedAvailableRelationalProgram> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableResultBridge {
public:
    const rfc::inference::AvailableRelationalFieldResult& native_owner() const & { return *owner_; }
    const rfc::inference::AvailableRelationalFieldResult& native_owner() const && = delete;
    static std::shared_ptr<AvailableResultBridge> restore(const std::string& image,
            std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!context) throw std::invalid_argument("independent context owner required");
        using namespace program_bridge_detail;
        auto owner = rfc::inference::decode_available_relational_result(bytes(image), context->native_owner(),
            reader(bounds, maximum), limits(bounds));
        return std::shared_ptr<AvailableResultBridge>(new AvailableResultBridge(std::move(owner), bounds, maximum));
    }
    static std::shared_ptr<AvailableResultBridge> adopt(rfc::inference::AvailableRelationalFieldResult owner,
            const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableResultBridge>(new AvailableResultBridge(std::move(owner), bounds, maximum));
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_available_relational_result(*owner_), maximum_);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent context owner required");
        owner_->admit(context->native_owner());
    }
    std::shared_ptr<AccessionResultBridge> result() const {
        return AccessionResultBridge::adopt(owner_->result(), bounds_, maximum_);
    }
    std::string program_archive(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent context owner required");
        owner_->admit(context->native_owner());
        const auto result = owner_->result();
        const rfc::inference::PreparedAvailableRelationalProgram program(result.result().program(),
            owner_->program_origins(), context->native_owner());
        return program_bridge_detail::bounded(rfc::inference::encode_available_relational_program(program), maximum_);
    }
    std::vector<std::string> dependency_archives(std::size_t maximum) const {
        if (owner_->obligations().size() > maximum) throw std::length_error("bounded native result origin dependencies");
        return admission_bridge_detail::dependencies(owner_->obligations(), maximum_);
    }
private:
    AvailableResultBridge(rfc::inference::AvailableRelationalFieldResult owner,
            std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::AvailableRelationalFieldResult>(std::move(owner))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::AvailableRelationalFieldResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AvailableResultBridge> AvailableProgramBridge::execute(const std::vector<AdmissionInput>& inputs,
        std::shared_ptr<AvailabilityContextBridge> context) const {
    if (!context) throw std::invalid_argument("independent context owner required");
    const auto pin = owner_; const auto authority = context; const auto frozen = inputs;
    using namespace program_bridge_detail;
    pin->admit(authority->native_owner());
    std::vector<rfc::inference::RelationalAvailableInput> native;
    for (const auto& input : frozen) {
        const auto asset_bytes = bytes(input.asset); const auto keys = admission_bridge_detail::input_keys(input.origins);
        (void)authority->native_owner().require(keys.asset, rfc::inference::RelationalDependencyRole::InputAsset,
            std::vector<std::byte>(asset_bytes.begin(), asset_bytes.end()));
        const auto asset = InputUseAssetBridge::restore(input.asset, bounds_, maximum_);
        rfc::numeric::ExactActionGuard guard(limits(bounds_));
        const auto d = interface_bridge_detail::accession_declaration(input.accession, guard);
        const auto field = interface_bridge_detail::field(input.field, bounds_, maximum_);
        native.push_back(pin->admit_input({asset->native_owner(), d, input.use, field}, keys,
            authority->native_owner(), limits(bounds_)));
    }
    return AvailableResultBridge::adopt(pin->execute(std::move(native), authority->native_owner(), limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
