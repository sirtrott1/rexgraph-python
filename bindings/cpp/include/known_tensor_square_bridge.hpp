#pragma once

#include "field_admission_bridge.hpp"
#include <rfc/inference/relational_available_tensor_square_codec.hpp>

namespace rexgraph::python {
struct KnownSquareDeclaration {
    std::uint64_t key{}, revision{}, basis{};
    std::size_t grade{}, cell{};
    std::vector<std::uint64_t> pairs;
};
struct KnownSquarePair {
    std::size_t grade{}, cell{};
    std::uint64_t basis{}, left{}, right{}, direction{};
};
namespace known_square_bridge_detail {
inline rfc::inference::RelationalKnownTensorSquareDeclaration declaration(const KnownSquareDeclaration& d) {
    return {d.key, d.revision, d.basis, program_bridge_detail::cell(d.grade, d.cell), d.pairs};
}
} // namespace known_square_bridge_detail

class AvailableKnownSquareBridge;
class AvailableKnownSquareInputBridge;
class AvailableKnownSquareResultBridge;
class AvailableKnownSquareProducerBridge {
public:
    AvailableKnownSquareProducerBridge(std::shared_ptr<AvailableProgramBridge> program,
        const KnownSquareDeclaration& d, std::uint64_t origin,
        std::shared_ptr<AvailabilityContextBridge> context,
        const std::vector<std::size_t>& bounds, std::size_t maximum)
        : bounds_(bounds), maximum_(maximum) {
        if (!program || !context) throw std::invalid_argument("actual available program and current context required");
        owner_ = std::make_shared<const rfc::inference::PreparedAvailableRelationalTensorSquare>(
            program->native_owner(), known_square_bridge_detail::declaration(d), origin, context->native_owner());
    }
    static std::string declaration_payload(std::shared_ptr<FieldProgramBridge> program,
        const KnownSquareDeclaration& d, std::size_t maximum) {
        if (!program) throw std::invalid_argument("actual original program owner required");
        return program_bridge_detail::bounded(rfc::inference::relational_tensor_square_dependency_payload(
            program->native_owner(), known_square_bridge_detail::declaration(d)), maximum);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent current context required");
        owner_->admit(context->native_owner());
    }
    std::shared_ptr<AvailableKnownSquareBridge> lift(const AdmissionInput& input,
        std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    std::shared_ptr<const rfc::inference::PreparedAvailableRelationalTensorSquare> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableKnownSquareBridge {
public:
    static std::shared_ptr<AvailableKnownSquareBridge> adopt(rfc::inference::AvailableRelationalTensorSquare value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableKnownSquareBridge>(new AvailableKnownSquareBridge(std::move(value), bounds, maximum));
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent current context required");
        owner_->admit(context->native_owner());
    }
    std::string field(unsigned kind) const {
        const auto square = owner_->square();
        if (kind == 0) return program_bridge_detail::bounded(program_bridge_detail::bytes(
            program_bridge_detail::tensor_bytes(square.context_field())), maximum_);
        if (kind == 1) return program_bridge_detail::bounded(program_bridge_detail::bytes(
            program_bridge_detail::tensor_bytes(square.tensor_square())), maximum_);
        if (kind == 2) return program_bridge_detail::bounded(program_bridge_detail::bytes(
            program_bridge_detail::tensor_bytes(square.lifted_field())), maximum_);
        throw std::invalid_argument("whole known field kind required");
    }
    std::string metric() const {
        const auto square = owner_->square(); const auto image = program_bridge_detail::matrix_bytes(square.metric());
        if (image.size() > maximum_) throw std::length_error("known product metric byte bound");
        return image;
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_known_tensor_square(owner_->square()), maximum_);
    }
    InterfaceInputInfo parent() const { return interface_bridge_detail::input_info(owner_->square().parent(), maximum_); }
    KnownSquarePair pair(std::size_t index) const {
        const auto p = owner_->square().pair_coordinate(index);
        return {p.context.grade.value, p.context.index, p.original_basis, p.left, p.right, p.lifted_direction};
    }
    void require_image(const std::string& field) const {
        owner_->square().require_image(interface_bridge_detail::field(field, bounds_, maximum_), program_bridge_detail::limits(bounds_));
    }
    std::vector<std::string> input_payloads(std::shared_ptr<FieldProgramBridge> target, const InterfaceAccession& d) const {
        if (!target) throw std::invalid_argument("actual downstream program owner required");
        const auto square = owner_->square(); const auto original = square.parent();
        AdmissionInput row; row.asset = program_bridge_detail::bounded(
            rfc::inference::encode_relational_input_use_asset(original.accession().asset()), maximum_);
        row.field = field(2); row.accession = d; row.use = original.use();
        return AvailableProgramBridge::input_payloads(target, row, bounds_, maximum_);
    }
    std::shared_ptr<AvailableKnownSquareInputBridge> admit_input(std::shared_ptr<AvailableProgramBridge> target,
        const InterfaceAccession& d, const std::vector<std::uint64_t>& origins,
        std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableKnownSquareBridge(rfc::inference::AvailableRelationalTensorSquare value,
        std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::AvailableRelationalTensorSquare>(std::move(value))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::AvailableRelationalTensorSquare> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableKnownSquareInputBridge {
public:
    const rfc::inference::AvailableRelationalTensorSquareInput& native_owner() const & { return *owner_; }
    const rfc::inference::AvailableRelationalTensorSquareInput& native_owner() const && = delete;
    static std::shared_ptr<AvailableKnownSquareInputBridge> adopt(rfc::inference::AvailableRelationalTensorSquareInput value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableKnownSquareInputBridge>(new AvailableKnownSquareInputBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableKnownSquareInputBridge> restore(const std::string& image,
        std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!context) throw std::invalid_argument("independent current context required");
        if (image.size() > maximum) throw std::length_error("known product input byte bound");
        auto value = rfc::inference::decode_available_relational_tensor_square_input(program_bridge_detail::bytes(image),
            context->native_owner(), program_bridge_detail::reader(bounds, maximum), program_bridge_detail::limits(bounds));
        return adopt(std::move(value), bounds, maximum);
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_available_relational_tensor_square_input(*owner_), maximum_);
    }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent current context required");
        owner_->admit(context->native_owner());
    }
    InterfaceInputInfo parent() const {
        return interface_bridge_detail::input_info(owner_->native_input().square().parent(), maximum_);
    }
    InterfaceInputInfo input() const { return interface_bridge_detail::input_info(owner_->native_input().input(), maximum_); }
    std::vector<std::string> dependencies(std::size_t maximum) const {
        if (owner_->obligations().size() > maximum) throw std::length_error("explicit complete product dependency bound");
        std::vector<std::string> out;
        for (const auto& origin : owner_->obligations())
            out.push_back(program_bridge_detail::bounded(origin.receipt(), maximum_));
        return out;
    }
    std::shared_ptr<AvailableKnownSquareResultBridge> execute(std::shared_ptr<AvailabilityContextBridge> context) const;
private:
    AvailableKnownSquareInputBridge(rfc::inference::AvailableRelationalTensorSquareInput value,
        std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::AvailableRelationalTensorSquareInput>(std::move(value))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::AvailableRelationalTensorSquareInput> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class AvailableKnownSquareResultBridge {
public:
    const rfc::inference::AvailableRelationalTensorSquareResult& native_owner() const & { return *owner_; }
    const rfc::inference::AvailableRelationalTensorSquareResult& native_owner() const && = delete;
    static std::shared_ptr<AvailableKnownSquareResultBridge> adopt(rfc::inference::AvailableRelationalTensorSquareResult value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<AvailableKnownSquareResultBridge>(new AvailableKnownSquareResultBridge(std::move(value), bounds, maximum));
    }
    static std::shared_ptr<AvailableKnownSquareResultBridge> restore(const std::string& image,
        std::shared_ptr<AvailabilityContextBridge> context, const std::vector<std::size_t>& bounds, std::size_t maximum) {
        if (!context) throw std::invalid_argument("independent current context required");
        if (image.size() > maximum) throw std::length_error("known product result byte bound");
        auto value = rfc::inference::decode_available_relational_tensor_square_result(program_bridge_detail::bytes(image),
            context->native_owner(), program_bridge_detail::reader(bounds, maximum), program_bridge_detail::limits(bounds));
        return adopt(std::move(value), bounds, maximum);
    }
    std::string archive() const {
        return program_bridge_detail::bounded(rfc::inference::encode_available_relational_tensor_square_result(*owner_), maximum_);
    }
    std::shared_ptr<AvailableKnownSquareInputBridge> input() const {
        return AvailableKnownSquareInputBridge::adopt(owner_->input(), bounds_, maximum_);
    }
    std::shared_ptr<AccessionResultBridge> result() const { return AccessionResultBridge::adopt(owner_->result(), bounds_, maximum_); }
    void admit(std::shared_ptr<AvailabilityContextBridge> context) const {
        if (!context) throw std::invalid_argument("independent current context required");
        owner_->admit(context->native_owner());
    }
private:
    AvailableKnownSquareResultBridge(rfc::inference::AvailableRelationalTensorSquareResult value,
        std::vector<std::size_t> bounds, std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::AvailableRelationalTensorSquareResult>(std::move(value))),
          bounds_(std::move(bounds)), maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::AvailableRelationalTensorSquareResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<AvailableKnownSquareBridge> AvailableKnownSquareProducerBridge::lift(const AdmissionInput& input,
    std::shared_ptr<AvailabilityContextBridge> context) const {
    if (!context) throw std::invalid_argument("independent current context required");
    const auto pin = owner_; const auto frozen = input; const auto authority = context;
    if (frozen.asset.size() > maximum_ || frozen.field.size() > maximum_)
        throw std::length_error("known product original input byte bound");
    pin->admit(authority->native_owner());
    const auto keys = admission_bridge_detail::input_keys(frozen.origins);
    const auto image = program_bridge_detail::bytes(frozen.asset);
    (void)authority->native_owner().require(keys.asset, rfc::inference::RelationalDependencyRole::InputAsset,
        {image.begin(), image.end()});
    const auto asset = InputUseAssetBridge::restore(frozen.asset, bounds_, maximum_);
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
    auto d = interface_bridge_detail::accession_declaration(frozen.accession, guard);
    auto field = interface_bridge_detail::field(frozen.field, bounds_, maximum_);
    auto value = pin->lift({asset->native_owner(), std::move(d), frozen.use, std::move(field)}, keys,
        authority->native_owner(), program_bridge_detail::limits(bounds_));
    return AvailableKnownSquareBridge::adopt(std::move(value), bounds_, maximum_);
}
inline std::shared_ptr<AvailableKnownSquareInputBridge> AvailableKnownSquareBridge::admit_input(
    std::shared_ptr<AvailableProgramBridge> target, const InterfaceAccession& d, const std::vector<std::uint64_t>& origins,
    std::shared_ptr<AvailabilityContextBridge> context) const {
    if (!target || !context) throw std::invalid_argument("actual downstream program and current context required");
    owner_->admit(context->native_owner()); target->native_owner().admit(context->native_owner());
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_));
    auto value = owner_->admit_input(target->native_owner(), interface_bridge_detail::accession_declaration(d, guard),
        admission_bridge_detail::input_keys(origins), context->native_owner(), program_bridge_detail::limits(bounds_));
    return AvailableKnownSquareInputBridge::adopt(std::move(value), bounds_, maximum_);
}
inline std::shared_ptr<AvailableKnownSquareResultBridge> AvailableKnownSquareInputBridge::execute(
    std::shared_ptr<AvailabilityContextBridge> context) const {
    if (!context) throw std::invalid_argument("independent current context required");
    return AvailableKnownSquareResultBridge::adopt(owner_->execute(context->native_owner(),
        program_bridge_detail::limits(bounds_)), bounds_, maximum_);
}
} // namespace rexgraph::python
