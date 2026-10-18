#pragma once

#include "literal_input_bridge.hpp"
#include <rfc/inference/relational_known_tensor_graph_codec.hpp>

namespace rexgraph::python {
// The caller pins the callback context throughout each synchronous native call.
struct GraphCheck {
    int (*run)(void*) = nullptr;
    void* context{};
    std::function<void()> callback() const {
        if (!run) return {};
        return [pin = *this] {
            if (!pin.run(pin.context)) throw std::runtime_error("native check callback failed");
        };
    }
};
struct GraphBlock {
    std::size_t grade{}, cell{};
    std::uint64_t source_basis{}, lifted_basis{};
    std::vector<std::uint64_t> left, right, products;
};
struct GraphDeclaration {
    std::uint64_t key{}, revision{};
    std::vector<GraphBlock> blocks;
};
namespace graph_bridge_detail {
inline rfc::inference::RelationalKnownTensorGraphDeclaration declaration(const GraphDeclaration& d) {
    rfc::inference::RelationalKnownTensorGraphDeclaration out{d.key,d.revision,{}};
    for (const auto& b : d.blocks)
        out.blocks.push_back({program_bridge_detail::cell(b.grade,b.cell),b.source_basis,b.lifted_basis,b.left,b.right,b.products});
    return out;
}
}
class KnownGraphInputBridge;
class KnownGraphBridge {
public:
    KnownGraphBridge(std::shared_ptr<LiteralInputBridge> literal, std::shared_ptr<AccessionInputBridge> input,
        const GraphDeclaration& d, const std::vector<std::size_t>& bounds, std::size_t maximum, GraphCheck check)
        : bounds_(bounds), maximum_(maximum) {
        using namespace rfc::inference;
        if (bool(literal) == bool(input)) throw std::invalid_argument("one actual literal or source input owner required");
        const auto declaration = graph_bridge_detail::declaration(d);
        if (literal) owner_ = std::make_shared<const OwnedRelationalKnownTensorGraph>(literal->native_owner(),declaration,program_bridge_detail::limits(bounds),check.callback());
        else owner_ = std::make_shared<const OwnedRelationalKnownTensorGraph>(input->native_owner(),declaration,program_bridge_detail::limits(bounds),check.callback());
    }
    static std::shared_ptr<KnownGraphBridge> adopt(rfc::inference::OwnedRelationalKnownTensorGraph value,
        const std::vector<std::size_t>& bounds, std::size_t maximum) {
        return std::shared_ptr<KnownGraphBridge>(new KnownGraphBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<KnownGraphBridge> restore(const std::string& image,
        const std::vector<std::size_t>& bounds, std::size_t maximum, GraphCheck check) {
        return adopt(rfc::inference::decode_relational_known_tensor_graph(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds),check.callback()),bounds,maximum);
    }
    const rfc::inference::OwnedRelationalKnownTensorGraph& native_owner() const & { return *owner_; }
    const rfc::inference::OwnedRelationalKnownTensorGraph& native_owner() const && = delete;
    std::string archive(GraphCheck check) const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_known_tensor_graph(*owner_,maximum_,check.callback()),maximum_);
    }
    GraphDeclaration declaration(std::size_t maximum) const {
        const auto& d = owner_->declaration();
        GraphDeclaration out{d.key,d.revision,{}}; std::size_t count=d.blocks.size();
        if (count>maximum) throw std::length_error("complete graph declaration record bound");
        for (const auto& b : d.blocks) {
            for (const auto n : {b.left.size(),b.right.size(),b.products.size()}) {
                if (n>maximum-count) throw std::length_error("complete graph declaration record bound");
                count+=n;
            }
            out.blocks.push_back({b.cell.grade.value,b.cell.index,b.source_basis,b.lifted_basis,b.left,b.right,b.products});
        }
        return out;
    }
    std::vector<ProgramStalkInfo> stalks(std::size_t maximum) const {
        std::size_t count=owner_->stalks().size();
        if (count>maximum) throw std::length_error("complete graph stalk record bound");
        std::vector<ProgramStalkInfo> out;
        for (const auto& s : owner_->stalks()) {
            if (s.directions.size()>maximum-count) throw std::length_error("complete graph stalk record bound");
            count+=s.directions.size();out.push_back(interface_bridge_detail::stalk_info(s));
        }
        return out;
    }
    std::string field() const { return program_bridge_detail::bounded(program_bridge_detail::bytes(program_bridge_detail::tensor_bytes(owner_->lifted_field())),maximum_); }
    std::string product(std::size_t grade,std::size_t cell) const {
        return program_bridge_detail::bounded(program_bridge_detail::bytes(program_bridge_detail::tensor_bytes(owner_->product(program_bridge_detail::cell(grade,cell)))),maximum_);
    }
    std::string matrix(unsigned kind) const {
        const rfc::numeric::ExactMatrix* value=nullptr;
        if (kind==0) value=&owner_->metric();
        else if (kind==1) value=&owner_->original_metric();
        else if (kind==2) value=&owner_->projection();
        else if (kind==3) value=&owner_->zero_section();
        else throw std::invalid_argument("known graph matrix kind required");
        const auto image=program_bridge_detail::matrix_bytes(*value);
        if (image.size()>maximum_) throw std::length_error("known graph matrix byte bound");
        return image;
    }
    std::size_t retained_entries() const { return owner_->retained_entries(); }
    std::shared_ptr<AccessionInputBridge> parent() const { return AccessionInputBridge::adopt(owner_->parent(),bounds_,maximum_); }
    std::shared_ptr<LiteralInputBridge> literal() const {
        const auto value=owner_->literal_parent(); return value?LiteralInputBridge::adopt(*value,bounds_,maximum_):nullptr;
    }
    bool same_owner(const KnownGraphBridge& other) const { return owner_->same_owner(*other.owner_); }
    void require_image(const std::string& image,GraphCheck check) const {
        owner_->require_image(interface_bridge_detail::field(image,bounds_,maximum_),program_bridge_detail::limits(bounds_),check.callback());
    }
    std::shared_ptr<KnownGraphInputBridge> admit(std::shared_ptr<InputAccessionBridge> target,GraphCheck check) const;
private:
    KnownGraphBridge(rfc::inference::OwnedRelationalKnownTensorGraph value,std::vector<std::size_t> bounds,std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::OwnedRelationalKnownTensorGraph>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::OwnedRelationalKnownTensorGraph> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class KnownGraphInputBridge {
public:
    static std::shared_ptr<KnownGraphInputBridge> adopt(rfc::inference::RelationalKnownTensorGraphInput value,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<KnownGraphInputBridge>(new KnownGraphInputBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<KnownGraphInputBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,
        std::size_t maximum,GraphCheck check) {
        return adopt(rfc::inference::decode_relational_known_tensor_graph_input(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds),check.callback()),bounds,maximum);
    }
    const rfc::inference::RelationalKnownTensorGraphInput& native_owner() const & { return *owner_; }
    const rfc::inference::RelationalKnownTensorGraphInput& native_owner() const && = delete;
    std::string archive(GraphCheck check) const {
        return program_bridge_detail::bounded(rfc::inference::encode_relational_known_tensor_graph_input(*owner_,maximum_,check.callback()),maximum_);
    }
    std::shared_ptr<KnownGraphBridge> graph() const { return KnownGraphBridge::adopt(owner_->graph(),bounds_,maximum_); }
    std::shared_ptr<AccessionInputBridge> input() const { return AccessionInputBridge::adopt(owner_->input(),bounds_,maximum_); }
    std::shared_ptr<FieldProgramBridge> program() const { return FieldProgramBridge::adopt(owner_->input().accession().program(),bounds_,maximum_); }
private:
    KnownGraphInputBridge(rfc::inference::RelationalKnownTensorGraphInput value,std::vector<std::size_t> bounds,std::size_t maximum)
        : owner_(std::make_shared<const rfc::inference::RelationalKnownTensorGraphInput>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::inference::RelationalKnownTensorGraphInput> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<KnownGraphInputBridge> KnownGraphBridge::admit(std::shared_ptr<InputAccessionBridge> target,GraphCheck check) const {
    if (!target) throw std::invalid_argument("actual target accession owner required");
    return KnownGraphInputBridge::adopt(owner_->admit(target->native_owner(),program_bridge_detail::limits(bounds_),check.callback()),bounds_,maximum_);
}
} // namespace rexgraph::python
