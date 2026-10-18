#pragma once

#include "known_tensor_graph_bridge.hpp"
#include "field_family_bridge.hpp"
#include <rfc/learning/relational_literal_field_family_codec.hpp>

namespace rexgraph::python {
struct LiteralFamilyInput {
    std::shared_ptr<LiteralInputBridge> literal;
    std::shared_ptr<KnownGraphInputBridge> graph;
};
class LiteralJointEvaluationBridge;
class LiteralJointResultBridge {
public:
    LiteralJointResultBridge(std::shared_ptr<FieldFamilyBridge> family,std::size_t member,
        const std::vector<LiteralFamilyInput>& inputs,const std::vector<std::size_t>& bounds,std::size_t maximum,GraphCheck check)
        : bounds_(bounds),maximum_(maximum) {
        if (!family) throw std::invalid_argument("actual prepared field family owner required");
        if (inputs.size()>bounds.at(0)) throw std::length_error("literal family input count admission");
        std::vector<rfc::learning::RelationalLiteralFamilyInput> values;
        for (const auto& value : inputs) {
            if (bool(value.literal)==bool(value.graph)) throw std::invalid_argument("one actual literal or graph input required");
            if (value.literal) values.emplace_back(value.literal->native_owner());
            else values.emplace_back(value.graph->native_owner());
        }
        owner_=std::make_shared<const rfc::learning::RelationalLiteralJointResult>(family->native_owner(),member,
            std::move(values),program_bridge_detail::limits(bounds),check.callback());
    }
    static std::shared_ptr<LiteralJointResultBridge> adopt(rfc::learning::RelationalLiteralJointResult value,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<LiteralJointResultBridge>(new LiteralJointResultBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<LiteralJointResultBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,
        std::size_t maximum,GraphCheck check) {
        return adopt(rfc::learning::decode_relational_literal_joint_result(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds),check.callback()),bounds,maximum);
    }
    std::string archive(GraphCheck check) const {
        return program_bridge_detail::bounded(rfc::learning::encode_relational_literal_joint_result(*owner_,maximum_,check.callback()),maximum_);
    }
    std::shared_ptr<JointResultBridge> raw() const { return JointResultBridge::adopt(owner_->raw_result(),bounds_,maximum_); }
    std::shared_ptr<FieldFamilyBridge> family() const { return FieldFamilyBridge::adopt(owner_->family(),bounds_,maximum_); }
    std::size_t member() const { return owner_->member(); }
    std::vector<LiteralFamilyInput> inputs(std::size_t maximum) const {
        if (owner_->inputs().size()>maximum) throw std::length_error("complete literal family input record bound");
        std::vector<LiteralFamilyInput> out;
        for (const auto& value : owner_->inputs()) {
            if (const auto* literal=std::get_if<rfc::inference::OwnedRelationalLiteralInput>(&value))
                out.push_back({LiteralInputBridge::adopt(*literal,bounds_,maximum_),{}});
            else out.push_back({{},KnownGraphInputBridge::adopt(std::get<rfc::inference::RelationalKnownTensorGraphInput>(value),bounds_,maximum_)});
        }
        return out;
    }
    std::shared_ptr<LiteralJointEvaluationBridge> evaluate(const std::vector<std::string>& values,GraphCheck check) const;
private:
    LiteralJointResultBridge(rfc::learning::RelationalLiteralJointResult value,std::vector<std::size_t> bounds,std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalLiteralJointResult>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalLiteralJointResult> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
class LiteralJointEvaluationBridge {
public:
    static std::shared_ptr<LiteralJointEvaluationBridge> adopt(rfc::learning::RelationalLiteralJointEvaluation value,
        const std::vector<std::size_t>& bounds,std::size_t maximum) {
        return std::shared_ptr<LiteralJointEvaluationBridge>(new LiteralJointEvaluationBridge(std::move(value),bounds,maximum));
    }
    static std::shared_ptr<LiteralJointEvaluationBridge> restore(const std::string& image,const std::vector<std::size_t>& bounds,
        std::size_t maximum,GraphCheck check) {
        return adopt(rfc::learning::decode_relational_literal_joint_evaluation(program_bridge_detail::bytes(image),
            program_bridge_detail::reader(bounds,maximum),program_bridge_detail::limits(bounds),check.callback()),bounds,maximum);
    }
    std::string archive(GraphCheck check) const {
        return program_bridge_detail::bounded(rfc::learning::encode_relational_literal_joint_evaluation(*owner_,maximum_,check.callback()),maximum_);
    }
    std::shared_ptr<LiteralJointResultBridge> parent() const { return LiteralJointResultBridge::adopt(owner_->parent(),bounds_,maximum_); }
    std::shared_ptr<JointEvaluationBridge> raw() const { return JointEvaluationBridge::adopt(owner_->raw_evaluation(),bounds_,maximum_); }
    std::shared_ptr<AccessionResultBridge> result() const { return AccessionResultBridge::adopt(owner_->result(),bounds_,maximum_); }
private:
    LiteralJointEvaluationBridge(rfc::learning::RelationalLiteralJointEvaluation value,std::vector<std::size_t> bounds,std::size_t maximum)
        : owner_(std::make_shared<const rfc::learning::RelationalLiteralJointEvaluation>(std::move(value))),bounds_(std::move(bounds)),maximum_(maximum) {}
    std::shared_ptr<const rfc::learning::RelationalLiteralJointEvaluation> owner_;
    std::vector<std::size_t> bounds_;
    std::size_t maximum_{};
};
inline std::shared_ptr<LiteralJointEvaluationBridge> LiteralJointResultBridge::evaluate(const std::vector<std::string>& values,GraphCheck check) const {
    rfc::numeric::ExactActionGuard guard(program_bridge_detail::limits(bounds_),check.callback());
    rfc::numeric::RationalVector free;
    guard.dimension(values.size());
    for (const auto& value : values) {
        guard.step(); rfc::numeric::Rational q;
        if (q.set_str(value,16)!=0 || q.get_den()==0) throw std::invalid_argument("exact complete free values required");
        q.canonicalize(); guard.value(q); free.push_back(std::move(q));
    }
    return LiteralJointEvaluationBridge::adopt(owner_->evaluate(std::move(free),program_bridge_detail::limits(bounds_),check.callback()),bounds_,maximum_);
}
} // namespace rexgraph::python
