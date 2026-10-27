#pragma once

#include "rfc/storage/native_series_codec.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace rexgraph::python {
struct SourceAspectValue {
    std::uint16_t grade{};
    std::uint64_t occurrence{};
    bool value{};
};
struct SourceAspectAddition {
    std::string name;
    unsigned role{};
    bool builder_default{}, has_fill{}, fill{};
    std::vector<SourceAspectValue> assignments;
};
struct SourceTypeAddition {
    std::string name;
    std::size_t minimum{}, maximum{};
};

class SourceEvolutionBridge {
    rfc::storage::NativeTemporalSeries history_;
    std::vector<std::size_t> bounds_;
    std::string archive_;

    static auto bytes(const std::string& data) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(data.data()), data.size());
    }
    static rfc::storage::NativeSeriesLimits policy(const std::vector<std::size_t>& values) {
        if (values.size() != 7 || std::ranges::any_of(values, [](auto n) { return n == 0; }))
            throw std::invalid_argument("source evolution requires seven positive resource limits");
        rfc::storage::NativeSeriesLimits out;
        out.max_archive_bytes = values[0];
        out.temporal.max_records = values[1];
        out.temporal.max_dimension = values[2];
        out.temporal.max_logical_size = values[3];
        out.max_segments = values[4];
        out.temporal.source_declarations.max_occurrences = values[5];
        out.temporal.source_declarations.max_aspect_bytes = values[6];
        return out;
    }
    SourceEvolutionBridge(rfc::storage::NativeTemporalSeries history, std::vector<std::size_t> bounds)
        : history_(std::move(history)), bounds_(std::move(bounds)) {
        const auto encoded = rfc::storage::encode_native_series(history_);
        archive_.assign(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    }
public:
    static std::shared_ptr<SourceEvolutionBridge> decode(const std::string& archive,
                                                         const std::vector<std::size_t>& bounds) {
        auto history = rfc::storage::decode_native_series(bytes(archive), policy(bounds));
        auto result = std::shared_ptr<SourceEvolutionBridge>(new SourceEvolutionBridge(std::move(history), bounds));
        if (result->archive_ != archive)
            throw std::invalid_argument("source evolution requires the canonical complete native series");
        return result;
    }
    std::string archive() const { return archive_; }
    std::vector<std::size_t> limits() const { return bounds_; }
    std::vector<std::uint64_t> identity() const {
        return {history_.source_namespace(), history_.first_sequence(), history_.last_sequence(), history_.segment_count()};
    }
    std::shared_ptr<SourceEvolutionBridge> append(const std::string& frame) const {
        if (frame.empty() || frame.size() > bounds_[0])
            throw std::length_error("source evolution frame exceeds byte admission");
        auto next = history_;
        next.append_encoded(bytes(frame));
        return std::shared_ptr<SourceEvolutionBridge>(new SourceEvolutionBridge(std::move(next), bounds_));
    }
    std::shared_ptr<SourceEvolutionBridge> add_declarations(
            const std::vector<SourceAspectAddition>& aspects,
            const std::vector<SourceTypeAddition>& types) const {
        using namespace rfc;
        if (aspects.empty() && types.empty())
            throw std::invalid_argument("source declaration additions cannot be empty");
        if (aspects.size() > bounds_[1] || types.size() > bounds_[1])
            throw std::length_error("source declaration count exceeds admission");
        const auto pin = history_.tail();
        const auto& base = pin.endpoint();
        auto target_signature = base.source.signature();
        storage::SourceDeclarationMigration migration{base.source.signature(), target_signature, {}, {}, {}};
        for (std::size_t i = 0; i < base.source.signature().aspects().size(); ++i) {
            const auto& aspect = base.source.signature().aspects().at(static_cast<composite::AspectId>(i));
            migration.aspects.push_back({aspect.name, aspect.name});
        }
        for (std::size_t i = 0; i < base.source.signature().type_count(); ++i) {
            const auto& type = base.source.signature().type(static_cast<core::RelationTypeId>(i));
            migration.types.push_back({type.name, type.name});
        }
        for (const auto& aspect : aspects) {
            if (aspect.role < 2 || aspect.role > 8)
                throw std::invalid_argument("source additions cannot replace signed incidence aspects");
            target_signature.aspects().add(aspect.name, static_cast<composite::AspectRole>(aspect.role), aspect.builder_default);
            storage::SourceAspectIntroduction introduction;
            introduction.target = aspect.name;
            if (aspect.has_fill) introduction.fill = aspect.fill;
            for (const auto& assignment : aspect.assignments)
                introduction.assignments.push_back({{core::Grade{assignment.grade}, core::OccurrenceKey{assignment.occurrence}}, assignment.value});
            migration.introduced.push_back(std::move(introduction));
        }
        for (const auto& type : types)
            target_signature.add_type(type.name, core::BoundaryPolicy::DeclaredExact, type.minimum, type.maximum);
        migration.target_signature = std::move(target_signature);
        auto target = base;
        if (base.sequence == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("source sequence is exhausted");
        target.sequence = base.sequence + 1;
        target.source = storage::recode_temporal_source(base.source, migration,
                                                       policy(bounds_).temporal.source_declarations);
        auto next = history_;
        next.append_target(target, {}, {}, storage::RelationalCodeEncoding::Automatic, std::move(migration));
        return std::shared_ptr<SourceEvolutionBridge>(new SourceEvolutionBridge(std::move(next), bounds_));
    }
};
} // namespace rexgraph::python
