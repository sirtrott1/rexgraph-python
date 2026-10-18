#pragma once
#include <rfc/field/relational_channels.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <string>

namespace rexgraph::python {
struct ScalarChannelResult {
    std::vector<std::vector<std::string>> fields;
    std::vector<std::string> energies;
    std::size_t steps{};
};
class ScalarChannels {
    using Q = rfc::numeric::Rational;
    using V = rfc::numeric::RationalVector;
    std::shared_ptr<const rfc::storage::TemporalEndpoint> endpoint_;
    std::shared_ptr<const rfc::field::PreparedRelationalChannels> channels_;
    rfc::numeric::ExactActionLimits limits_;
    static std::vector<std::string> strings(const V& values) {
        std::vector<std::string> out; out.reserve(values.size());
        for (const auto& q : values) out.push_back(q.get_str(16));
        return out;
    }
    V values(const std::vector<std::string>& text) const {
        rfc::numeric::ExactActionGuard guard(limits_);
        guard.dimension(text.size()); guard.entries(text.size());
        V out; out.reserve(text.size());
        for (const auto& item : text) {
            guard.step(); Q q;
            if (q.set_str(item,16) != 0 || q.get_den() == 0)
                throw std::invalid_argument("invalid exact relational scalar");
            q.canonicalize(); guard.value(q); out.push_back(std::move(q));
        }
        return out;
    }
public:
    ScalarChannels(const std::string& archive, const std::vector<std::string>& weights,
        bool count, const std::vector<std::size_t>& bounds, std::size_t max_bytes) {
        using namespace rfc;
        if (bounds.size() != 4) throw std::invalid_argument("four relational action limits required");
        limits_ = {bounds[0],bounds[1],bounds[2],bounds[3]};
        storage::NativeSeriesLimits archive_limits; archive_limits.max_archive_bytes = max_bytes;
        auto history = storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()),archive.size()),archive_limits);
        endpoint_ = history.tail().pin();
        auto source = std::shared_ptr<const core::RelationalComplex>(endpoint_, &endpoint_->source);
        channels_ = std::make_shared<const field::PreparedRelationalChannels>(std::move(source), values(weights),
            count ? field::CoparticipationReading::Count : field::CoparticipationReading::Share, limits_);
    }
    std::size_t relations() const { return channels_->size(); }
    std::size_t boundaries() const { return channels_->lower_size(); }
    std::size_t occurrences() const { return channels_->source().grade(rfc::core::Grade{1}).incidence_count(); }
    std::size_t retained_entries() const { return channels_->retained_entries(); }
    std::size_t preparation_steps() const { return channels_->preparation_steps(); }
    std::vector<std::string> incidence() const { return strings(channels_->realized_incidence()); }
    std::vector<std::vector<std::size_t>> occurrence_cells() const {
        const auto& arena = channels_->source().grade(rfc::core::Grade{1});
        std::vector<std::vector<std::size_t>> out; out.reserve(arena.incidence_count());
        for (std::size_t p = 0; p < arena.incidence_count(); ++p)
            out.push_back({arena.upper_index[p],arena.lower_index[p]});
        return out;
    }
    std::vector<std::uint64_t> occurrence_keys() const {
        const auto& arena = channels_->source().grade(rfc::core::Grade{1});
        std::vector<std::uint64_t> out; out.reserve(arena.incidence_count());
        for (const auto& key : arena.occurrence_identities) out.push_back(key.value);
        return out;
    }
    std::vector<std::size_t> occurrence_activity() const {
        const auto& source = channels_->source(); const auto& arena = source.grade(rfc::core::Grade{1});
        std::vector<std::size_t> out; out.reserve(arena.incidence_count());
        for (std::size_t p = 0; p < arena.incidence_count(); ++p)
            out.push_back(arena.aspects.get(source.signature().existence_aspect(),p) ? 1 : 0);
        return out;
    }
    std::vector<std::string> traces() const {
        return strings(V(channels_->traces().begin(),channels_->traces().end()));
    }
    std::vector<std::vector<std::string>> diagonals() const {
        std::vector<std::vector<std::string>> out;
        for (const auto& field : channels_->diagonals()) out.push_back(strings(field));
        return out;
    }
    ScalarChannelResult apply(const std::vector<std::string>& input, bool normalized) const {
        const auto native = channels_->apply_all(values(input),normalized);
        ScalarChannelResult out; out.steps = native.steps;
        for (const auto& field : native.fields) out.fields.push_back(strings(field));
        out.energies = strings(V(native.energies.begin(),native.energies.end()));
        return out;
    }
};
} // namespace rexgraph::python
