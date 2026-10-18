#pragma once
#include <rfc/local/owned_fiber_system.hpp>
#include <rfc/local/prepared_field_action.hpp>
#include <rfc/numeric/checked_exact_actions.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <string>

namespace rexgraph::python {
struct StalkMapEntry { std::size_t row{}, column{}; std::string value; };
struct StalkFieldResult {
    std::vector<std::vector<std::string>> response, boundary, occurrences;
    std::size_t steps{};
};
struct StalkInjectionResult {
    std::vector<std::string> field, boundary;
    std::size_t steps{}, columns{}, occurrences{};
};

class StalkFieldResponse {
    using Q = rfc::numeric::Rational;
    using Vector = rfc::numeric::RationalVector;
    using Matrix = rfc::numeric::ExactMatrix;
    using Guard = rfc::numeric::ExactActionGuard;
    using Pin = std::shared_ptr<const rfc::storage::TemporalEndpoint>;
    struct State {
        Pin endpoint;
        rfc::local::OwnedLinearFiberSystem fibers;
        rfc::local::PreparedGradeFieldAction action;
        rfc::numeric::PreparedExactSolve solve;
        std::vector<Matrix> maps;
        std::vector<std::size_t> upper_offsets, lower_offsets;
        rfc::numeric::ExactActionLimits limits;
        Q lambda;
        std::size_t preparation_steps{}, retained_entries{};
        State(Pin p, rfc::local::OwnedLinearFiberSystem f,
              rfc::local::PreparedGradeFieldAction a, rfc::numeric::PreparedExactSolve s,
              std::vector<Matrix> m, std::vector<std::size_t> u, std::vector<std::size_t> l,
              rfc::numeric::ExactActionLimits b, Q q, std::size_t steps, std::size_t entries)
            : endpoint(std::move(p)), fibers(std::move(f)), action(std::move(a)), solve(std::move(s)),
              maps(std::move(m)), upper_offsets(std::move(u)), lower_offsets(std::move(l)),
              limits(b), lambda(std::move(q)), preparation_steps(steps), retained_entries(entries) {}
    };
    std::shared_ptr<const State> state_;
    static Q scalar(const std::string& text, Guard& guard) {
        Q q;
        if (q.set_str(text, 16) != 0 || q.get_den() == 0)
            throw std::invalid_argument("invalid exact stalk coefficient");
        q.canonicalize(); guard.value(q); return q;
    }
    static std::vector<std::string> strings(const Vector& values) {
        std::vector<std::string> out; out.reserve(values.size());
        for (const auto& q : values) out.push_back(q.get_str(16));
        return out;
    }
    static Pin decode(const std::string& archive, std::size_t max_bytes) {
        rfc::storage::NativeSeriesLimits limits; limits.max_archive_bytes = max_bytes;
        auto history = rfc::storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()), archive.size()), limits);
        return history.tail().pin();
    }
    static void require_grade_one(const rfc::core::RelationalComplex& source) {
        if (source.grade_count() < 2) throw std::invalid_argument("stalk field requires grade one");
        for (std::size_t k = 2; k < source.grade_count(); ++k)
            if (source.grade(rfc::core::Grade{static_cast<std::uint16_t>(k)}).cell_count())
                throw std::invalid_argument("stalk field profile has no higher differential");
    }
public:
    StalkFieldResponse(const std::string& archive,
                      const std::vector<std::vector<std::string>>& lower_axes,
                      const std::vector<std::vector<std::string>>& upper_axes,
                      const std::vector<std::vector<StalkMapEntry>>& declarations,
                      const std::string& parameter, const std::vector<std::size_t>& bounds,
                      std::size_t max_bytes) {
        using namespace rfc;
        if (bounds.size() != 4 || std::any_of(bounds.begin(), bounds.end(), [](auto n){ return !n; }))
            throw std::invalid_argument("four positive stalk action bounds required");
        numeric::ExactActionLimits limits{bounds[0], bounds[1], bounds[2], bounds[3]}; Guard guard(limits);
        auto endpoint = decode(archive, max_bytes); const auto& source = endpoint->source;
        require_grade_one(source); const auto& arena = source.grade(core::Grade{1});
        if (lower_axes.size() != source.grade(core::Grade{0}).cell_count() ||
            upper_axes.size() != arena.cell_count() || declarations.size() != arena.incidence_count())
            throw std::invalid_argument("complete original stalk and occurrence declarations required");
        local::LinearStalkBundle bundle(source); std::vector<std::size_t> lower{0}, upper{0};
        const auto axes = [&](const auto& names, auto grade, auto& offsets) {
            for (std::size_t i = 0; i < names.size(); ++i) {
                guard.step(); const auto& row = names[i];
                if (row.empty() || std::set<std::string>(row.begin(), row.end()).size() != row.size() ||
                    std::any_of(row.begin(), row.end(), [](const auto& s){ return s.empty(); }))
                    throw std::invalid_argument("distinct nonempty axes required for every stalk");
                bundle.set_dimension({grade, static_cast<std::uint32_t>(i)}, row.size());
                offsets.push_back(machine::checked_size_add(offsets.back(), row.size())); guard.dimension(offsets.back());
            }
        };
        axes(lower_axes, core::Grade{0}, lower); axes(upper_axes, core::Grade{1}, upper);
        const auto dimension = machine::checked_size_add(lower.back(), upper.back()); guard.dimension(dimension);
        std::size_t entries = machine::checked_size_add(dimension,
            machine::checked_size_mul(arena.incidence_count(),
                machine::checked_size_add(source.signature().aspects().size(), std::size_t{8})));
        guard.entries(entries);
        local::LinearRelationalFiberSystem fibers(std::move(bundle)); std::vector<Matrix> maps;
        for (std::size_t slot = 0; slot < declarations.size(); ++slot) {
            guard.step(); const auto lo = arena.lower_index[slot], up = arena.upper_index[slot];
            Matrix map(lower_axes[lo].size(), upper_axes[up].size()); std::set<std::pair<std::size_t, std::size_t>> seen;
            entries = machine::checked_size_add(entries, declarations[slot].size()); guard.entries(entries);
            const auto original_coefficient = source.incidence_coefficient(core::Grade{1}, slot);
            guard.value(original_coefficient);
            const auto coefficient = arena.aspects.get(source.signature().existence_aspect(), slot)
                ? original_coefficient : Q{0};
            for (const auto& entry : declarations[slot]) {
                guard.step();
                if (entry.row >= map.rows() || entry.column >= map.cols() || !seen.emplace(entry.row, entry.column).second)
                    throw std::invalid_argument("distinct flag coordinates within original stalk dimensions required");
                map.set(entry.row, entry.column, guard.mul(coefficient, scalar(entry.value, guard)));
            }
            fibers.set_transport(core::Grade{1}, slot, map); maps.push_back(std::move(map));
        }
        local::OwnedLinearFiberSystem owned(fibers);
        auto action = local::PreparedGradeFieldAction::factored(owned.fibers(), core::Grade{1});
        const auto lambda = scalar(parameter, guard);
        if (lambda <= 0) throw std::invalid_argument("positive rational stalk Green parameter required");
        Matrix augmented(dimension, dimension);
        for (std::size_t i = 0; i < dimension; ++i) { guard.step(); augmented.set(i, i, Q{1}); }
        for (const auto& entry : action.entries()) {
            guard.step(); const auto row = upper.back() + entry.target;
            augmented.set(entry.source, row, guard.add(augmented.get(entry.source, row), guard.mul(lambda, entry.coefficient)));
            augmented.set(row, entry.source, guard.sub(augmented.get(row, entry.source), entry.coefficient));
        }
        numeric::PreparedExactSolve solve(std::move(augmented), guard);
        entries = machine::checked_size_add(machine::checked_size_mul(entries, 4), solve.stored_entries()); guard.entries(entries);
        state_ = std::make_shared<const State>(std::move(endpoint), std::move(owned), std::move(action),
            std::move(solve), std::move(maps), std::move(upper), std::move(lower), limits, lambda, guard.steps(), entries);
    }
    std::vector<std::size_t> upper_offsets() const { return state_->upper_offsets; }
    std::vector<std::size_t> lower_offsets() const { return state_->lower_offsets; }
    std::size_t preparation_steps() const { return state_->preparation_steps; }
    std::size_t retained_entries() const { return state_->retained_entries; }
    std::vector<std::size_t> occurrence_lower() const {
        const auto& rows = state_->endpoint->source.grade(rfc::core::Grade{1}).lower_index;
        return {rows.begin(), rows.end()};
    }
    std::vector<std::size_t> occurrence_upper() const {
        const auto& rows = state_->endpoint->source.grade(rfc::core::Grade{1}).upper_index;
        return {rows.begin(), rows.end()};
    }
    std::vector<std::string> occurrence_coefficients() const {
        Vector values;
        const auto& source = state_->endpoint->source;
        for (std::size_t slot = 0; slot < state_->maps.size(); ++slot)
            values.push_back(source.incidence_coefficient(rfc::core::Grade{1}, slot));
        return strings(values);
    }
    std::vector<std::size_t> occurrence_activity() const {
        std::vector<std::size_t> values;
        const auto& source = state_->endpoint->source;
        const auto& arena = source.grade(rfc::core::Grade{1});
        for (std::size_t slot = 0; slot < state_->maps.size(); ++slot)
            values.push_back(arena.aspects.get(source.signature().existence_aspect(), slot) ? 1 : 0);
        return values;
    }
    StalkFieldResult apply(const std::vector<std::vector<std::string>>& fields) const {
        using namespace rfc; const auto pin = state_; Guard guard(pin->limits); StalkFieldResult result;
        guard.dimension(fields.size()); guard.entries(pin->retained_entries);
        const auto n = pin->upper_offsets.back(), m = pin->lower_offsets.back();
        std::size_t occurrence_size{};
        for (const auto& map : pin->maps) occurrence_size = machine::checked_size_add(occurrence_size, map.rows());
        guard.entries(machine::checked_size_add(pin->retained_entries, machine::checked_size_mul(fields.size(),
            machine::checked_size_add(machine::checked_size_add(machine::checked_size_mul(n, 3), m), occurrence_size))));
        std::vector<std::size_t> rows(m); for (std::size_t i = 0; i < m; ++i) rows[i] = i;
        const auto& arena = pin->endpoint->source.grade(core::Grade{1});
        for (const auto& field : fields) {
            if (field.size() != n) throw std::invalid_argument("complete original upper stalk field required");
            Vector input; for (const auto& word : field) input.push_back(scalar(word, guard));
            Vector rhs(n + m); std::copy(input.begin(), input.end(), rhs.begin());
            const auto solved = pin->solve.apply(rhs, guard); Vector response(solved.begin(), solved.begin() + n);
            const auto boundary = pin->action.apply_selected(response, rows, pin->limits.integer_bits, [&guard]{ guard.step(); });
            if (!std::equal(boundary.begin(), boundary.end(), solved.begin() + n))
                throw std::logic_error("stalk boundary differs from its native solve auxiliary");
            Vector residual = response;
            for (const auto& entry : pin->action.entries()) {
                guard.step(); residual[entry.source] = guard.add(residual[entry.source],
                    guard.mul(pin->lambda, guard.mul(entry.coefficient, boundary[entry.target])));
            }
            if (residual != input) throw std::logic_error("stalk Green equation has nonzero exact residual");
            Vector occurrences, assembled(m);
            for (std::size_t slot = 0; slot < pin->maps.size(); ++slot) {
                guard.step(); const auto up = arena.upper_index[slot], lo = arena.lower_index[slot];
                const Vector local(response.begin() + pin->upper_offsets[up], response.begin() + pin->upper_offsets[up + 1]);
                const auto contribution = numeric::checked_apply(pin->maps[slot], local, guard);
                occurrences.insert(occurrences.end(), contribution.begin(), contribution.end());
                for (std::size_t i = 0; i < contribution.size(); ++i)
                    assembled[pin->lower_offsets[lo] + i] = guard.add(assembled[pin->lower_offsets[lo] + i], contribution[i]);
            }
            if (assembled != boundary) throw std::logic_error("original occurrence vectors do not assemble to the boundary");
            result.response.push_back(strings(response)); result.boundary.push_back(strings(boundary));
            result.occurrences.push_back(strings(occurrences));
        }
        result.steps = guard.steps(); return result;
    }
    StalkInjectionResult inject(const std::string& archive, const std::vector<std::size_t>& mapping,
                               const std::vector<std::string>& amplitudes, std::size_t max_bytes) const {
        using namespace rfc; const auto pin = state_; Guard guard(pin->limits); guard.entries(pin->retained_entries);
        auto source = decode(archive, max_bytes); require_grade_one(source->source);
        const auto n = source->source.grade(core::Grade{0}).cell_count();
        const auto& arena = source->source.grade(core::Grade{1});
        if (mapping.size() != n || amplitudes.size() != arena.cell_count())
            throw std::invalid_argument("complete acquisition injection and primary amplitudes required");
        guard.dimension(machine::checked_size_add(n, arena.cell_count()));
        guard.entries(machine::checked_size_add(pin->retained_entries,
            machine::checked_size_add(machine::checked_size_mul(n, 4), machine::checked_size_mul(arena.incidence_count(), 4))));
        std::set<std::size_t> seen;
        for (const auto row : mapping) {
            guard.step(); if (row >= pin->upper_offsets.back() || !seen.insert(row).second)
                throw std::invalid_argument("declared acquisition injection must name distinct actual stalk coordinates");
        }
        local::LinearRelationalFiberSystem fibers{local::LinearStalkBundle(source->source)};
        auto action = local::PreparedGradeFieldAction::factored(fibers, core::Grade{1});
        Vector u; for (const auto& word : amplitudes) u.push_back(scalar(word, guard));
        std::vector<std::size_t> rows(n); for (std::size_t i = 0; i < n; ++i) rows[i] = i;
        const auto boundary = action.apply_selected(u, rows, pin->limits.integer_bits, [&guard]{ guard.step(); });
        Vector field(pin->upper_offsets.back());
        for (std::size_t i = 0; i < n; ++i) { guard.step(); field[mapping[i]] = boundary[i]; }
        return {strings(field), strings(boundary), guard.steps(), arena.cell_count(), arena.incidence_count()};
    }
};
} // namespace rexgraph::python
