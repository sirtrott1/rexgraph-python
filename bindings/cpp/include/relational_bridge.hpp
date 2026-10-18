#pragma once
#include <rfc/local/owned_fiber_system.hpp>
#include <rfc/local/prepared_field_action.hpp>
#include <rfc/numeric/checked_exact_actions.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <set>
#include <map>
#include <string>

namespace rexgraph::python {
struct RelationResponseResult {
    std::vector<std::string> gram, response, boundary, contributions, membership_contributions, role_contributions;
    std::size_t fields{}, steps{};
};

// Each lower stalk has a shared membership component and retained role axes.
// Each occurrence sends its scalar through its declared membership and role gains.
// Prepared augmented equations use original occurrence maps.
class RelationResponse {
    using Q=rfc::numeric::Rational;
    using Vector=rfc::numeric::RationalVector;
    using Matrix=rfc::numeric::ExactMatrix;
    struct State {
        std::shared_ptr<const rfc::storage::TemporalEndpoint> endpoint;
        rfc::local::OwnedLinearFiberSystem fibers;
        rfc::local::PreparedGradeFieldAction action;
        rfc::numeric::PreparedExactSolve solve;
        rfc::numeric::ExactActionLimits limits;
        std::vector<std::size_t> occurrence_axes,offsets;
        Vector membership_coefficients,role_coefficients;
        Q lambda;
        std::size_t upper{},lower{},preparation_steps{};
        State(std::shared_ptr<const rfc::storage::TemporalEndpoint> source,
              rfc::local::OwnedLinearFiberSystem system,
              rfc::local::PreparedGradeFieldAction prepared,
              rfc::numeric::PreparedExactSolve factor,
              rfc::numeric::ExactActionLimits bounds,std::vector<std::size_t> axes,
              std::vector<std::size_t> lower_offsets,Vector membership,Vector roles,Q parameter,std::size_t steps)
            :endpoint(std::move(source)),fibers(std::move(system)),action(std::move(prepared)),
             solve(std::move(factor)),limits(bounds),occurrence_axes(std::move(axes)),
             offsets(std::move(lower_offsets)),membership_coefficients(std::move(membership)),
             role_coefficients(std::move(roles)),lambda(std::move(parameter)),
             upper(action.statistics().source_dimension),lower(action.statistics().target_dimension),
             preparation_steps(steps) {}
    };
    std::shared_ptr<const State> state_;
    static Q scalar(const std::string& text) {
        Q out;
        if(out.set_str(text,16)!=0 || out.get_den()==0)throw std::invalid_argument("invalid exact relation field");
        out.canonicalize();return out;
    }
    static std::vector<std::string> strings(const Vector& values) {
        std::vector<std::string> result;result.reserve(values.size());
        for(const auto& value:values)result.push_back(value.get_str(16));
        return result;
    }
public:
    RelationResponse(const std::string& archive,const std::vector<std::vector<std::string>>& lower_axes,
                     const std::vector<std::size_t>& occurrence_axes,const std::string& parameter,
                     const std::vector<std::size_t>& bounds,std::size_t max_bytes,
                     const std::vector<std::vector<std::string>>& role_gains={}) {
        using namespace rfc;
        if(bounds.size()!=4)throw std::invalid_argument("relation response requires four action limits");
        numeric::ExactActionLimits limits{bounds[0],bounds[1],bounds[2],bounds[3]};
        numeric::ExactActionGuard guard(limits);
        storage::NativeSeriesLimits archive_limits;archive_limits.max_archive_bytes=max_bytes;
        auto history=storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()),archive.size()),archive_limits);
        auto endpoint=history.tail().pin();const auto& source=endpoint->source;
        if(source.grade_count()<2 || source.grade(core::Grade{1}).cell_count()==0)
            throw std::invalid_argument("relation response requires nonempty primary relations");
        for(std::size_t k=2;k<source.grade_count();++k)
            if(source.grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count()!=0)
                throw std::invalid_argument("this relation response profile has no higher stalk differential");
        const auto& arena=source.grade(core::Grade{1});
        const auto n=arena.cell_count();
        std::map<std::string,std::pair<Q,Q>> gains;
        guard.entries(machine::checked_size_mul(role_gains.size(),std::size_t{2}));
        for(const auto& row:role_gains) {
            guard.step();
            if(row.size()!=3 || row[0]=="membership")throw std::invalid_argument("named role and two exact gains required");
            const auto id=source.signature().aspects().id(row[0]);
            if(source.signature().aspects().at(id).role!=composite::AspectRole::RelationRole)
                throw std::invalid_argument("transport gain must name a retained role");
            const auto membership=scalar(row[1]),role=scalar(row[2]);
            guard.value(membership);guard.value(role);
            if(membership<0 || role<0)throw std::invalid_argument("nonnegative role gains required");
            if(!gains.emplace(row[0],std::pair{membership,role}).second)
                throw std::invalid_argument("distinct role gains required");
        }
        if(lower_axes.size()!=source.grade(core::Grade{0}).cell_count() || occurrence_axes.size()!=arena.incidence_count())
            throw std::invalid_argument("complete original stalk and occurrence axes required");
        local::LinearStalkBundle bundle(source);
        std::vector<std::size_t> offsets{0};
        for(std::size_t i=0;i<lower_axes.size();++i) {
            const auto& names=lower_axes[i];
            if(names.size()<2 || names.front()!="membership" || std::set<std::string>(names.begin(),names.end()).size()!=names.size())
                throw std::invalid_argument("stalk requires membership followed by distinct role axes");
            for(std::size_t j=1;j<names.size();++j)(void)source.signature().aspects().id(names[j]);
            bundle.set_dimension({core::Grade{0},static_cast<std::uint32_t>(i)},names.size());
            offsets.push_back(machine::checked_size_add(offsets.back(),names.size()));
            guard.dimension(offsets.back());
        }
        guard.dimension(machine::checked_size_add(n,offsets.back()));
        guard.entries(machine::checked_size_add(arena.incidence_count(),lower_axes.size()));
        local::LinearRelationalFiberSystem fibers(std::move(bundle));
        Vector membership_coefficients,role_coefficients;
        for(std::size_t slot=0;slot<arena.incidence_count();++slot) {
            guard.step();const auto lower=arena.lower_index[slot];const auto row=occurrence_axes[slot];
            if(row==0 || row>=lower_axes[lower].size())throw std::invalid_argument("occurrence axis outside original role stalk");
            const auto aspect=source.signature().aspects().id(lower_axes[lower][row]);
            if(source.signature().aspects().at(aspect).role!=composite::AspectRole::RelationRole || !arena.aspects.get(aspect,slot))
                throw std::invalid_argument("occurrence axis is not its retained role aspect");
            std::size_t role_count{};
            for(const auto id:source.signature().aspects().aspects_with_role(composite::AspectRole::RelationRole)) {
                guard.step();if(arena.aspects.get(id,slot))++role_count;
            }
            if(role_count!=1)throw std::invalid_argument("one declared transport role required per occurrence");
            Matrix map(lower_axes[lower].size(),1);
            const auto coefficient=source.incidence_coefficient(core::Grade{1},slot);guard.value(coefficient);
            const auto gain=gains.find(lower_axes[lower][row]);
            const auto membership=guard.mul(coefficient,gain==gains.end()?Q{1}:gain->second.first);
            const auto role=guard.mul(coefficient,gain==gains.end()?Q{1}:gain->second.second);
            map.set(0,0,membership);map.set(row,0,role);
            membership_coefficients.push_back(membership);role_coefficients.push_back(role);
            fibers.set_transport(core::Grade{1},slot,std::move(map));
        }
        local::OwnedLinearFiberSystem owned(fibers);
        auto action=local::PreparedGradeFieldAction::factored(owned.fibers(),core::Grade{1});
        const auto lambda=scalar(parameter);guard.value(lambda);
        if(lambda<=0)throw std::invalid_argument("relation Green parameter must be positive");
        const auto dimension=machine::checked_size_add(n,offsets.back());Matrix augmented(dimension,dimension);
        for(std::size_t i=0;i<dimension;++i)augmented.set(i,i,Q{1});
        for(const auto& entry:action.entries()) {
            const auto lower=n+entry.target;
            augmented.set(entry.source,lower,guard.add(augmented.get(entry.source,lower),guard.mul(lambda,entry.coefficient)));
            augmented.set(lower,entry.source,guard.sub(augmented.get(lower,entry.source),entry.coefficient));
        }
        numeric::PreparedExactSolve solve(std::move(augmented),guard);
        state_=std::make_shared<const State>(std::move(endpoint),std::move(owned),std::move(action),
            std::move(solve),limits,occurrence_axes,std::move(offsets),std::move(membership_coefficients),
            std::move(role_coefficients),lambda,guard.steps());
    }
    std::size_t relations() const {return state_->upper;}
    std::size_t lower_dimension() const {return state_->lower;}
    std::size_t occurrences() const {return state_->occurrence_axes.size();}
    std::size_t preparation_steps() const {return state_->preparation_steps;}
    std::size_t factor_entries() const {return state_->solve.stored_entries();}
    RelationResponseResult moments(const std::vector<std::vector<std::string>>& fields) const {
        using namespace rfc;const auto pin=state_;numeric::ExactActionGuard guard(pin->limits);
        if(fields.empty())throw std::invalid_argument("at least one relation field required");
        guard.dimension(fields.size());guard.entries(machine::checked_size_mul(fields.size(),pin->upper));
        std::vector<Vector> inputs,responses;Vector first_boundary;
        std::vector<std::size_t> rows(pin->lower);
        for(std::size_t i=0;i<rows.size();++i){guard.step();rows[i]=i;}
        for(const auto& encoded:fields) {
            if(encoded.size()!=pin->upper)throw std::invalid_argument("field must cover every retained primary relation");
            Vector input;input.reserve(encoded.size());for(const auto& value:encoded)input.push_back(scalar(value));
            guard.vector(input);Vector rhs(pin->upper+pin->lower);std::copy(input.begin(),input.end(),rhs.begin());
            const auto solved=pin->solve.apply(rhs,guard);
            Vector response(solved.begin(),solved.begin()+pin->upper);
            const auto boundary=pin->action.apply_selected(response,rows,pin->limits.integer_bits,[&guard]{guard.step();});
            guard.vector(boundary);
            if(!std::equal(boundary.begin(),boundary.end(),solved.begin()+pin->upper))
                throw std::logic_error("native boundary differs from the Green auxiliary");
            if(inputs.empty())first_boundary=boundary;
            inputs.push_back(std::move(input));responses.push_back(std::move(response));
        }
        RelationResponseResult result;result.fields=fields.size();
        Vector gram;guard.entries(machine::checked_size_mul(fields.size(),fields.size()));
        for(const auto& input:inputs)for(const auto& response:responses) {
            Q value{};for(std::size_t i=0;i<input.size();++i)value=guard.add(value,guard.mul(input[i],response[i]));
            gram.push_back(std::move(value));
        }
        result.gram=strings(gram);result.response=strings(responses.front());
        result.boundary=strings(first_boundary);
        const auto& arena=pin->endpoint->source.grade(core::Grade{1});Vector contributions;
        Vector membership,roles;
        for(std::size_t slot=0;slot<arena.incidence_count();++slot) {
            contributions.push_back(guard.mul(pin->endpoint->source.incidence_coefficient(core::Grade{1},slot),responses.front()[arena.upper_index[slot]]));
            membership.push_back(guard.mul(pin->membership_coefficients[slot],responses.front()[arena.upper_index[slot]]));
            roles.push_back(guard.mul(pin->role_coefficients[slot],responses.front()[arena.upper_index[slot]]));
        }
        result.contributions=strings(contributions);result.membership_contributions=strings(membership);
        result.role_contributions=strings(roles);result.steps=guard.steps();return result;
    }
};
}
