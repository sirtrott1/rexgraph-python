#pragma once
#include <rfc/local/owned_fiber_system.hpp>
#include <rfc/local/prepared_field_action.hpp>
#include <rfc/local/tensor_field.hpp>
#include <rfc/numeric/checked_exact_actions.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <algorithm>
#include <optional>
#include <set>
#include <string>

namespace rexgraph::python {
struct BoundaryMapEntry { std::size_t row{}, column{}; std::string value; };
struct BoundaryTrace {
    std::vector<std::vector<std::string>> fields, occurrences;
    std::vector<std::vector<std::vector<std::string>>> channels;
    std::size_t steps{};
};
struct BoundaryInjection {
    std::vector<std::string> field, boundary;
    std::size_t steps{}, columns{}, occurrences{};
};

class StalkTransportDeclaration {
    friend class StalkBoundaryTower;
    using Q = rfc::numeric::Rational;
    using Vector = rfc::numeric::RationalVector;
    using Matrix = rfc::numeric::ExactMatrix;
    using Guard = rfc::numeric::ExactActionGuard;
    using Pin = std::shared_ptr<const rfc::storage::TemporalEndpoint>;
    struct State {
        Pin endpoint;
        rfc::local::OwnedLinearFiberSystem fibers;
        std::vector<rfc::local::PreparedGradeFieldAction> actions;
        std::vector<std::vector<Matrix>> maps;
        std::vector<std::vector<std::size_t>> offsets;
        rfc::numeric::ExactActionLimits limits;
        std::size_t preparation_steps{}, retained_entries{};
        State(Pin p, rfc::local::OwnedLinearFiberSystem f,
              std::vector<rfc::local::PreparedGradeFieldAction> a,
              std::vector<std::vector<Matrix>> m, std::vector<std::vector<std::size_t>> o,
              rfc::numeric::ExactActionLimits b, std::size_t steps, std::size_t entries)
            : endpoint(std::move(p)), fibers(std::move(f)), actions(std::move(a)),
              maps(std::move(m)), offsets(std::move(o)), limits(b),
              preparation_steps(steps), retained_entries(entries) {}
    };
    std::shared_ptr<const State> state_;
    static Q scalar(const std::string& text, Guard& guard) {
        Q q;
        if (q.set_str(text,16)!=0 || q.get_den()==0)
            throw std::invalid_argument("invalid exact boundary coefficient");
        q.canonicalize();guard.value(q);return q;
    }
    static std::vector<std::string> strings(const Vector& values) {
        std::vector<std::string> out;out.reserve(values.size());
        for(const auto& q:values)out.push_back(q.get_str(16));
        return out;
    }
    static Pin decode(const std::string& archive,std::size_t max_bytes) {
        rfc::storage::NativeSeriesLimits limits;limits.max_archive_bytes=max_bytes;
        auto history=rfc::storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()),archive.size()),limits);
        return history.tail().pin();
    }
    static Vector apply(const State& state,std::size_t grade,const Vector& input,Guard& guard) {
        std::vector<std::size_t> rows(state.offsets[grade-1].back());
        for(std::size_t i=0;i<rows.size();++i)rows[i]=i;
        return state.actions[grade-1].apply_selected(input,rows,state.limits.integer_bits,[&]{guard.step();});
    }
    StalkTransportDeclaration(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& declarations,
        const std::vector<std::size_t>& bounds,std::size_t max_bytes,bool strict) {
        using namespace rfc;
        if(bounds.size()!=4 || std::any_of(bounds.begin(),bounds.end(),[](auto n){return !n;}))
            throw std::invalid_argument("four positive boundary action bounds required");
        numeric::ExactActionLimits limits{bounds[0],bounds[1],bounds[2],bounds[3]};Guard guard(limits);
        auto endpoint=decode(archive,max_bytes);const auto& source=endpoint->source;
        if(axes.empty() || axes.size()!=source.grade_count() || declarations.size()!=axes.size() || !declarations[0].empty())
            throw std::invalid_argument("complete original grades and an empty grade-zero flag family required");
        local::LinearStalkBundle bundle(source);std::vector<std::vector<std::size_t>> offsets;
        std::size_t dimension{},entries{};
        for(std::size_t k=0;k<axes.size();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
            if(axes[k].size()!=source.grade(grade).cell_count())
                throw std::invalid_argument("stalks must cover every original cell at each grade");
            std::vector<std::size_t> local{0};
            for(std::size_t i=0;i<axes[k].size();++i) {
                guard.step();const auto& names=axes[k][i];
                if(names.empty() || std::set<std::string>(names.begin(),names.end()).size()!=names.size() ||
                   std::any_of(names.begin(),names.end(),[](const auto& n){return n.empty();}))
                    throw std::invalid_argument("distinct nonempty axes required for every original stalk");
                dimension=machine::checked_size_add(dimension,names.size());guard.dimension(dimension);
                bundle.set_dimension({grade,static_cast<std::uint32_t>(i)},names.size());
                local.push_back(machine::checked_size_add(local.back(),names.size()));
            }
            offsets.push_back(std::move(local));
        }
        entries=dimension;guard.entries(entries);
        local::LinearRelationalFiberSystem fibers(std::move(bundle));
        std::vector<std::vector<Matrix>> maps(axes.size());
        for(std::size_t k=1;k<axes.size();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};const auto& arena=source.grade(grade);
            if(declarations[k].size()!=arena.incidence_count())
                throw std::invalid_argument("one declared block required per original occurrence at each grade");
            for(std::size_t slot=0;slot<arena.incidence_count();++slot) {
                guard.step();const auto lo=arena.lower_index[slot],up=arena.upper_index[slot];
                Matrix map(axes[k-1][lo].size(),axes[k][up].size());
                const auto original=source.incidence_coefficient(grade,slot);guard.value(original);
                const auto coefficient=arena.aspects.get(source.signature().existence_aspect(),slot)?original:Q{0};
                entries=machine::checked_size_add(entries,machine::checked_size_add(declarations[k][slot].size(),8));guard.entries(entries);
                std::set<std::pair<std::size_t,std::size_t>> seen;
                for(const auto& entry:declarations[k][slot]) {
                    guard.step();
                    if(entry.row>=map.rows() || entry.column>=map.cols() || !seen.emplace(entry.row,entry.column).second)
                        throw std::invalid_argument("flag coordinates must lie within distinct original stalk directions");
                    map.set(entry.row,entry.column,guard.mul(coefficient,scalar(entry.value,guard)));
                }
                fibers.set_transport(grade,slot,map);maps[k].push_back(std::move(map));
            }
        }
        std::optional<Matrix> previous;
        if(strict)for(std::size_t k=1;k<axes.size();++k) {
            auto boundary=local::GradeTensorField(fibers,core::Grade{static_cast<std::uint16_t>(k)}).matrix();
            guard.matrix(boundary);
            if(previous && numeric::checked_multiply(*previous,boundary,guard).nonzeros())
                throw std::invalid_argument("declared stalk maps violate the complete strict chain family");
            previous=std::move(boundary);
        }
        local::OwnedLinearFiberSystem owned(fibers);std::vector<local::PreparedGradeFieldAction> actions;
        if(strict)for(std::size_t k=1;k<axes.size();++k) {
            actions.push_back(local::PreparedGradeFieldAction::factored(owned.fibers(),core::Grade{static_cast<std::uint16_t>(k)}));
            entries=machine::checked_size_add(entries,actions.back().statistics().entries);
        }
        entries=machine::checked_size_mul(entries,4);guard.entries(entries);
        state_=std::make_shared<const State>(std::move(endpoint),std::move(owned),std::move(actions),
            std::move(maps),std::move(offsets),limits,guard.steps(),entries);
    }
public:
    StalkTransportDeclaration(const StalkTransportDeclaration&) = default;
    StalkTransportDeclaration(StalkTransportDeclaration&&) = delete;
    StalkTransportDeclaration& operator=(const StalkTransportDeclaration&) = delete;
    StalkTransportDeclaration& operator=(StalkTransportDeclaration&&) = delete;
    StalkTransportDeclaration(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& declarations,
        const std::vector<std::size_t>& bounds,std::size_t max_bytes)
        :StalkTransportDeclaration(archive,axes,declarations,bounds,max_bytes,false) {}
    auto offsets() const {return state_->offsets;}
    rfc::local::OwnedLinearFiberSystem fiber_owner() const {return state_->fibers;}
    std::size_t preparation_steps() const {return state_->preparation_steps;}
    std::size_t retained_entries() const {return state_->retained_entries;}
    std::vector<std::vector<std::size_t>> occurrence_cells(std::size_t grade) const {
        if(!grade || grade>=state_->offsets.size())throw std::out_of_range("positive original occurrence grade required");
        const auto& arena=state_->endpoint->source.grade(rfc::core::Grade{static_cast<std::uint16_t>(grade)});
        std::vector<std::vector<std::size_t>> out;
        for(std::size_t slot=0;slot<arena.incidence_count();++slot)out.push_back({arena.upper_index[slot],arena.lower_index[slot]});
        return out;
    }
    std::vector<std::string> occurrence_coefficients(std::size_t grade) const {
        (void)occurrence_cells(grade);Vector values;
        for(std::size_t slot=0;slot<state_->maps[grade].size();++slot)
            values.push_back(state_->endpoint->source.incidence_coefficient(rfc::core::Grade{static_cast<std::uint16_t>(grade)},slot));
        return strings(values);
    }
    std::vector<std::string> occurrence_keys(std::size_t grade) const {
        (void)occurrence_cells(grade);std::vector<std::string> out;
        for(const auto& key:state_->endpoint->source.grade(rfc::core::Grade{static_cast<std::uint16_t>(grade)}).occurrence_identities)
            out.push_back(std::to_string(key.value));
        return out;
    }
    std::vector<std::size_t> occurrence_activity(std::size_t grade) const {
        (void)occurrence_cells(grade);std::vector<std::size_t> out;const auto& source=state_->endpoint->source;
        const auto& arena=source.grade(rfc::core::Grade{static_cast<std::uint16_t>(grade)});
        for(std::size_t slot=0;slot<arena.incidence_count();++slot)
            out.push_back(arena.aspects.get(source.signature().existence_aspect(),slot));
        return out;
    }
};

class StalkBoundaryTower : public StalkTransportDeclaration {
public:
    StalkBoundaryTower(const std::string& archive,
        const std::vector<std::vector<std::vector<std::string>>>& axes,
        const std::vector<std::vector<std::vector<BoundaryMapEntry>>>& declarations,
        const std::vector<std::size_t>& bounds,std::size_t max_bytes)
        :StalkTransportDeclaration(archive,axes,declarations,bounds,max_bytes,true) {}
    BoundaryTrace execute(std::size_t grade,std::size_t steps,const std::vector<std::string>& values) const {
        using namespace rfc;const auto pin=state_;Guard guard(pin->limits);
        if(grade>=pin->offsets.size() || steps>grade || values.size()!=pin->offsets[grade].back())
            throw std::invalid_argument("boundary trace must name a complete original grade and adjacent steps");
        std::size_t retained=pin->retained_entries;
        for(std::size_t i=0;i<=steps;++i) {
            retained=machine::checked_size_add(retained,pin->offsets[grade-i].back());
            if(i<steps)for(const auto& map:pin->maps[grade-i])retained=machine::checked_size_add(retained,map.rows());
            if(i+1<steps)retained=machine::checked_size_add(retained,machine::checked_size_mul(
                pin->offsets[grade-i-1].size()-1,pin->offsets[grade-i-2].back()));
        }
        guard.entries(retained);Vector field;for(const auto& q:values)field.push_back(scalar(q,guard));
        BoundaryTrace result;result.fields.push_back(strings(field));
        for(std::size_t i=0;i<steps;++i) {
            const auto k=grade-i;const auto& arena=pin->endpoint->source.grade(core::Grade{static_cast<std::uint16_t>(k)});
            auto boundary=apply(*pin,k,field,guard);Vector occurrences,assembled(boundary.size());
            for(std::size_t slot=0;slot<pin->maps[k].size();++slot) {
                const auto up=arena.upper_index[slot],lo=arena.lower_index[slot];
                const Vector local(field.begin()+static_cast<Vector::difference_type>(pin->offsets[k][up]),
                    field.begin()+static_cast<Vector::difference_type>(pin->offsets[k][up+1]));
                const auto value=numeric::checked_apply(pin->maps[k][slot],local,guard);
                occurrences.insert(occurrences.end(),value.begin(),value.end());
                for(std::size_t j=0;j<value.size();++j)assembled[pin->offsets[k-1][lo]+j]=guard.add(assembled[pin->offsets[k-1][lo]+j],value[j]);
            }
            if(assembled!=boundary)throw std::logic_error("original occurrence vectors do not assemble to the boundary");
            std::vector<std::vector<std::string>> channels;
            if(i+1<steps) {
                Vector total(pin->offsets[k-2].back());
                for(std::size_t cell=0;cell+1<pin->offsets[k-1].size();++cell) {
                    Vector selected(boundary.size());
                    for(auto j=pin->offsets[k-1][cell];j<pin->offsets[k-1][cell+1];++j)selected[j]=boundary[j];
                    const auto contribution=apply(*pin,k-1,selected,guard);
                    for(std::size_t j=0;j<total.size();++j)total[j]=guard.add(total[j],contribution[j]);
                    channels.push_back(strings(contribution));
                }
                if(std::any_of(total.begin(),total.end(),[](const auto& q){return q!=0;}))
                    throw std::logic_error("retained intermediate channels have a nonzero strict chain sum");
            }
            result.occurrences.push_back(strings(occurrences));result.channels.push_back(std::move(channels));
            result.fields.push_back(strings(boundary));field=std::move(boundary);
        }
        result.steps=guard.steps();return result;
    }
    BoundaryInjection inject(std::size_t grade,const std::string& archive,const std::vector<std::size_t>& mapping,
                             const std::vector<std::string>& amplitudes,std::size_t max_bytes) const {
        using namespace rfc;const auto pin=state_;Guard guard(pin->limits);
        if(grade>=pin->offsets.size())throw std::out_of_range("original injection grade required");
        auto source=decode(archive,max_bytes);
        if(source->source.grade_count()!=2)throw std::invalid_argument("acquisition requires its declared grade-one source");
        const auto n=source->source.grade(core::Grade{0}).cell_count();const auto& arena=source->source.grade(core::Grade{1});
        if(mapping.size()!=n || amplitudes.size()!=arena.cell_count())throw std::invalid_argument("complete acquisition coordinates required");
        guard.dimension(machine::checked_size_add(n,arena.cell_count()));
        guard.entries(machine::checked_size_add(pin->retained_entries,machine::checked_size_add(
            machine::checked_size_mul(n,4),machine::checked_size_mul(arena.incidence_count(),4))));
        std::set<std::size_t> seen;
        for(const auto row:mapping) {
            guard.step();
            if(row>=pin->offsets[grade].back() || !seen.insert(row).second)
                throw std::invalid_argument("injection must name distinct actual original grade stalk coordinates");
        }
        local::LinearRelationalFiberSystem fibers{local::LinearStalkBundle(source->source)};
        auto action=local::PreparedGradeFieldAction::factored(fibers,core::Grade{1});
        Vector input;for(const auto& q:amplitudes)input.push_back(scalar(q,guard));
        std::vector<std::size_t> rows(n);for(std::size_t i=0;i<n;++i)rows[i]=i;
        const auto boundary=action.apply_selected(input,rows,pin->limits.integer_bits,[&]{guard.step();});
        Vector field(pin->offsets[grade].back());
        for(std::size_t i=0;i<n;++i){guard.step();field[mapping[i]]=boundary[i];}
        return {strings(field),strings(boundary),guard.steps(),arena.cell_count(),arena.incidence_count()};
    }
};
} // namespace rexgraph::python
