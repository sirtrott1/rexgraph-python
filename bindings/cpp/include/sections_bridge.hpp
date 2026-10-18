#pragma once
#include <rfc/local/constraint_system.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <set>
#include <string>

namespace rexgraph::python {
struct SectionChart {
    std::string name;
    std::size_t relation{};
    std::vector<std::size_t> support;
    std::vector<std::vector<std::size_t>> allowed;
};
struct SectionResult {
    std::vector<std::vector<std::size_t>> sections;
    std::vector<std::size_t> conflict_charts, conflict_cells;
    std::size_t candidate_count{}, domain_count{}, admitted_work{};
    bool satisfiable{}, subset_minimal{}, constraint_complete{};
};

class RelationalSections {
    using System=rfc::local::FiniteConstraintSystem<std::size_t>;
    std::shared_ptr<const rfc::storage::TemporalEndpoint> endpoint_;
    System system_;
    std::vector<std::vector<std::size_t>> domains_;
    std::vector<rfc::algebra::SubcomplexMask> cover_;
    std::vector<std::size_t> enumeration_charts_;
    std::size_t candidates_{1}, domain_count_{1}, work_{};
    static auto decode(const std::string& archive,std::size_t max_bytes) {
        rfc::storage::NativeSeriesLimits limits;limits.max_archive_bytes=max_bytes;
        auto history=rfc::storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()),archive.size()),limits);
        return history.tail().pin();
    }
    static std::size_t bounded_mul(std::size_t a,std::size_t b,std::size_t limit,const char* message) {
        if(b && a>limit/b)throw std::invalid_argument(message);
        return a*b;
    }
public:
    RelationalSections(const std::string& archive,const std::vector<std::vector<std::size_t>>& domains,
                       const std::vector<SectionChart>& charts,const std::vector<std::size_t>& limits,
                       std::size_t max_bytes,const std::vector<std::size_t>& enumeration_charts)
        :endpoint_(decode(archive,max_bytes)),system_(endpoint_->source),domains_(domains),enumeration_charts_(enumeration_charts) {
        using namespace rfc;
        if(limits.size()!=3 || !limits[0] || !limits[1] || !limits[2])
            throw std::invalid_argument("three positive section limits required");
        const auto& source=endpoint_->source;
        if(source.grade_count()!=2 || domains.size()!=source.grade(core::Grade{0}).cell_count())
            throw std::invalid_argument("section profile requires a complete grade-one carrier");
        const auto structure=machine::checked_size_add(source.grade(core::Grade{1}).cell_count(),source.grade(core::Grade{1}).incidence_count());
        std::size_t entries=machine::checked_size_add(machine::checked_size_add(domains.size(),charts.size()),structure),tuple_work=structure+1;
        if(entries>limits[1])throw std::invalid_argument("section declaration exceeds entry admission");
        const auto add_entries=[&](std::size_t count) {
            if(count>limits[1]-entries)throw std::invalid_argument("section declaration exceeds entry admission");
            entries+=count;
        };
        for(std::size_t i=0;i<domains.size();++i) {
            const auto& values=domains[i];
            if(values.empty() || std::set<std::size_t>(values.begin(),values.end()).size()!=values.size())
                throw std::invalid_argument("section domains require distinct finite values");
            domain_count_=bounded_mul(domain_count_,values.size(),limits[0],"section Cartesian admission exceeded");
            add_entries(values.size());
            system_.set_domain({core::Grade{0},static_cast<std::uint32_t>(i)},values);
        }
        for(const auto& chart:charts) {
            add_entries(chart.support.size());
            add_entries(chart.allowed.size());
            const auto count=bounded_mul(chart.support.size(),chart.allowed.size(),limits[1],"section tuple admission exceeded");
            add_entries(count);tuple_work=machine::checked_size_add(tuple_work,count);
            if(chart.relation>=source.grade(core::Grade{1}).cell_count())throw std::out_of_range("section chart relation");
            auto patch=system_.carrier().cell_patch({core::Grade{1},static_cast<std::uint32_t>(chart.relation)});
            std::vector<core::CellId> support;
            for(const auto cell:chart.support) {
                if(cell>=domains.size())throw std::out_of_range("section chart lower cell");
                support.push_back({core::Grade{0},static_cast<std::uint32_t>(cell)});
            }
            system_.add_chart(chart.name,patch,std::move(support),chart.allowed);
            cover_.push_back(std::move(patch));
        }
        if(enumeration_charts_.empty())candidates_=domain_count_;
        else {
            std::set<std::size_t> selected,covered;
            for(const auto index:enumeration_charts_) {
                if(index>=charts.size() || !selected.insert(index).second)
                    throw std::invalid_argument("distinct retained enumeration charts required");
                const auto& chart=charts[index];
                candidates_=bounded_mul(candidates_,chart.allowed.size(),limits[0],"section tuple product admission exceeded");
                covered.insert(chart.support.begin(),chart.support.end());
            }
            if(covered.size()!=domains.size())throw std::invalid_argument("enumeration charts do not cover every finite variable");
        }
        // Admit the full enumeration and the native cover/core searches before execution.
        work_=bounded_mul(std::max(domain_count_,candidates_),domains.size()+1,limits[2],"section work admission exceeded");
        work_=bounded_mul(work_,charts.size()+2,limits[2],"section work admission exceeded");
        work_=bounded_mul(work_,charts.size()+2,limits[2],"section work admission exceeded");
        work_=bounded_mul(work_,tuple_work,limits[2],"section work admission exceeded");
    }
    SectionResult sections() const {
        using namespace rfc;
        SectionResult result;result.candidate_count=candidates_;result.domain_count=domain_count_;result.admitted_work=work_;
        const auto report=system_.diagnose_cover(cover_);
        result.constraint_complete=report.constraint_complete;
        if(!report.constraint_complete)throw std::logic_error("section cover omits a retained chart");
        std::vector<std::size_t> slots(enumeration_charts_.empty()?domains_.size():enumeration_charts_.size(),0);
        std::set<std::vector<std::size_t>> unique;
        for(std::size_t candidate=0;candidate<candidates_;++candidate) {
            System::Assignment assignment;
            bool compatible=true;
            if(enumeration_charts_.empty()) {
                for(std::size_t i=0;i<slots.size();++i)
                    assignment.emplace(core::CellId{core::Grade{0},static_cast<std::uint32_t>(i)},domains_[i][slots[i]]);
            } else {
                algebra::SubcomplexMask patch(endpoint_->source);
                for(std::size_t i=0;i<slots.size();++i) {
                    const auto index=enumeration_charts_[i];const auto& chart=system_.charts()[index];
                    System::Assignment local;
                    for(std::size_t j=0;j<chart.support.size();++j)local.emplace(chart.support[j],chart.allowed[slots[i]][j]);
                    if(!system_.compatible_on_overlap(assignment,patch,local,cover_[index])) {compatible=false;break;}
                    assignment=system_.glue(assignment,patch,local,cover_[index]);patch=join(patch,cover_[index]);
                }
            }
            if(compatible && system_.globally_valid(assignment)) {
                std::vector<std::size_t> values;
                for(std::size_t i=0;i<domains_.size();++i)values.push_back(assignment.at({core::Grade{0},static_cast<std::uint32_t>(i)}));
                unique.insert(std::move(values));
            }
            for(std::size_t i=slots.size();i>0;--i) {
                const auto size=enumeration_charts_.empty()?domains_[i-1].size():system_.charts()[enumeration_charts_[i-1]].allowed.size();
                if(++slots[i-1]<size)break;
                slots[i-1]=0;
            }
        }
        result.sections.assign(unique.begin(),unique.end());
        result.satisfiable=!result.sections.empty();
        if(result.satisfiable!=system_.global_section().has_value())throw std::logic_error("native section enumeration disagrees with search");
        if(!result.satisfiable) {
            const auto core=system_.exact_global_unsat_core();
            if(!core || !core->subset_minimal)throw std::logic_error("native section conflict has no minimal certificate");
            result.conflict_charts=core->chart_indices;result.subset_minimal=core->subset_minimal;
            for(const auto cell:core->cells) {
                if(cell.grade.value!=0)throw std::logic_error("section conflict outside declared variable grade");
                result.conflict_cells.push_back(cell.index);
            }
        }
        return result;
    }
};
}
