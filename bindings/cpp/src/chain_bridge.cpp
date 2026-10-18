#include "chain_bridge.hpp"
#include "rfc/storage/temporal_endpoint.hpp"
#include "rfc/geometry/chain_map.hpp"
#include "rfc/geometry/semantic_realization.hpp"
#include "rfc/learning/semantic_network.hpp"
#include "rfc/machine/checked_size.hpp"
#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>

namespace rexgraph::python {
std::vector<std::vector<std::uint64_t>> observe_context_counts(
    const std::vector<std::vector<std::uint64_t>>& counts,
    const std::vector<std::uint32_t>& features, const std::vector<std::uint32_t>& labels,
    std::uint64_t maximum) {
    if (counts.empty() || counts.front().empty() || features.size() != labels.size() || !maximum)
        throw std::invalid_argument("context count dimensions or observation limit");
    const auto width = counts.front().size();
    std::uint64_t total = 0;
    for (const auto& row : counts) {
        if (row.size() != width) throw std::invalid_argument("context count width differs");
        for (const auto value : row) {
            if (value > maximum - total) throw std::length_error("context observation limit");
            total += value;
        }
    }
    if (features.size() > maximum - total) throw std::length_error("context observation limit");
    for (std::size_t i = 0; i < features.size(); ++i)
        if (features[i] >= counts.size() || labels[i] >= width)
            throw std::out_of_range("context feature or label differs");
    auto result = counts;
    for (std::size_t i = 0; i < features.size(); ++i) ++result[features[i]][labels[i]];
    return result;
}
namespace {
using namespace rfc;
numeric::Rational scalar(const std::string& value, std::size_t bits) {
    numeric::Rational q;
    if (q.set_str(value, 16) != 0 || q.get_den() == 0)
        throw std::invalid_argument("invalid exact chain coefficient");
    q.canonicalize();
    if (mpz_sizeinbase(q.get_num().get_mpz_t(), 2) > bits ||
        mpz_sizeinbase(q.get_den().get_mpz_t(), 2) > bits)
        throw std::length_error("chain coefficient bit bound exceeded");
    return q;
}
void bound(std::size_t value, std::size_t limit, const char* message) {
    if (value > limit) throw std::length_error(message);
}
}
struct ChainMap::State {
    std::shared_ptr<const rfc::storage::TemporalEndpoint> source, target;
    rfc::geometry::RelationalChainMap map;
    std::vector<std::size_t> limits;
    std::unique_ptr<rfc::geometry::SemanticScaleRealization> realization;
    std::unique_ptr<rfc::learning::CompiledMachineSemanticMap> integer;
    State(std::shared_ptr<const rfc::storage::TemporalEndpoint> s,
          std::shared_ptr<const rfc::storage::TemporalEndpoint> t,
          std::vector<rfc::geometry::LinearGradeMap> family,
          std::uint64_t denominator, std::vector<std::size_t> policy)
        : source(std::move(s)), target(std::move(t)),
          map(source->source, target->source, std::move(family)), limits(std::move(policy)) {
        if (!map.is_chain_map()) throw std::invalid_argument("complete map does not commute with boundary");
        if (denominator) {
            for (const auto* complex : {&source->source, &target->source})
                for (std::size_t k = 0; k < complex->grade_count(); ++k)
                    for (const auto& scale : complex->grade(rfc::core::Grade{static_cast<std::uint16_t>(k)}).cell_scales)
                        if (scale != 1) throw std::invalid_argument("integer chain profile requires unit retained lattice scales");
            std::vector<rfc::geometry::LinearGradeMap> scaled;
            const rfc::numeric::Rational n{rfc::numeric::Integer{std::to_string(denominator)}};
            for (std::size_t k = 0; k < source->source.grade_count(); ++k) {
                const rfc::core::Grade grade{static_cast<std::uint16_t>(k)};
                rfc::geometry::LinearGradeMap row(source->source.grade(grade).cell_count());
                for (std::size_t i = 0; i < row.size(); ++i)
                    for (const auto& term : map.image(grade, i)) row[i].push_back({term.target, term.coefficient * n});
                scaled.push_back(std::move(row));
            }
            realization = std::make_unique<rfc::geometry::SemanticScaleRealization>(
                source->source.semantic_scale(), target->source.semantic_scale(),
                rfc::geometry::RelationalChainMap(source->source, target->source, std::move(scaled)));
            integer = std::make_unique<rfc::learning::CompiledMachineSemanticMap>(*realization);
        }
    }
};
ChainMap::ChainMap(std::shared_ptr<const State> state) : state_(std::move(state)) {}
std::shared_ptr<ChainMap> ChainMap::prepare(const std::shared_ptr<Source>& source,
    const std::shared_ptr<Source>& target, const ChainFamily& family,
    std::uint64_t denominator, const std::vector<std::size_t>& limits) {
    if (!source || !target || limits.size() != 3)
        throw std::invalid_argument("chain map requires two owners and three resource limits");
    auto s = source->endpoint(), t = target->endpoint();
    if (family.size() != s->source.grade_count() || t->source.grade_count() < family.size())
        throw std::invalid_argument("chain map requires the complete grade family");
    std::size_t records = family.size(), work = 0;
    std::vector<rfc::geometry::LinearGradeMap> maps;
    for (std::size_t k = 0; k < family.size(); ++k) {
        const rfc::core::Grade grade{static_cast<std::uint16_t>(k)};
        if (family[k].size() != s->source.grade(grade).cell_count())
            throw std::invalid_argument("chain map source cell count differs");
        bound(rfc::machine::checked_size_add(s->source.grade(grade).cell_count(),
              t->source.grade(grade).cell_count()), limits[0], "chain map coordinate bound exceeded");
        records = rfc::machine::checked_size_add(records, family[k].size());
        bound(records, limits[0], "chain map record bound exceeded");
        rfc::geometry::LinearGradeMap row(family[k].size());
        for (std::size_t i = 0; i < row.size(); ++i) {
            records = rfc::machine::checked_size_add(records, family[k][i].size());
            bound(records, limits[0], "chain map record bound exceeded");
            for (const auto& term : family[k][i]) {
                if (term.target >= t->source.grade(grade).cell_count())
                    throw std::out_of_range("chain map target cell index");
                row[i].push_back({term.target, scalar(term.coefficient, limits[2])});
            }
        }
        const auto incidences = rfc::machine::checked_size_add(
            s->source.grade(grade).incidence_count(), t->source.grade(grade).incidence_count());
        work = rfc::machine::checked_size_add(work, rfc::machine::checked_size_mul(
            rfc::machine::checked_size_add(incidences, 1), records));
        bound(work, limits[1], "chain map validation work bound exceeded");
        maps.push_back(std::move(row));
    }
    return std::shared_ptr<ChainMap>(new ChainMap(std::make_shared<State>(
        std::move(s), std::move(t), std::move(maps), denominator, limits)));
}
std::vector<std::string> ChainMap::push(std::uint16_t grade, const std::vector<std::string>& values) const {
    const auto owner = state_;
    const rfc::core::Grade g{grade};
    if (grade >= owner->source->source.grade_count() || values.size() != owner->source->source.grade(g).cell_count())
        throw std::invalid_argument("chain field differs from its declared grade");
    rfc::numeric::RationalVector input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(scalar(value, owner->limits[2]));
    auto output = owner->map.push_field(g, input);
    std::vector<std::string> result;
    result.reserve(output.size());
    for (auto& q : output) { q.canonicalize(); static_cast<void>(scalar(q.get_str(16), owner->limits[2])); result.push_back(q.get_str(16)); }
    return result;
}
std::vector<std::int64_t> ChainMap::push_integer(std::uint16_t grade, const std::vector<std::int64_t>& values) const {
    const auto owner = state_;
    if (!owner->integer) throw std::invalid_argument("integer execution requires a declared common denominator");
    if (grade >= owner->source->source.grade_count()) throw std::invalid_argument("integer field grade differs");
    auto output = owner->integer->push(rfc::field::MachineLatticeField(
        owner->source->source, rfc::core::Grade{grade}, values)).values();
    for (const auto value : output) {
        const rfc::numeric::Rational q{rfc::numeric::Integer{std::to_string(value)}};
        static_cast<void>(scalar(q.get_str(16), owner->limits[2]));
    }
    return output;
}

LiteralTokenChart literal_token_chart(const std::string& raw, const std::vector<std::string>& words,
    const std::vector<LiteralTokenSpan>& spans, std::uint64_t name_space, std::size_t max_bytes,
    const std::vector<std::size_t>& limits) {
    if(raw.empty() || raw.size()>8192 || words.empty() || words.size()>128 || words.size()!=spans.size())
        throw std::invalid_argument("token chart dimensions");
    if(!name_space || name_space==std::numeric_limits<std::uint64_t>::max() || !max_bytes ||
        limits.size()!=3 || std::any_of(limits.begin(),limits.end(),[](auto n){return !n;}))
        throw std::invalid_argument("token chart namespace or resource limits");
    std::size_t previous=0;
    for(std::size_t i=0;i<spans.size();++i) {
        const auto [begin,end]=spans[i];
        if(begin<previous || begin>=end || end>raw.size() || end-begin!=words[i].size() ||
            raw.compare(begin,end-begin,words[i])!=0)
            throw std::invalid_argument("ordered literal token span required");
        previous=end;
    }
    const auto declaration=[](std::uint64_t ns,std::uint16_t scale,const std::vector<std::size_t>& sizes) {
        SourceDeclaration d;d.name_space=ns;d.max_grade=2;d.semantic_scale=scale;
        d.types.push_back({"ordered-interval",2,4});
        for(std::size_t i=0;i<sizes[0];++i) d.witnesses.push_back(i);
        for(std::size_t k=0;k<sizes.size();++k)
            d.carriers.push_back({10+k,0,1,static_cast<std::uint16_t>(k),sizes[k]});
        return d;
    };
    const auto relation=[](std::uint16_t grade,std::uint64_t key,const std::vector<std::uint64_t>& lower,
        const std::vector<bool>& negative,std::uint64_t first) {
        SourceRelation r;r.grade=grade;r.key=key;
        for(std::size_t i=0;i<lower.size();++i)
            r.occurrences.push_back({first+i,lower[i],true,negative[i],"1",{}});
        return r;
    };
    const auto n=raw.size(), count=words.size();
    auto bytes=declaration(name_space,0,{n+1,n,0});bytes.relations.reserve(n);
    SourceField octets;octets.key=1001;octets.carrier=11;octets.full=true;
    octets.indices.reserve(n);octets.values.reserve(n);
    constexpr char digits[]="0123456789abcdef";
    for(std::size_t i=0;i<n;++i) {
        bytes.relations.push_back(relation(1,20000+i,{i,i+1},{true,false},2*i));
        const auto byte=static_cast<unsigned char>(raw[i]);
        std::string value;
        if(byte>=16) value.push_back(digits[byte>>4]);
        value.push_back(digits[byte&15]);
        octets.indices.push_back(i);octets.values.push_back(std::move(value));
    }
    bytes.fields.push_back(std::move(octets));
    auto tokens=declaration(name_space+1,1,{2*count,3*count-2,count-1});
    tokens.carriers.push_back({20,9200601,0,0,count});
    std::vector<std::size_t> cuts;cuts.reserve(2*count);
    for(const auto& span:spans) {cuts.push_back(span.begin);cuts.push_back(span.end);}
    std::vector<std::pair<std::size_t,std::size_t>> intervals;
    for(std::size_t i=0;i<count;++i) intervals.emplace_back(2*i,2*i+1);
    for(std::size_t i=0;i+1<count;++i) intervals.emplace_back(2*i+1,2*i+2);
    for(std::size_t i=0;i+1<count;++i) intervals.emplace_back(2*i,2*i+3);
    for(std::size_t i=0;i<intervals.size();++i) {
        const auto [a,b]=intervals[i];
        tokens.relations.push_back(relation(1,30000+i,{a,b},{true,false},20000+2*i));
    }
    for(std::size_t i=0;i+1<count;++i)
        tokens.relations.push_back(relation(2,40000+i,
            {30000+i,30000+count+i,30000+i+1,30000+2*count-1+i},{false,false,false,true},40000+4*i));
    ChainFamily family(3);
    for(const auto cut:cuts) family[0].push_back({{static_cast<std::uint32_t>(cut),"1"}});
    for(const auto& [a,b]:intervals) {
        std::vector<ChainTerm> image;image.reserve(cuts[b]-cuts[a]);
        for(auto j=cuts[a];j<cuts[b];++j) image.push_back({static_cast<std::uint32_t>(j),"1"});
        family[1].push_back(std::move(image));
    }
    family[2].resize(count-1);
    LiteralTokenChart result;
    result.bytes=Source::build(bytes,max_bytes);result.tokens=Source::build(tokens,max_bytes);
    result.map=ChainMap::prepare(result.tokens,result.bytes,family,0,limits);
    result.family=std::move(family);
    result.spans=spans;
    return result;
}

std::vector<LiteralTokenSpan> literal_whitespace_spans(const std::string& raw) {
    if(raw.empty() || raw.size()>8192) throw std::invalid_argument("token chart dimensions");
    const auto separator=[](unsigned char byte) {return byte==32 || (byte>=9 && byte<=13);};
    std::vector<LiteralTokenSpan> spans;
    for(std::size_t i=0;i<raw.size();) {
        if(separator(static_cast<unsigned char>(raw[i]))) {++i;continue;}
        const auto begin=i;
        do {++i;} while(i<raw.size() && !separator(static_cast<unsigned char>(raw[i])));
        if(spans.size()==128) throw std::invalid_argument("token chart dimensions");
        spans.push_back({begin,i});
    }
    if(spans.empty()) throw std::invalid_argument("token chart dimensions");
    return spans;
}

LiteralTokenChart literal_whitespace_chart(const std::string& raw, std::uint64_t name_space,
    std::size_t max_bytes, const std::vector<std::size_t>& limits) {
    const auto spans=literal_whitespace_spans(raw);
    std::vector<std::string> words;words.reserve(spans.size());
    for(const auto& span:spans) words.push_back(raw.substr(span.begin,span.end-span.begin));
    return literal_token_chart(raw,words,spans,name_space,max_bytes,limits);
}
}
