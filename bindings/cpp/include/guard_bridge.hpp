#pragma once
#include <array>
#include <rfc/algebra/materialize.hpp>
#include <rfc/algebra/owned_gluing.hpp>
#include <rfc/core/boundary_fiber.hpp>
#include <rfc/geometry/chain_map.hpp>
#include <rfc/storage/native_series_codec.hpp>

namespace rexgraph::python {
struct GuardMatch {
    bool matched{};
    std::string reason;
    std::vector<std::vector<std::size_t>> cells, relations, occurrences;
    std::size_t steps{};
};
struct GuardOccurrence {
    std::size_t slot{},lower{};
    std::uint64_t key{};
    bool exists{},negative{};
    std::string coefficient;
    std::vector<std::string> roles;
};
struct GuardBoundary {
    std::string type;
    std::vector<GuardOccurrence> occurrences;
};
struct DictionaryLicense {
    std::size_t synset{};
    std::vector<std::size_t> left, right;
    GuardMatch witness;
};
struct DictionaryPair {
    std::size_t origin{},target{};
    std::vector<DictionaryLicense> licenses;
};
struct DictionaryComparison {
    std::vector<DictionaryPair> pairs;
    std::size_t steps{};
};

class RelationGuards {
    std::shared_ptr<const rfc::storage::TemporalEndpoint> endpoint_;
    rfc::numeric::ExactActionLimits limits_;
    std::size_t preparation_steps_{};
    static auto decode(const std::string& archive,const std::vector<std::size_t>& bounds,std::size_t max_bytes,
        std::size_t max_logical_size) {
        if(bounds.size()!=3 || !bounds[0] || !bounds[1] || !bounds[2])
            throw std::invalid_argument("three positive relation guard limits required");
        rfc::storage::NativeSeriesLimits limits;limits.max_archive_bytes=max_bytes;
        limits.temporal.max_records=bounds[0];limits.temporal.max_dimension=bounds[0];
        limits.temporal.max_logical_size=max_logical_size;
        auto history=rfc::storage::decode_native_series(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(archive.data()),archive.size()),limits);
        return history.tail().pin();
    }
public:
    RelationGuards(const std::string& archive,const std::vector<std::size_t>& limits,std::size_t max_bytes,
        std::size_t max_logical_size=0)
        :endpoint_(decode(archive,limits,max_bytes,max_logical_size)) {
        using namespace rfc;
        if(limits.size()!=3 || !limits[0] || !limits[1] || !limits[2])
            throw std::invalid_argument("three positive relation guard limits required");
        limits_={limits[0],limits[0],limits[1],limits[2]};
        const auto& source=endpoint_->source;
        if(source.grade_count()!=2)throw std::invalid_argument("relation guard profile requires grade one");
        numeric::ExactActionGuard guard(limits_);
        guard.entries(algebra::owned_gluing_detail::words(source));
        algebra::owned_gluing_detail::source(source,guard);
        preparation_steps_=guard.steps();
    }
    std::size_t preparation_steps() const {return preparation_steps_;}
    std::size_t relation_count() const {return endpoint_->source.grade(rfc::core::Grade{1}).cell_count();}
    std::uint64_t relation_key(std::size_t row) const {
        return endpoint_->source.grade(rfc::core::Grade{1}).relation_identities.at(row).value;
    }
    void verify_original_row(const RelationGuards& other,std::size_t row,std::size_t target,
                             rfc::numeric::ExactActionGuard& guard) const {
        using namespace rfc;
        using namespace algebra::owned_gluing_detail;
        const auto& a=endpoint_->source;const auto& b=other.endpoint_->source;const core::Grade g{1},zero{0};
        const auto resident=add(source_words(),other.source_words());guard.entries(mul(resident,3));
        require(relation_key(row)==other.relation_key(target),"pattern changes original relation identity");
        algebra::SubcomplexMask left(a),right(b);left.select({g,static_cast<std::uint32_t>(row)});
        right.select({g,static_cast<std::uint32_t>(target)});left.close_boundaries(a);right.close_boundaries(b);
        auto x=algebra::materialize_subcomplex(a,left),y=algebra::materialize_subcomplex(b,right);
        std::map<core::IdentityKey,std::uint32_t> cells;
        for(std::size_t i=0;i<y.complex.grade(zero).cell_count();++i) {
            guard.step();cells.emplace(y.complex.grade(zero).identities[i],static_cast<std::uint32_t>(i));
        }
        algebra::RetainedSourceMap mapping;mapping.cells.resize(2);mapping.occurrences.resize(2);
        for(const auto identity:x.complex.grade(zero).identities) {
            guard.step();const auto found=cells.find(identity);
            require(found!=cells.end(),"pattern changes original lower identity");mapping.cells[0].push_back(found->second);
        }
        require(mapping.cells[0].size()==cells.size(),"pattern changes original boundary carrier");
        for(std::size_t i=0;i<mapping.cells[0].size();++i) {
            guard.step();require(x.complex.grade(zero).cell_scales[i]==y.complex.grade(zero).cell_scales[mapping.cells[0][i]],
                "pattern changes original lower scale");
        }
        mapping.cells[1]={0};const auto& xf=x.complex.grade(g);const auto& yf=y.complex.grade(g);
        require(xf.cell_scales[0]==yf.cell_scales[0],"pattern changes original relation scale");
        require(xf.incidence_count()==yf.incidence_count(),"pattern changes original occurrence count");
        for(std::size_t i=0;i<xf.incidence_count();++i) {
            guard.step();require(xf.occurrence_identities[i]==yf.occurrence_identities[i],"pattern changes original occurrence identity");
            require(xf.lattice_coefficient[i]==yf.lattice_coefficient[i],"pattern changes original lattice coefficient");
            mapping.occurrences[1].push_back(static_cast<std::uint32_t>(i));
        }
        algebra::owned_gluing_detail::map(x.complex,y.complex,mapping,true,guard);
    }
    rfc::numeric::ExactActionLimits limits() const {return limits_;}
    std::size_t source_words() const {return rfc::algebra::owned_gluing_detail::words(endpoint_->source);}
    GuardBoundary boundary(std::size_t row) const {
        using namespace rfc;
        const auto& source=endpoint_->source;const core::Grade g{1};const auto& arena=source.grade(g);
        if(row>=arena.cell_count())throw std::out_of_range("guard relation row");
        GuardBoundary result;result.type=source.signature().type(arena.relation_types[row]).name;
        const auto [begin,end]=arena.incidence_range(row);
        for(auto p=begin;p<end;++p) {
            GuardOccurrence item;item.slot=p;item.lower=arena.lower_index[p];item.key=arena.occurrence_identities[p].value;
            item.exists=arena.aspects.get(source.signature().existence_aspect(),p);
            item.negative=arena.aspects.get(source.signature().orientation_aspect(),p);
            item.coefficient=source.incidence_coefficient(g,p).get_str();
            for(std::size_t j=0;j<source.signature().aspects().size();++j) {
                const auto id=static_cast<composite::AspectId>(j);const auto& aspect=source.signature().aspects().at(id);
                if(aspect.role==composite::AspectRole::RelationRole && arena.aspects.get(id,p))item.roles.push_back(aspect.name);
            }
            result.occurrences.push_back(std::move(item));
        }
        return result;
    }
    GuardMatch encoding_accession(const RelationGuards& target,
        const std::vector<std::size_t>& origins,const std::vector<std::size_t>& targets,
        std::size_t remaining) const {
        using namespace rfc;
        using namespace algebra::owned_gluing_detail;
        if(origins.empty() || origins.size()!=targets.size() || origins.size()>128)
            throw std::invalid_argument("encoding accession requires paired bounded original rows");
        if(!remaining)throw std::length_error("encoding accession work admission exceeded");
        auto bounds=limits_;
        bounds.steps=std::min({bounds.steps,target.limits_.steps,remaining});
        bounds.entries=std::min(bounds.entries,target.limits_.entries);
        numeric::ExactActionGuard guard(bounds);
        const auto& a=endpoint_->source;const auto& b=target.endpoint_->source;
        const core::Grade g{1},zero{0};
        const auto ambient_words=add(source_words(),target.source_words());
        guard.entries(mul(ambient_words,3));
        for(std::size_t i=0;i<mul(ambient_words,2);++i)guard.step();
        GuardMatch result;
        const auto fail=[&](const char* why){result.reason=why;result.steps=guard.steps();
            result.cells.clear();result.relations.clear();result.occurrences.clear();return result;};
        if(a.semantic_scale()!=b.semantic_scale())return fail("semantic-scale");
        for(const auto* c:{&a,&b}) {
            std::size_t existence{},orientation{};
            for(std::size_t i=0;i<c->signature().aspects().size();++i) {
                guard.step();const auto& d=c->signature().aspects().at(static_cast<composite::AspectId>(i));
                if(d.role==composite::AspectRole::Existence)++existence;
                else if(d.role==composite::AspectRole::Orientation)++orientation;
                else if(d.role!=composite::AspectRole::RelationRole || d.default_value)return fail("encoding-aspect-schema");
            }
            if(existence!=1 || orientation!=1)return fail("encoding-aspect-schema");
        }
        for(const bool orientation:{false,true}) {
            guard.step();const auto& ad=a.signature().aspects().at(orientation?a.signature().orientation_aspect():a.signature().existence_aspect());
            const auto& bd=b.signature().aspects().at(orientation?b.signature().orientation_aspect():b.signature().existence_aspect());
            if(ad.name!=bd.name || ad.role!=bd.role || ad.default_value!=bd.default_value)return fail("encoding-aspect-semantics");
        }
        algebra::SubcomplexMask left(a),right(b);
        std::map<std::uint32_t,std::uint32_t> lower,upper;
        std::set<std::uint32_t> images;
        const auto& aa=a.grade(g);const auto& ba=b.grade(g);
        for(std::size_t i=0;i<origins.size();++i) {
            guard.step();const auto x=origins[i],y=targets[i];
            if(x>=aa.cell_count() || y>=ba.cell_count())throw std::out_of_range("encoding accession row");
            if(!upper.emplace(static_cast<std::uint32_t>(x),static_cast<std::uint32_t>(y)).second ||
               !images.insert(static_cast<std::uint32_t>(y)).second)
                throw std::invalid_argument("encoding accession rows must be distinct on both sides");
            const auto xf=boundary(x),yf=target.boundary(y);
            if(xf.type!="use-encoding" || yf.type!="use-encoding" ||
               xf.occurrences.size()!=2 || yf.occurrences.size()!=2)return fail("encoding-fiber");
            const auto& xt=a.signature().type(aa.relation_types[x]);
            const auto& yt=b.signature().type(ba.relation_types[y]);
            if(xt.min_arity!=yt.min_arity || xt.max_arity!=yt.max_arity ||
               xt.boundary_policy!=yt.boundary_policy || xt.constructor!=core::CompositeConstructorKind::None ||
               yt.constructor!=core::CompositeConstructorKind::None || xt.constraints.shape!=core::RelationShape::Generic ||
               yt.constraints.shape!=core::RelationShape::Generic || xt.constraints.distinguished_role ||
               yt.constraints.distinguished_role || !xt.constraints.rules.empty() || !yt.constraints.rules.empty())
                return fail("encoding-type");
            if(aa.cell_scales[x]!=ba.cell_scales[y])return fail("relation-scale");
            for(std::size_t j=0;j<2;++j) {
                guard.step();const auto& u=xf.occurrences[j];const auto& v=yf.occurrences[j];
                const std::vector<std::string> roles{j?"literal-record":"encoded-use"};
                if(u.roles!=roles || v.roles!=roles || !u.exists || !v.exists ||
                   u.negative || v.negative || u.coefficient!="1" || v.coefficient!="1")
                    return fail("encoding-occurrence");
                if(aa.lattice_coefficient[u.slot]!=ba.lattice_coefficient[v.slot] ||
                   a.grade(zero).cell_scales[u.lower]!=b.grade(zero).cell_scales[v.lower])
                    return fail("encoding-scale");
                const auto [found,inserted]=lower.emplace(static_cast<std::uint32_t>(u.lower),static_cast<std::uint32_t>(v.lower));
                if(!inserted && found->second!=v.lower)return fail("shared-identity-split");
                result.occurrences.push_back({u.slot,v.slot});
            }
            left.select({g,static_cast<std::uint32_t>(x)});right.select({g,static_cast<std::uint32_t>(y)});
        }
        std::set<std::uint32_t> lower_images;
        for(const auto& [x,y]:lower) {
            guard.step();if(!lower_images.insert(y).second)return fail("shared-identity-merge");
            result.cells.push_back({x,y});
        }
        left.close_boundaries(a);right.close_boundaries(b);
        auto x=algebra::materialize_subcomplex(a,left),y=algebra::materialize_subcomplex(b,right);
        std::vector<geometry::LinearGradeMap> family(2);
        for(const auto cell:x.to_ambient[0]) {
            guard.step();family[0].push_back({{y.from_ambient[0].at(lower.at(cell)),numeric::Rational{1}}});
        }
        for(const auto row:x.to_ambient[1]) {
            guard.step();family[1].push_back({{y.from_ambient[1].at(upper.at(row)),numeric::Rational{1}}});
        }
        // Both grades cover the complete closed encoding carriers.
        guard.entries(add(mul(add(words(a),words(b)),3),mul(add(lower.size(),upper.size()),16)));
        for(std::size_t i=0;i<mul(upper.size(),16);++i)guard.step();
        if(!geometry::RelationalChainMap(x.complex,y.complex,std::move(family)).is_chain_map())
            return fail("encoding-chain-map");
        for(const auto& [u,v]:upper)result.relations.push_back({u,v});
        result.matched=true;result.steps=guard.steps();return result;
    }
    GuardMatch match(const std::vector<std::size_t>& origins,const std::vector<std::size_t>& targets,
                     const std::vector<std::size_t>& fixed,std::size_t remaining) const {
        using namespace rfc;
        using namespace algebra::owned_gluing_detail;
        if(origins.empty() || origins.size()!=targets.size())throw std::invalid_argument("guard needs paired complete relation rows");
        if(!remaining)throw std::length_error("relation guard work admission exceeded");
        auto bounds=limits_;bounds.steps=std::min(bounds.steps,remaining);
        numeric::ExactActionGuard guard(bounds);const auto& source=endpoint_->source;
        const core::Grade g{1};const auto& arena=source.grade(g);
        algebra::SubcomplexMask left(source),right(source);
        std::map<std::uint32_t,std::uint32_t> lower,upper;
        std::set<std::uint32_t> lower_images,upper_images;
        GuardMatch result;
        const auto fail=[&](const char* message){result.reason=message;result.steps=guard.steps();return result;};
        for(std::size_t i=0;i<origins.size();++i) {
            guard.step();const auto a=origins[i],b=targets[i];
            if(a>=arena.cell_count() || b>=arena.cell_count())throw std::out_of_range("guard relation row");
            if(!upper.emplace(static_cast<std::uint32_t>(a),static_cast<std::uint32_t>(b)).second ||
               !upper_images.insert(static_cast<std::uint32_t>(b)).second)
                throw std::invalid_argument("guard relation rows must be distinct on both sides");
            left.select({g,static_cast<std::uint32_t>(a)});right.select({g,static_cast<std::uint32_t>(b)});
        }
        left.close_boundaries(source);right.close_boundaries(source);
        std::set<std::size_t> anchors;
        for(const auto cell:fixed) {
            guard.step();
            if(!anchors.insert(cell).second || cell>=source.grade(core::Grade{0}).cell_count() ||
               !left.contains({core::Grade{0},static_cast<std::uint32_t>(cell)}))
                throw std::invalid_argument("guard anchors must be distinct boundary cells");
        }
        for(std::size_t i=0;i<origins.size();++i) {
            guard.step();const auto a=origins[i],b=targets[i];
            const auto [ab,ae]=arena.incidence_range(a);const auto [bb,be]=arena.incidence_range(b);
            if(ae-ab!=be-bb)return fail("occurrence-count");
            for(std::size_t p=ab;p<ae;++p) {
                guard.step();const auto x=arena.lower_index[p],y=arena.lower_index[bb+p-ab];
                const auto [it,inserted]=lower.emplace(x,y);
                if(!inserted && it->second!=y)return fail("shared-identity-split");
                if(inserted && !lower_images.insert(y).second)return fail("shared-identity-merge");
            }
        }
        for(const auto cell:fixed) {
            guard.step();
            if(lower.at(static_cast<std::uint32_t>(cell))!=cell)return fail("pinned-identity");
        }
        // Close through every retained occurrence, including inactive flags.
        guard.entries(add(words(source),mul(add(lower.size(),upper.size()),8)));
        auto a=algebra::materialize_subcomplex(source,left),b=algebra::materialize_subcomplex(source,right);
        guard.entries(add(words(source),add(words(a.complex),words(b.complex))));
        algebra::RetainedSourceMap map;map.cells.resize(2);map.occurrences.resize(2);
        for(const auto cell:a.to_ambient[0]) {
            guard.step();map.cells[0].push_back(b.from_ambient[0].at(lower.at(cell)));
        }
        for(const auto row:a.to_ambient[1]) {
            guard.step();const auto image=upper.at(row);map.cells[1].push_back(b.from_ambient[1].at(image));
            const auto [begin,end]=arena.incidence_range(row);
            const auto target_begin=arena.incidence_range(image).first;
            const auto local_begin=b.complex.grade(g).incidence_range(b.from_ambient[1].at(image)).first;
            for(auto p=begin;p<end;++p) {
                guard.step();map.occurrences[1].push_back(local_begin+p-begin);
                result.occurrences.push_back({p,target_begin+p-begin});
            }
        }
        try {algebra::owned_gluing_detail::map(a.complex,b.complex,map,true,guard);}
        catch(const std::invalid_argument&) {result.occurrences.clear();return fail("typed-occurrence-map");}
        for(const auto& [x,y]:lower)result.cells.push_back({x,y});
        for(const auto& [x,y]:upper)result.relations.push_back({x,y});
        result.matched=true;result.steps=guard.steps();return result;
    }
    DictionaryComparison shared_synsets(const std::vector<std::vector<std::size_t>>& pairs,
                                        std::size_t remaining) const {
        using namespace rfc;
        using algebra::owned_gluing_detail::add;
        using algebra::owned_gluing_detail::mul;
        if(!remaining)throw std::length_error("dictionary comparison work admission exceeded");
        auto bounds=limits_;bounds.steps=std::min(bounds.steps,remaining);
        numeric::ExactActionGuard guard(bounds);
        const auto& source=endpoint_->source;const core::Grade g{1};
        const auto& arena=source.grade(g);
        std::size_t nested{},stored=add(algebra::owned_gluing_detail::words(source),mul(pairs.size(),6));
        guard.entries(stored);
        const auto step=[&] {
            if(nested>=bounds.steps || guard.steps()>=bounds.steps-nested)
                throw std::length_error("dictionary comparison work admission exceeded");
            guard.step();
        };
        const auto retain=[&](std::size_t words) {stored=add(stored,words);guard.entries(stored);};
        std::set<std::pair<std::size_t,std::size_t>> distinct;
        for(const auto& pair:pairs) {
            step();
            if(pair.size()!=2 || pair[0]>=source.grade(core::Grade{0}).cell_count() ||
               pair[1]>=source.grade(core::Grade{0}).cell_count())throw std::invalid_argument("dictionary pair needs two original lower cells");
            if(!distinct.emplace(pair[0],pair[1]).second)throw std::invalid_argument("dictionary pairs must be distinct");
        }
        struct Link {std::size_t row{},target{};};
        using Index=std::map<std::size_t,std::vector<Link>>;
        std::array<Index,3> indices;
        const std::array<std::string,3> types{"record-entry","entry-sense","sense-synset"};
        const std::array<std::string,4> roles{"dictionary-record","dictionary-entry","dictionary-sense","dictionary-synset"};
        for(std::size_t row=0;row<arena.cell_count();++row) {
            step();const auto& type=source.signature().type(arena.relation_types[row]).name;
            const auto it=std::find(types.begin(),types.end(),type);
            if(it==types.end())continue;
            const auto layer=static_cast<std::size_t>(it-types.begin());
            const core::BoundaryOccurrenceFiber fiber(source,g,row);
            if(fiber.size()!=2)throw std::invalid_argument("dictionary column requires two complete role occurrences");
            bool active=true;
            for(std::size_t slot=0;slot<2;++slot) {
                step();const auto& flag=fiber.occurrences()[slot];
                std::vector<std::string> declared;
                for(std::size_t j=0;j<source.signature().aspects().size();++j) {
                    step();const auto id=static_cast<composite::AspectId>(j);const auto& aspect=source.signature().aspects().at(id);
                    if(aspect.role==composite::AspectRole::RelationRole && flag.aspect(id))declared.push_back(aspect.name);
                }
                if(declared!=std::vector<std::string>{roles[layer+slot]})throw std::invalid_argument("dictionary column differs from its ordered role declaration");
                if(!flag.exists || flag.coefficient==0)active=false;
                else if(flag.negative || flag.coefficient!=1)throw std::invalid_argument("dictionary comparison requires unsigned unit columns");
            }
            if(active) {
                retain(4);indices[layer][fiber.occurrences()[0].lower.index].push_back({row,fiber.occurrences()[1].lower.index});
            }
        }
        struct Route {std::size_t synset{};std::vector<std::size_t> rows;};
        std::map<std::size_t,std::vector<Route>> routes;
        const auto lookup=[](const Index& index,std::size_t key)->const std::vector<Link>& {
            static const std::vector<Link> empty;
            const auto it=index.find(key);return it==index.end()?empty:it->second;
        };
        for(const auto& pair:pairs)for(const auto record:pair) {
            step();if(routes.contains(record))continue;
            retain(2);auto& result=routes[record];
            for(const auto a:lookup(indices[0],record)) {
                step();
                for(const auto b:lookup(indices[1],a.target)) {
                    step();
                    for(const auto c:lookup(indices[2],b.target)) {
                        step();retain(5);result.push_back({c.target,{a.row,b.row,c.row}});
                    }
                }
            }
        }
        DictionaryComparison result;
        for(const auto& pair:pairs) {
            step();DictionaryPair item;item.origin=pair[0];item.target=pair[1];
            std::map<std::size_t,std::vector<const Route*>> targets;
            std::size_t temporary{};
            for(const auto& route:routes.at(pair[1])) {
                step();temporary=add(temporary,3);guard.entries(add(stored,temporary));targets[route.synset].push_back(&route);
            }
            for(const auto& a:routes.at(pair[0])) {
                step();const auto it=targets.find(a.synset);if(it==targets.end())continue;
                for(const auto* b:it->second) {
                    step();retain(64);
                    guard.entries(add(add(stored,temporary),mul(algebra::owned_gluing_detail::words(source),3)));
                    auto witness=match(a.rows,b->rows,{a.synset},bounds.steps-guard.steps()-nested);
                    nested=add(nested,witness.steps);
                    if(witness.matched)item.licenses.push_back({a.synset,a.rows,b->rows,std::move(witness)});
                }
            }
            result.pairs.push_back(std::move(item));
        }
        result.steps=add(guard.steps(),nested);return result;
    }
};

struct DictionarySection {
    std::vector<std::size_t> choices,left,right,fixed;
    GuardMatch witness;
};
struct DictionarySections {
    std::vector<DictionarySection> sections;
    std::size_t steps{},domain_count{},candidate_count{},native_calls{},overlap_rejections{};
};

class DictionaryGuards {
    std::shared_ptr<const RelationGuards> owner_;
    DictionaryComparison comparison_;
    std::size_t retained_{};
public:
    DictionaryGuards(const std::shared_ptr<RelationGuards>& owner,
                     const std::vector<std::vector<std::size_t>>& pairs,std::size_t remaining)
        :owner_(owner) {
        using namespace rfc::algebra::owned_gluing_detail;
        if(!owner_)throw std::invalid_argument("dictionary catalog requires its owning source");
        comparison_=owner_->shared_synsets(pairs,remaining);
        retained_=add(owner_->source_words(),mul(comparison_.pairs.size(),6));
        for(const auto& p:comparison_.pairs)retained_=add(retained_,mul(p.licenses.size(),64));
        rfc::numeric::ExactActionGuard(owner_->limits()).entries(retained_);
    }
    DictionaryComparison comparison() const {return comparison_;}
    DictionarySections sections(const std::vector<std::size_t>& indices,
        const std::vector<std::size_t>& origins,const std::vector<std::size_t>& targets,
        const std::vector<std::size_t>& fixed,std::size_t candidates,std::size_t remaining) const {
        using namespace rfc;
        using algebra::owned_gluing_detail::add;
        using algebra::owned_gluing_detail::mul;
        if(!candidates)throw std::invalid_argument("joint dictionary candidates must be positive");
        if(!remaining)throw std::length_error("joint dictionary work admission exceeded");
        auto bounds=owner_->limits();bounds.steps=std::min(bounds.steps,remaining);
        numeric::ExactActionGuard guard(bounds);std::size_t nested{},retained=retained_;
        const auto step=[&] {
            if(nested>=bounds.steps || guard.steps()>=bounds.steps-nested)
                throw std::length_error("joint dictionary work admission exceeded");
            guard.step();
        };
        const auto words=add(retained_,mul(add(indices.size(),add(origins.size(),fixed.size())),32));
        guard.entries(add(words,mul(owner_->source_words(),3)));
        std::set<std::size_t> distinct;
        for(const auto i:indices) {
            step();if(i>=comparison_.pairs.size() || !distinct.insert(i).second)
                throw std::invalid_argument("joint dictionary pair indices must be distinct catalog positions");
        }
        if(origins.empty() && (indices.empty() || !targets.empty() || !fixed.empty()))
            throw std::invalid_argument("joint dictionary needs a seed map or record pairs");
        DictionarySections result;result.domain_count=1;
        for(const auto i:indices)if(comparison_.pairs[i].licenses.empty())result.domain_count=0;
        if(result.domain_count)for(const auto i:indices) {
            step();const auto n=comparison_.pairs[i].licenses.size();
            if(result.domain_count>candidates/n)throw std::length_error("joint dictionary candidate admission exceeded");
            result.domain_count*=n;
        }
        GuardMatch seed;
        if(!origins.empty()) {
            seed=owner_->match(origins,targets,fixed,bounds.steps-guard.steps()-nested);
            nested=add(nested,seed.steps);++result.native_calls;
            if(!seed.matched) {result.steps=add(guard.steps(),nested);return result;}
        }
        for(std::size_t ordinal=0;ordinal<result.domain_count;++ordinal) {
            step();++result.candidate_count;
            std::map<std::size_t,std::size_t> cells,inverse_cells,rows,inverse_rows;
            const auto merge=[&](auto& map,auto& inverse,const std::vector<std::vector<std::size_t>>& values) {
                for(const auto& v:values) {
                    step();const auto [a,new_a]=map.emplace(v[0],v[1]);
                    const auto [b,new_b]=inverse.emplace(v[1],v[0]);
                    if((!new_a && a->second!=v[1]) || (!new_b && b->second!=v[0]))return false;
                }
                return true;
            };
            (void)merge(cells,inverse_cells,seed.cells);(void)merge(rows,inverse_rows,seed.relations);
            DictionarySection section;section.fixed=fixed;std::size_t value=ordinal;bool compatible=true;
            for(const auto i:indices) {
                step();const auto& pair=comparison_.pairs[i];
                const auto choice=value%pair.licenses.size();value/=pair.licenses.size();section.choices.push_back(choice);
                const auto& route=pair.licenses[choice];
                if(!merge(cells,inverse_cells,route.witness.cells) || !merge(rows,inverse_rows,route.witness.relations)) {
                    compatible=false;break;
                }
                if(std::find(section.fixed.begin(),section.fixed.end(),route.synset)==section.fixed.end())section.fixed.push_back(route.synset);
            }
            if(!compatible) {++result.overlap_rejections;continue;}
            for(const auto& [a,b]:rows) {step();section.left.push_back(a);section.right.push_back(b);}
            const auto section_words=add(16,add(mul(section.choices.size(),2),
                add(mul(rows.size(),10),add(mul(cells.size(),4),mul(section.fixed.size(),2)))));
            guard.entries(add(add(add(words,retained-retained_),section_words),mul(owner_->source_words(),3)));
            section.witness=owner_->match(section.left,section.right,section.fixed,bounds.steps-guard.steps()-nested);
            nested=add(nested,section.witness.steps);++result.native_calls;
            if(!section.witness.matched)throw std::logic_error("joint dictionary overlap accepted an invalid whole map");
            retained=add(retained,section_words);guard.entries(retained);result.sections.push_back(std::move(section));
        }
        result.steps=add(guard.steps(),nested);return result;
    }
};
} // namespace rexgraph::python
