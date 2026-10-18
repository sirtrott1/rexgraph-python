#pragma once
#include "guard_bridge.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace rexgraph::python {
struct LexicalSpelling {
    std::size_t row{};
    std::string raw;
    std::vector<std::size_t> uses, encodings, begins, ends;
};
struct LexicalRoute {
    std::size_t spelling{}, owner{}, entry{}, sense{}, synset{};
    std::vector<std::size_t> rows, slots;
    std::vector<std::uint64_t> occurrence_keys;
};
struct LexicalUse {
    std::size_t first{}, last{}, begin{}, end{};
    std::vector<std::size_t> routes;
};
struct LexicalLookup {
    std::vector<LexicalUse> uses;
    std::size_t steps{};
};
struct LexicalPattern {std::size_t pattern{}, synset{};};

class LexicalRoutes {
    friend class LexicalContextFields;
    struct Link {std::size_t row{}, target{};};
    struct Trie {std::map<unsigned char,std::size_t> next;std::vector<std::size_t> routes;};
    std::shared_ptr<const RelationGuards> owner_;
    std::vector<LexicalRoute> routes_;
    std::vector<std::vector<std::pair<std::size_t,std::size_t>>> partitions_;
    std::vector<Trie> trie_{1};
    std::map<std::uint64_t,std::size_t> original_rows_;
    std::size_t words_{}, steps_{};
    std::size_t entry_limit_, work_limit_;
    static void flag(const GuardOccurrence& f,const char* role,bool negative) {
        if(f.roles!=std::vector<std::string>{role} || f.negative!=negative)
            throw std::invalid_argument("lexical route differs from ordered role and orientation");
        if(f.exists && f.coefficient!="0" && f.coefficient!=(negative?"-1":"1"))
            throw std::invalid_argument("lexical route requires unit occurrence magnitude");
    }
    static bool active(const GuardOccurrence& f) {return f.exists && f.coefficient!="0";}
    void step() {
        if(steps_==work_limit_)throw std::length_error("lexical preparation work admission exceeded");
        ++steps_;
    }
    void retain(std::size_t n) {
        words_=rfc::machine::checked_size_add(words_,n);
        if(words_>entry_limit_)throw std::length_error("lexical catalog entry admission exceeded");
    }
public:
    LexicalRoutes(const std::shared_ptr<RelationGuards>& owner,const std::vector<LexicalSpelling>& spellings)
        :owner_(owner),entry_limit_(owner?owner->limits().entries:0),work_limit_(owner?owner->limits().steps:0) {
        if(!owner_ || spellings.empty())throw std::invalid_argument("lexical catalog needs an owning source and spellings");
        retain(owner_->source_words());
        std::map<std::size_t,std::vector<Link>> senses, synsets, forms;
        for(std::size_t row=0;row<owner_->relation_count();++row) {
            step();const auto column=owner_->boundary(row);
            retain(2);original_rows_.emplace(owner_->relation_key(row),row);
            auto* index=column.type=="entry-sense"?&senses:column.type=="sense-synset"?&synsets:
                column.type=="form-entry"?&forms:nullptr;
            if(!index)continue;
            const char* role=column.type=="entry-sense"?"sense":column.type=="sense-synset"?"synset":"member";
            if(column.occurrences.size()!=2)throw std::invalid_argument("lexical route needs a complete binary record column");
            flag(column.occurrences[0],"owner",true);flag(column.occurrences[1],role,false);
            step();step();
            if(active(column.occurrences[0]) && active(column.occurrences[1])) {
                retain(4);(*index)[column.occurrences[0].lower].push_back({row,column.occurrences[1].lower});
            }
        }
        std::set<std::size_t> spelling_rows, uses, encodings;
        std::map<std::size_t,std::string> literals;
        for(std::size_t i=0;i<spellings.size();++i) {
            step();const auto& spelling=spellings[i];
            if(spelling.raw.empty() || spelling.raw.size()>8192 || spelling.uses.empty() || spelling.uses.size()>128 ||
               spelling.encodings.size()!=spelling.uses.size() || spelling.begins.size()!=spelling.uses.size() ||
               spelling.ends.size()!=spelling.uses.size() || !spelling_rows.insert(spelling.row).second)
                throw std::invalid_argument("distinct bounded complete lexical spelling required");
            retain(spelling.raw.size());
            retain(3);
            partitions_.emplace_back();
            const auto column=owner_->boundary(spelling.row);
            if((column.type!="record-lemma" && column.type!="record-form") || column.occurrences.size()!=spelling.uses.size()+1)
                throw std::invalid_argument("lexical spelling differs from its original text column");
            flag(column.occurrences[0],"owner",true);bool enabled=active(column.occurrences[0]);
            std::size_t previous{};
            for(std::size_t j=0;j<spelling.uses.size();++j) {
                step();const auto& f=column.occurrences[j+1];flag(f,"text-use",false);enabled=enabled && active(f);
                if(f.lower!=spelling.uses[j] || !uses.insert(f.lower).second ||
                    !encodings.insert(spelling.encodings[j]).second || spelling.begins[j]<previous ||
                    spelling.begins[j]>=spelling.ends[j] || spelling.ends[j]>spelling.raw.size())
                    throw std::invalid_argument("lexical spelling differs from ordered original uses");
                previous=spelling.ends[j];const auto encoding=owner_->boundary(spelling.encodings[j]);
                partitions_.back().emplace_back(spelling.begins[j],spelling.ends[j]);
                if(encoding.type!="use-encoding" || encoding.occurrences.size()!=2 || encoding.occurrences[0].lower!=f.lower)
                    throw std::invalid_argument("lexical spelling lost its original use encoding");
                flag(encoding.occurrences[0],"encoded-use",false);flag(encoding.occurrences[1],"literal-record",false);
                if(!active(encoding.occurrences[0]) || !active(encoding.occurrences[1]))
                    throw std::invalid_argument("lexical spelling requires active literal accession");
                const auto bytes=spelling.raw.substr(spelling.begins[j],spelling.ends[j]-spelling.begins[j]);
                const auto [it,inserted]=literals.emplace(encoding.occurrences[1].lower,bytes);
                if(!inserted && it->second!=bytes)throw std::invalid_argument("shared literal identity has conflicting bytes");
                retain(10);
            }
            if(!enabled)continue;
            const auto owner_cell=column.occurrences[0].lower;
            std::size_t terminal{};
            for(const char value:spelling.raw) {
                const auto byte=static_cast<unsigned char>(value);
                step();const auto found=trie_[terminal].next.find(byte);
                if(found==trie_[terminal].next.end()) {
                    retain(4);const auto next=trie_.size();trie_[terminal].next.emplace(byte,next);
                    trie_.push_back({});terminal=next;
                } else terminal=found->second;
            }
            const auto append=[&](std::size_t entry,const std::vector<std::size_t>& prefix) {
                for(const auto a:senses[entry])for(const auto b:synsets[a.target]) {
                    step();LexicalRoute route;route.spelling=i;route.owner=owner_cell;route.entry=entry;
                    route.sense=a.target;route.synset=b.target;route.rows=prefix;
                    route.rows.push_back(a.row);route.rows.push_back(b.row);
                    for(const auto row:route.rows)for(const auto& f:owner_->boundary(row).occurrences) {
                        step();route.slots.push_back(f.slot);route.occurrence_keys.push_back(f.key);
                    }
                    retain(8+route.rows.size()+2*route.slots.size());
                    const auto id=routes_.size();routes_.push_back(std::move(route));
                    retain(1);trie_[terminal].routes.push_back(id);
                }
            };
            std::vector<std::size_t> prefix{spelling.row};
            prefix.insert(prefix.end(),spelling.encodings.begin(),spelling.encodings.end());
            if(column.type=="record-lemma")append(owner_cell,prefix);
            else for(const auto form:forms[owner_cell]) {
                step();auto path=prefix;path.push_back(form.row);append(form.target,path);
            }
        }
    }
    std::size_t preparation_steps() const {return steps_;}
    const std::vector<LexicalRoute>& routes() const {return routes_;}
    std::vector<LexicalPattern> bind_patterns(const std::shared_ptr<RelationGuards>& patterns) const {
        if(!patterns)throw std::invalid_argument("pattern source owner required");
        rfc::numeric::ExactActionGuard guard(owner_->limits());std::vector<LexicalPattern> result;
        guard.entries(rfc::machine::checked_size_add(words_,patterns->source_words()));
        for(std::size_t i=0;i<patterns->relation_count();++i) {
            guard.step();const auto found=original_rows_.find(patterns->relation_key(i));
            if(found==original_rows_.end())throw std::invalid_argument("pattern outside original dictionary relations");
            owner_->verify_original_row(*patterns,found->second,i,guard);
            const auto column=owner_->boundary(found->second);
            if(column.type=="record-definition" || column.type=="record-example") {
                if(column.occurrences.empty())throw std::invalid_argument("original text pattern needs its owner");
                flag(column.occurrences[0],"owner",true);
                if(active(column.occurrences[0]))result.push_back({i,column.occurrences[0].lower});
            }
        }
        return result;
    }
    LexicalLookup lookup(const std::string& raw,const std::vector<std::size_t>& begins,
                         const std::vector<std::size_t>& ends,std::size_t candidate_limit,std::size_t remaining) const {
        if(raw.empty() || raw.size()>8192 || begins.empty() || begins.size()>128 || ends.size()!=begins.size() ||
            !candidate_limit || !remaining)throw std::invalid_argument("bounded original query intervals required");
        remaining=std::min(remaining,work_limit_);LexicalLookup result;std::size_t count{},previous{},stored=words_;
        const auto tick=[&] {if(result.steps==remaining)throw std::length_error("lexical lookup work admission exceeded");++result.steps;};
        for(std::size_t i=0;i<begins.size();++i) {
            tick();if(begins[i]<previous || begins[i]>=ends[i] || ends[i]>raw.size())
                throw std::invalid_argument("ordered disjoint lexical query intervals required");
            previous=ends[i];
        }
        for(std::size_t i=0;i<begins.size();++i) {
            std::size_t node{},last=i;
            for(std::size_t pos=begins[i];pos<ends.back();++pos) {
                tick();const auto it=trie_[node].next.find(static_cast<unsigned char>(raw[pos]));
                if(it==trie_[node].next.end())break;
                node=it->second;
                while(last<ends.size() && ends[last]<pos+1) {tick();++last;}
                if(last<ends.size() && ends[last]==pos+1 && !trie_[node].routes.empty()) {
                    const auto n=trie_[node].routes.size();
                    if(n>candidate_limit-count)throw std::length_error("lexical route candidate admission exceeded");
                    count+=n;stored=rfc::machine::checked_size_add(stored,rfc::machine::checked_size_add(n,5));
                    if(stored>entry_limit_)throw std::length_error("lexical lookup entry admission exceeded");
                    result.uses.push_back({i,last,begins[i],pos+1,trie_[node].routes});
                }
            }
        }
        return result;
    }
};

struct LexicalContextField {
    LexicalLookup lookup;
    std::vector<std::string> axes, missing;
    std::vector<std::size_t> entries;
    std::size_t steps{};
};

class LexicalContextFields {
    std::shared_ptr<const LexicalRoutes> routes_;
    std::string digest_;
    bool kinds_{}, phrases_{}, annotations_{}, bound_{};
    std::vector<std::string> kinds_by_route_;
    std::map<std::string,std::size_t> frame_;
    std::size_t retained_{};
    static std::string hexadecimal(const std::string& bytes) {
        constexpr char digits[]="0123456789abcdef";
        std::string result;result.reserve(bytes.size()*2);
        for(const char value:bytes) {
            const auto byte=static_cast<unsigned char>(value);
            result.push_back(digits[byte>>4]);result.push_back(digits[byte&15]);
        }
        return result;
    }
public:
    LexicalContextFields(const std::shared_ptr<LexicalRoutes>& routes,const std::string& digest,
                         bool kinds,bool phrases,bool annotations,bool bound,const std::vector<std::string>& axes)
        :routes_(routes),digest_(digest),kinds_(kinds),phrases_(phrases),annotations_(annotations),bound_(bound) {
        if(!routes_ || digest_.size()!=64 || digest_.find_first_not_of("0123456789abcdef")!=std::string::npos ||
                (!bound_ && !axes.empty()) || (bound_ && axes.empty()))
            throw std::invalid_argument("owned lexical routes and a declared context field required");
        retained_=routes_->words_;
        const auto retain=[&](std::size_t count) {
            retained_=rfc::machine::checked_size_add(retained_,count);
            if(retained_>routes_->entry_limit_)throw std::length_error("lexical field entry admission exceeded");
        };
        for(std::size_t i=0;i<axes.size();++i) {
            if(axes[i].empty() || axes[i].size()>256 || !frame_.emplace(axes[i],i).second)
                throw std::invalid_argument("distinct bounded context directions required");
            retain(2+(axes[i].size()+7)/8);
        }
        if(kinds_)for(const auto& route:routes_->routes_) {
            const auto kind=routes_->owner_->boundary(route.rows.front()).type;
            retain(1+(kind.size()+7)/8);kinds_by_route_.push_back(kind);
        }
    }
    LexicalContextField lookup(const std::string& raw,const std::vector<std::size_t>& begins,
                              const std::vector<std::size_t>& ends,const std::vector<std::string>& annotations,
                              std::size_t candidates,std::size_t work) const {
        if((annotations_ && annotations.size()!=begins.size()) || (!annotations_ && !annotations.empty()))
            throw std::invalid_argument("complete declared original past annotations required");
        LexicalContextField result;
        result.lookup=routes_->lookup(raw,begins,ends,candidates,work);
        result.steps=result.lookup.steps;work=std::min(work,routes_->work_limit_);
        auto stored=retained_;
        const auto tick=[&] {
            if(result.steps==work)throw std::length_error("lexical field work admission exceeded");
            ++result.steps;
        };
        const auto retain=[&](std::size_t count) {
            stored=rfc::machine::checked_size_add(stored,count);
            if(stored>routes_->entry_limit_)throw std::length_error("lexical field entry admission exceeded");
        };
        std::vector<std::set<std::size_t>> single(begins.size());
        std::vector<std::set<std::string>> kinds(kinds_?begins.size():0);
        std::map<std::pair<std::size_t,std::size_t>,std::set<std::size_t>> phrases;
        retain(2*begins.size());
        for(const auto& use:result.lookup.uses)retain(5+use.routes.size());
        for(const auto& use:result.lookup.uses)for(const auto index:use.routes) {
            tick();const auto& route=routes_->routes_[index];
            const auto& partition=routes_->partitions_[route.spelling];
            bool matched=partition.size()==use.last-use.first+1;
            for(std::size_t i=0;matched && i<partition.size();++i) {
                tick();matched=partition[i]==std::make_pair(begins[use.first+i]-use.begin,ends[use.first+i]-use.begin);
            }
            if(!matched)continue;
            if(use.first==use.last) {
                if(single[use.first].insert(route.synset).second)retain(2);
                if(kinds_ && kinds[use.first].insert(kinds_by_route_[index]).second)retain(2);
            } else if(phrases_ && phrases[{use.first,use.last}].insert(route.synset).second)retain(4);
        }
        const auto append=[&](std::string axis) {
            tick();if(axis.empty() || axis.size()>256)
                throw std::invalid_argument("bounded original context direction required");
            retain(2+(axis.size()+7)/8);
            if(bound_) {
                const auto found=frame_.find(axis);
                if(found==frame_.end())result.missing.push_back(axis);
                else result.entries.push_back(found->second);
            }
            result.axes.push_back(std::move(axis));
        };
        for(std::size_t position=0;position<begins.size();++position) {
            const auto prefix="position:"+std::to_string(position)+":";
            if(annotations_) {
                tick();const auto& name=annotations[position];
                static const std::set<std::string> allowed{"ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ",
                    "NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"};
                if(!allowed.contains(name))throw std::invalid_argument("declared original UPOS annotation required");
                append(prefix+"observed-upos:"+name);
            }
            if(single[position].empty()) {
                for(auto i=begins[position];i<ends[position];++i)tick();
                append(prefix+"literal:"+hexadecimal(raw.substr(begins[position],ends[position]-begins[position])));
            } else {
                for(const auto key:single[position])append(prefix+"dictionary:"+digest_+":synset:"+std::to_string(key));
                if(kinds_)for(const auto& kind:kinds[position])append(prefix+"lexical-kind:"+kind);
            }
        }
        for(const auto& [interval,keys]:phrases)for(const auto key:keys)
            append("phrase:"+std::to_string(interval.first)+":"+std::to_string(interval.second)+
                ":dictionary:"+digest_+":synset:"+std::to_string(key));
        std::sort(result.entries.begin(),result.entries.end());
        return result;
    }
};
}
