#include "bridge.hpp"
#include "guard_bridge.hpp"
#include <future>
#include <iostream>

namespace {
std::size_t checks{};
void check(bool value) {++checks;if(!value)throw std::runtime_error("context accession check failed");}
template<class F> void rejects(F action) {
    bool rejected{};try {action();}catch(const std::exception&) {rejected=true;}check(rejected);
}
rexgraph::python::SourceDeclaration declaration(bool target) {
    using namespace rexgraph::python;
    SourceDeclaration d;d.name_space=target?981252:981251;d.max_grade=1;d.semantic_scale=2;
    d.role_aspects=target?std::vector<std::string>{"unused","literal-record","encoded-use"}:
        std::vector<std::string>{"encoded-use","literal-record"};
    if(target)d.types.push_back({"unused",0,0});
    d.types.push_back({"use-encoding",0,0});
    d.witnesses={0,1,2,3};
    for(std::uint64_t i=0;i<2;++i) {
        SourceRelation r;r.grade=1;r.key=i+1;r.type=target?1:0;
        r.occurrences={{2*i+1,i,true,false,"1",{target?2u:0u}},
            {2*i+2,2,true,false,"1",{1}}};
        d.relations.push_back(std::move(r));
    }
    return d;
}
}
int main() {
    using namespace rexgraph::python;
    const std::vector<std::size_t> limits{1000000,64000000,16384};
    const auto prepare=[&](const SourceDeclaration& d) {
        const auto source=Source::build(d,1u<<24u);
        return std::make_shared<RelationGuards>(source->info().archive,limits,1u<<24u);
    };
    const auto left=prepare(declaration(false)),right=prepare(declaration(true));
    const auto mapped=left->encoding_accession(*right,{0,1},{0,1},64000000);
    check(mapped.matched);check(mapped.cells.size()==3);check(mapped.relations.size()==2);
    check(mapped.occurrences.size()==4);check(mapped.cells==std::vector<std::vector<std::size_t>>{{0,0},{1,1},{2,2}});
    check(left->encoding_accession(*right,{0,1},{0,1},mapped.steps).matched);
    rejects([&]{left->encoding_accession(*right,{0,1},{0,1},mapped.steps-1);});
    rejects([&]{left->encoding_accession(*right,{},{},64000000);});
    rejects([&]{left->encoding_accession(*right,{0},{0,1},64000000);});
    rejects([&]{left->encoding_accession(*right,{0},{0},0);});
    rejects([&]{left->encoding_accession(*right,{0,0},{0,1},64000000);});
    rejects([&]{left->encoding_accession(*right,{0,1},{0,0},64000000);});
    rejects([&]{left->encoding_accession(*right,{2},{0},64000000);});
    for(const unsigned fault:{0u,1u,2u,3u,4u,5u,6u,7u,8u}) {
        auto d=declaration(true);auto& r=d.relations[0];
        if(fault==0)r.occurrences[0].exists=false;
        else if(fault==1)r.occurrences[0].negative=true;
        else if(fault==2)r.occurrences[0].magnitude="0";
        else if(fault==3)r.occurrences[0].magnitude="2";
        else if(fault==4)r.occurrences[0].aspects={1};
        else if(fault==5)d.types[1].name="different-encoding";
        else if(fault==6)d.semantic_scale=3;
        else if(fault==7)d.relations[1].occurrences[1].lower=3;
        else d.relations[1].occurrences[0].lower=0;
        const auto failed=left->encoding_accession(*prepare(d),{0,1},{0,1},64000000);
        check(!failed.matched);check(failed.cells.empty() && failed.relations.empty() && failed.occurrences.empty());
    }
    auto reordered=declaration(true);std::swap(reordered.relations[0].occurrences[0],reordered.relations[0].occurrences[1]);
    check(!left->encoding_accession(*prepare(reordered),{0},{0},64000000).matched);
    std::vector<std::future<GuardMatch>> readers;
    for(std::size_t i=0;i<8;++i)readers.push_back(std::async(std::launch::async,[&] {
        return left->encoding_accession(*right,{0,1},{0,1},64000000);
    }));
    for(auto& task:readers) {const auto value=task.get();check(value.matched && value.steps==mapped.steps && value.cells==mapped.cells);}
    check(left->encoding_accession(*right,{0,1},{0,1},64000000).matched);
    std::cout<<checks<<" exact context accession checks passed\n";
}
