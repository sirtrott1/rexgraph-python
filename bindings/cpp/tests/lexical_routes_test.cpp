#include "bridge.hpp"
#include "lexical_routes_bridge.hpp"
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
std::size_t checks{};
void check(bool value) {++checks;if(!value)throw std::runtime_error("lexical route check failed");}
template<class F> void rejects(F action) {
    bool rejected{};try {action();}catch(const std::exception&) {rejected=true;}check(rejected);
}
rexgraph::python::SourceDeclaration declaration() {
    using namespace rexgraph::python;
    SourceDeclaration d;d.name_space=981245;d.max_grade=1;d.semantic_scale=2;
    d.role_aspects={"owner","text-use","encoded-use","literal-record","sense","synset","member"};
    d.types={{"record-lemma",0,0},{"use-encoding",0,0},{"entry-sense",0,0},{"sense-synset",0,0},
        {"record-form",0,0},{"form-entry",0,0},{"record-definition",0,0}};
    for(std::size_t i=0;i<10;++i)d.witnesses.push_back(i);
    const auto row=[&](unsigned type,std::uint64_t a,std::uint64_t b,unsigned role_a,unsigned role_b,bool negative) {
        SourceRelation r;r.grade=1;r.key=d.relations.size()+1;r.type=type;
        r.occurrences={{2*r.key-1,a,true,negative,"1",{role_a}},{2*r.key,b,true,false,"1",{role_b}}};
        d.relations.push_back(std::move(r));
    };
    row(0,0,1,0,1,true);row(1,1,2,2,3,false);row(2,0,3,0,4,true);row(3,3,4,0,5,true);
    row(2,0,5,0,4,true);row(3,5,6,0,5,true);row(4,7,8,0,1,true);row(1,8,9,2,3,false);
    row(5,7,0,0,6,true);row(3,3,4,0,5,true);row(6,4,1,0,1,true);return d;
}
}
int main() {
    using namespace rexgraph::python;
    const std::vector<std::size_t> limits{1000000,64000000,16384};
    const auto prepare=[&](const SourceDeclaration& d) {
        const auto source=Source::build(d,1u<<24u);
        return std::make_shared<RelationGuards>(source->info().archive,limits,1u<<24u);
    };
    const std::vector<LexicalSpelling> spellings{{0,"bank",{1},{1},{0},{4}}, {6,"banks",{8},{7},{0},{5}}};
    auto owner=prepare(declaration());LexicalRoutes catalog(owner,spellings);owner.reset();
    check(catalog.routes().size()==6);
    const auto found=catalog.lookup("bank banks bank",{0,5,11},{4,10,15},9,64000000);
    check(found.uses.size()==3);check(found.uses[0].routes==found.uses[2].routes);
    check(found.uses[0].routes.size()==3);check(found.uses[1].routes.size()==3);
    check(catalog.routes()[0].rows.size()==4);
    check(catalog.routes()[0].slots.size()==8);check(catalog.routes()[1].synset==catalog.routes()[0].synset);
    check(catalog.lookup("Bank",{0},{4},9,64000000).uses.empty());
    rejects([&]{catalog.lookup("bank banks bank",{0,5,11},{4,10,15},8,64000000);});
    rejects([&]{catalog.lookup("bank banks bank",{0,5,11},{4,10,15},9,found.steps-1);});
    check(catalog.lookup("bank banks bank",{0,5,11},{4,10,15},9,found.steps).uses.size()==3);
    rejects([&]{catalog.lookup("bank",{0,1},{2,4},9,64000000);});
    rejects([&]{catalog.lookup("bank",{0},{0},9,64000000);});
    rejects([&]{catalog.lookup("bank",{0},{5},9,64000000);});
    rejects([&]{catalog.lookup("bank",{0},{4},0,64000000);});
    rejects([&]{catalog.lookup("bank",{0},{4},9,0);});
    rejects([&]{catalog.lookup("",{0},{1},9,64000000);});
    rejects([&]{catalog.lookup(std::string(8193,'x'),{0},{4},9,64000000);});
    rejects([&]{LexicalRoutes(nullptr,spellings);});
    rejects([&]{LexicalRoutes(prepare(declaration()),{});});
    auto subset=declaration();subset.relations={subset.relations.back()};subset.witnesses={1,4};subset.name_space=981246;
    const auto binding=catalog.bind_patterns(prepare(subset));
    check(binding.size()==1 && binding[0].pattern==0 && binding[0].synset==4);
    for(const unsigned fault:{0u,1u,2u,3u,4u,5u,6u}) {
        auto d=subset;auto& r=d.relations[0];
        if(fault==0)r.key=50;else if(fault==1)r.occurrences[0].key=100;
        else if(fault==2)r.occurrences[0].magnitude="2";else if(fault==3)r.occurrences[0].negative=false;
        else if(fault==4)r.occurrences[0].exists=false;else if(fault==5)r.occurrences[1].aspects={6};
        else d.semantic_scale=3;
        rejects([&]{catalog.bind_patterns(prepare(d));});
    }
    for(const bool zero:{false,true}) {
        auto d=declaration();auto& flag=d.relations[2].occurrences[1];
        if(zero)flag.magnitude="0";else flag.exists=false;
        LexicalRoutes partial(prepare(d),spellings);
        check(partial.routes().size()==2);check(partial.lookup("bank",{0},{4},9,64000000).uses[0].routes.size()==1);
    }
    for(const unsigned fault:{0u,1u,2u}) {
        auto d=declaration();auto& flag=d.relations[2].occurrences[1];
        if(fault==0)flag.negative=true;else if(fault==1)flag.magnitude="2";else flag.aspects={1};
        rejects([&]{LexicalRoutes(prepare(d),spellings);});
    }
    for(const unsigned fault:{0u,1u,2u,3u,4u}) {
        auto values=spellings;
        if(fault==0)values[0].uses={8};else if(fault==1)values[0].encodings={7};
        else if(fault==2)values[0].ends={5};else if(fault==3)values[0].row=2;else values.push_back(values[0]);
        rejects([&]{LexicalRoutes(prepare(declaration()),values);});
    }
    std::vector<std::future<std::size_t>> threads;
    for(std::size_t i=0;i<16;++i)threads.push_back(std::async(std::launch::async,[&]{return catalog.lookup("bank",{0},{4},9,64000000).uses[0].routes.size();}));
    for(auto& result:threads)check(result.get()==3);
    auto retained=std::make_shared<LexicalRoutes>(prepare(declaration()),spellings);
    const std::string digest(64,'a');
    LexicalContextFields annotations(retained,digest,true,true,true,false,{});
    const auto field=annotations.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,64000000);
    check(field.axes.front()=="position:0:observed-upos:NOUN");
    check(std::find(field.axes.begin(),field.axes.end(),"position:1:observed-upos:VERB")!=field.axes.end());
    LexicalContextFields bound(retained,digest,true,true,true,true,field.axes);
    const auto mapped=bound.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,64000000);
    check(mapped.missing.empty() && mapped.entries.size()==field.axes.size());
    const auto unknown=bound.lookup("bank banks",{0,5},{4,10},{"ADJ","VERB"},9,64000000);
    check(unknown.missing==std::vector<std::string>{"position:0:observed-upos:ADJ"});
    check(annotations.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,field.steps).axes==field.axes);
    rejects([&]{annotations.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,field.steps-1);});
    rejects([&]{annotations.lookup("bank banks",{0,5},{4,10},{"NOUN"},9,64000000);});
    rejects([&]{annotations.lookup("bank banks",{0,5},{4,10},{"NOUN","BAD"},9,64000000);});
    LexicalContextFields lexical(retained,digest,true,true,false,false,{});
    rejects([&]{lexical.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,64000000);});
    check(lexical.lookup("bank banks",{0,5},{4,10},{},9,64000000).axes.size()+2==field.axes.size());
    std::vector<std::future<std::vector<std::string>>> fields;
    for(std::size_t i=0;i<8;++i)fields.push_back(std::async(std::launch::async,[&]{
        return annotations.lookup("bank banks",{0,5},{4,10},{"NOUN","VERB"},9,64000000).axes;}));
    for(auto& result:fields)check(result.get()==field.axes);
    std::cout<<checks<<" exact native checks\n";
}
