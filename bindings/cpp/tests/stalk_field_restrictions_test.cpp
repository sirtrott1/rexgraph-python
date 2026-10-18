#include "bridge.hpp"
#include "stalk_parameter_relations_bridge.hpp"
#include <future>
#include <iostream>

namespace {
std::size_t checks{};
void check(bool value) {++checks;if(!value)throw std::runtime_error("whole-field restriction check failed");}
template<class F> void rejects(F action) {
    bool rejected{};try {action();}catch(const std::exception&) {rejected=true;}check(rejected);
}
auto transport(bool fixed=false) {
    using namespace rexgraph::python;
    SourceDeclaration d;d.name_space=981270;d.max_grade=1;d.semantic_scale=2;
    d.role_aspects={"role"};d.types.push_back({"column",0,0});d.witnesses={0};
    d.relations={{1,1,0,{{1,0,true,false,"2/3",{0}}}},{1,2,0,{{2,0,true,true,"3/2",{0}}}}};
    const auto source=Source::build(d,1u<<24u);
    const std::vector<std::vector<std::vector<std::string>>> axes{{{"meaning"}},{{"input"},{"input"}}};
    std::vector<std::vector<std::vector<BoundaryMapEntry>>> maps{{},{{},{}}};
    std::vector<TransportSlot> slots{{0,0,0},{1,0,0}};
    std::vector<TransportObservation> observations{{{"1","1"},{"1"},{1}}};
    if(fixed) {maps[1][0]={{0,0,"2"}};maps[1][1]={{0,0,"-1"}};slots.clear();observations.clear();}
    return std::make_shared<StalkTransportFit>(source->info().archive,axes,maps,slots,observations,
        std::vector<std::size_t>{4096,16000000,64000000,16384},1u<<24u);
}
}
int main() {
    using namespace rexgraph::python;
    const auto fit=transport();const auto before=fit->receipt();
    const StalkParameterRelations owner(fit,{{"unrestricted",{},"0"}},true);
    const TransportObservation sample{{"1","0"},{"2"},{1}};
    const auto open=owner.restrict_fields({}, {sample});
    check(open.status==2 && open.base_consistent && open.family.consistent);
    check(open.additional_rank==1 && open.family.particular==std::vector<std::string>{"3","2/3"});
    check(open.family.kernel.empty() && open.family.free_columns.empty());
    check(open.constraints==std::vector<std::vector<std::size_t>>{{0,0,0}});
    check(open.contradiction.empty());
    check(owner.restrict_fields({},{{{"1","1"},{"1"},{1}}}).status==0);
    const auto excluded=owner.restrict_fields({},{{{"1","1"},{"2"},{1}}});
    check(excluded.status==1 && !excluded.family.consistent);
    check(excluded.contradiction==std::vector<std::string>{"1"} && excluded.contradiction_value=="1");
    const auto joint=owner.restrict_fields({}, {sample,{{"1","0"},{"3"},{1}}});
    check(joint.status==1 && joint.constraints.size()==2 && joint.contradiction.size()==2);
    check(joint.contradiction_value!="0");
    const StalkParameterRelations selected(fit,{{"first-zero",{{0,"1"}},"0"},{"first-three",{{0,"1"}},"3"}});
    check(selected.restrict_fields({1},{sample}).status==0);
    check(selected.restrict_fields({0},{sample}).status==1);
    check(selected.restrict_fields({0,1},{sample}).status==3);
    check(selected.restrict_fields({0,1},{sample}).contradiction.empty());
    const StalkParameterRelations fixed(transport(true),{{"unrestricted",{},"0"}},true);
    check(fixed.restrict_fields({}, {{{"1","0"},{"4/3"},{1}}}).status==0);
    check(fixed.restrict_fields({}, {{{"1","0"},{"3"},{1}}}).status==1);
    check(fit->receipt().particular==before.particular && fit->receipt().kernel.size()==before.kernel.size());
    rejects([&]{owner.restrict_fields({},{});});
    rejects([&]{owner.restrict_fields({0,0},{sample});});
    rejects([&]{owner.restrict_fields({1},{sample});});
    rejects([&]{owner.restrict_fields({},{{{"1"},{"2"},{1}}});});
    rejects([&]{owner.restrict_fields({},{{{"1","0"},{"2"},{2}}});});
    rejects([&]{owner.restrict_fields({},{{{"1","0"},{"2"},{0}}});});
    rejects([&]{owner.restrict_fields({},{{{"1","0"},{"2","0"},{1}}});});
    rejects([&]{owner.restrict_fields({},{{{"1","0"},{"not-rational"},{1}}});});
    std::vector<std::future<FieldConditionReceipt>> jobs;
    for(std::size_t i=0;i<8;++i)jobs.push_back(std::async(std::launch::async,[&]{return owner.restrict_fields({}, {sample});}));
    for(auto& job:jobs) {const auto result=job.get();check(result.family.particular==open.family.particular && result.family.steps==open.family.steps);}
    check(owner.restrict_fields({}, {sample}).family.particular==open.family.particular);
    std::cout<<checks<<" exact whole-field restriction checks passed\n";
}
