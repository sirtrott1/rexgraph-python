#include "chain_bridge.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
std::size_t checks = 0;
void require(bool value) { ++checks; if (!value) throw std::runtime_error("chain bridge assertion failed"); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
std::shared_ptr<rexgraph::python::Source> load(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    const std::string image{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    require(!image.empty());
    return rexgraph::python::Source::load(image.data(), image.size(), 1u << 20u);
}
}
int main(int argc, char** argv) {
    using namespace rexgraph::python;
    if (argc != 3) throw std::invalid_argument("chain bridge test requires chart and pattern archives");
    auto source = load(argv[1]), target = load(argv[2]);
    ChainFamily family;
    for (const auto width : std::array<std::uint32_t,3>{4,4,1}) {
        std::vector<std::vector<ChainTerm>> row;
        for (std::uint32_t i=0; i<width; ++i) row.push_back({{i,"5/8"},{width+i,"3/8"},{2*width+i,"0"}});
        family.push_back(std::move(row));
    }
    const std::vector<std::size_t> limits{1000000,64000000,16384};
    auto map=ChainMap::prepare(source,target,family,8,limits);
    for (std::uint16_t k=0;k<3;++k) {
        const auto width=family[k].size();
        for (std::size_t i=0;i<width;++i) {
            std::vector<std::string> q(width,"0");q[i]="1";
            std::vector<std::int64_t> n(width,0);n[i]=1;
            const auto rational=map->push(k,q);
            const auto integer=map->push_integer(k,n);
            require(rational.size()==3*width && integer.size()==rational.size());
            for (std::size_t j=0;j<3*width;++j) {
                const auto expected=j==i?"5/8":j==width+i?"3/8":"0";
                require(rational[j]==expected);
                require(integer[j]==(j==i?5:j==width+i?3:0));
            }
        }
    }
    rejects([&]{auto incomplete=family;incomplete.pop_back();ChainMap::prepare(source,target,incomplete,8,limits);});
    rejects([&]{auto broken=family;broken[2][0].clear();ChainMap::prepare(source,target,broken,8,limits);});
    rejects([&]{ChainMap::prepare(source,target,family,1,limits);});
    rejects([&]{map->push_integer(1,{0,0,INT64_MAX,0});});
    rejects([&]{map->push(3,{});});
    rejects([&]{map->push(1,{"1"});});
    rejects([&]{map->push(1,{"0","0","1/0","0"});});
    rejects([&]{ChainMap::prepare(source,target,family,8,{1,1,1});});
    source.reset();target.reset();
    require(map->push_integer(2,{1})==std::vector<std::int64_t>({5,3,0}));
    std::vector<std::vector<std::uint64_t>> initial(17,std::vector<std::uint64_t>(3));
    std::vector<std::uint32_t> features,labels;
    auto reference=initial;
    for(std::uint32_t i=0;i<10000;++i){
        features.push_back(i%17);labels.push_back(i%3);++reference[i%17][i%3];
    }
    require(observe_context_counts(initial,features,labels,10000)==reference);
    require(initial==std::vector<std::vector<std::uint64_t>>(17,std::vector<std::uint64_t>(3)));
    rejects([&]{observe_context_counts(initial,{17},{0},10000);});
    rejects([&]{observe_context_counts(initial,{0},{3},10000);});
    rejects([&]{observe_context_counts(initial,{0},{},10000);});
    rejects([&]{observe_context_counts({{0},{0,0}},{0},{0},10000);});
    rejects([&]{observe_context_counts(initial,features,labels,9999);});
    rejects([&]{observe_context_counts({{UINT64_MAX}},{0},{0},UINT64_MAX);});
    require(observe_context_counts({{UINT64_MAX}},{},{},UINT64_MAX)==std::vector<std::vector<std::uint64_t>>{{UINT64_MAX}});
    std::cout << "{\"assertions\":" << checks << "}\n";
}
