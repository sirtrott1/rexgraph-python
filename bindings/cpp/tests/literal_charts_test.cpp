#include "chain_bridge.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
std::size_t checks=0;
void check(bool condition) {
    ++checks;
    if(!condition) throw std::runtime_error("literal chart assertion failed");
}
template<class F> void rejects(F action) {
    bool rejected=false;
    try {action();} catch(const std::exception&) {rejected=true;}
    check(rejected);
}
}

int main() {
    using namespace rexgraph::python;
    const std::vector<std::size_t> limits{1000000,64000000,16384};
    for(std::size_t count=1;count<=128;++count) {
        const std::string raw(count,'a');
        std::vector<std::string> words(count,"a");
        std::vector<LiteralTokenSpan> spans;
        for(std::size_t i=0;i<count;++i) spans.push_back({i,i+1});
        auto chart=literal_token_chart(raw,words,spans,920060,1u<<24u,limits);
        const auto tokens=chart.tokens->info(),bytes=chart.bytes->info();
        check(tokens.name_space==920061 && bytes.name_space==920060);
        check(tokens.coordinates[0].size()==2*count && tokens.coordinates[1].size()==3*count-2);
        check(tokens.coordinates[2].size()==count-1 && bytes.coordinates[2].empty());
        check(chart.bytes->field(1001)->info().values==std::vector<std::string>(count,"61"));
        check(chart.family[0].size()==2*count && chart.family[1].size()==3*count-2);
        check(chart.family[2].size()==count-1);
        if(count>1) {
            check(chart.family[0][1][0].target==chart.family[0][2][0].target);
            check(chart.family[1][count].empty());
        }
        std::vector<std::string> input(2*count,"0");input[0]="1";input.back()="-1";
        auto expected=std::vector<std::string>(count+1,"0");expected[0]="1";expected.back()="-1";
        check(chart.map->push(0,input)==expected);
        check(chart.map->push(2,std::vector<std::string>(count-1,"7")).empty());
        const auto image=chart.tokens->info().archive;
        check(Source::load(image.data(),image.size(),1u<<24u)->info().archive==image);
        auto owning=chart.map;chart.tokens.reset();chart.bytes.reset();
        check(owning->push(0,input)==expected);
    }
    const auto make=[&](std::string raw,std::vector<std::string> words,std::vector<LiteralTokenSpan> spans,
        std::uint64_t ns=920060,std::size_t bytes=1u<<24u,std::vector<std::size_t> policy={1000000,64000000,16384}) {
        return literal_token_chart(raw,words,spans,ns,bytes,policy);
    };
    auto supported=make(std::string("a\0",2),{"a",std::string(1,'\0')},{{0,1},{1,2}});
    check(supported.bytes->field(1001)->info().values==std::vector<std::string>({"61","0"}));
    rejects([&]{make("a",{"a"},{{1,0}});});
    rejects([&]{make("a",{"a"},{{0,2}});});
    rejects([&]{make("a",{"a"},{{0,0}});});
    rejects([&]{make("a",{"b"},{{0,1}});});
    rejects([&]{make("a",{},{});});
    rejects([&]{make("a",{"a","a"},{{0,1},{0,1}});});
    rejects([&]{make("a",{"a"},{{0,1}},0);});
    rejects([&]{make("a",{"a"},{{0,1}},std::numeric_limits<std::uint64_t>::max());});
    rejects([&]{make("a",{"a"},{{0,1}},920060,0);});
    rejects([&]{make("a",{"a"},{{0,1}},920060,1);});
    rejects([&]{make("a",{"a"},{{0,1}},920060,1u<<24u,{0,1,1});});
    rejects([&]{make("a",{"a"},{{0,1}},920060,1u<<24u,{1,1,1});});
    rejects([&]{make("a",{"a"},{{0,1}},920060,1u<<24u,{1000000,1,16384});});
    rejects([&]{make(std::string(8193,'a'),{"a"},{{0,1}});});
    check(make("a",{"a"},{{0,1}},std::numeric_limits<std::uint64_t>::max()-1).tokens->info().name_space==
        std::numeric_limits<std::uint64_t>::max());
    auto whitespace=literal_whitespace_chart(" a\tb\nc\rd\ve\ff ",920060,1u<<24u,limits);
    check(whitespace.spans.size()==6);
    for(std::size_t i=0;i<6;++i) {
        check(whitespace.spans[i].begin==2*i+1);
        check(whitespace.spans[i].end==2*i+2);
    }
    rejects([&]{literal_whitespace_chart(" \t\r\n\v\f",920060,1u<<24u,limits);});
    rejects([&]{literal_whitespace_chart("",920060,1u<<24u,limits);});
    check(literal_whitespace_chart(std::string("\xff \x80",3),920060,1u<<24u,limits).spans.size()==2);
    const auto scanned=literal_whitespace_spans(std::string("\0\xff \x80",4));
    check(scanned.size()==2 && scanned[0].begin==0 && scanned[0].end==2 &&
        scanned[1].begin==3 && scanned[1].end==4);
    rejects([&]{literal_whitespace_spans("");});
    rejects([&]{literal_whitespace_spans(" \t\r\n\v\f");});
    rejects([&]{literal_whitespace_spans(std::string(8193,'x'));});
    std::string many;
    for(std::size_t i=0;i<128;++i) many+="x ";
    check(literal_whitespace_spans(many).size()==128);
    rejects([&]{literal_whitespace_spans(many+"x");});
    std::cout << checks << " exact native checks\n";
}
