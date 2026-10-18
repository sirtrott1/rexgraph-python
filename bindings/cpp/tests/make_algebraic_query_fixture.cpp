#include <rfc/database/prepared_algebraic_query.hpp>
#include <rfc/storage/algebraic_archive_codec.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace rfc;
namespace {
void rows(const std::vector<numeric::AlgebraicElement>& values) {
    std::cout<<'[';bool first=true;
    for (const auto& value:values) {
        if (!first) std::cout<<',';
        first=false;std::cout<<'[';bool start=true;
        for (const auto& q:value.coordinates()) {if (!start)std::cout<<',';start=false;std::cout<<'"'<<q.get_str()<<'"';}
        std::cout<<']';
    }std::cout<<']';
}
void keys(const database::DerivedAlgebraicQuery& result) {
    std::cout<<'[';bool first=true;
    for (const auto& record:result.records()) {if (!first)std::cout<<',';first=false;std::cout<<record.key();}
    std::cout<<']';
}
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    for (const auto* profile:{"rational","real_cubic","complex_quadratic","complex_quartic"}) {
        std::ifstream in(std::string(argv[1])+"/"+profile+".rfcxa",std::ios::binary);
        if (!in) return 3;
        const std::string raw{std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
        const auto archive=storage::decode_algebraic_archive({reinterpret_cast<const std::byte*>(raw.data()),raw.size()});
        const auto snapshot=database::AlgebraicSnapshot::publish(archive,4001,5);
        const runtime::WorkAvailability availability{4001,5};runtime::CpuTileExecutor executor({2,4,1000000});
        database::AlgebraicQueryRequest request;request.kind=database::AlgebraicRecordKind::Vector;
        const auto query=database::PreparedAlgebraicQuery::prepare(snapshot,request,availability,runtime::ExecutionLease{},1);
        std::cout<<"{\"profile\":\""<<profile<<"\",\"vectors\":";keys(query.execute(availability,executor));
        request.all_of.push_back({{0},database::AlgebraicEquality::Equal,archive.bundle().field().scalar(numeric::Rational{0})});
        const auto zero_query=database::PreparedAlgebraicQuery::prepare(snapshot,request,availability);
        std::cout<<",\"zero_vectors\":";keys(zero_query.execute(availability,executor));
        const auto window=database::AlgebraicTensorWindow::prepare(snapshot.record(90),{1,0},{3,2},availability);
        std::cout<<",\"window_indices\":[2,3,4,5],\"window\":";
        std::vector<numeric::AlgebraicElement> selected;
        for (std::size_t i=0;i<window.supported_count();++i) selected.push_back(window.coordinate(i));
        rows(selected);std::cout<<",\"sum\":";rows({*window.sum(availability)});
        std::cout<<",\"occurrences\":[";bool first=true;
        const auto count=archive.bundle().source().grade(core::Grade{1}).lower_index.size();
        for (std::size_t i=0;i<count;++i) {
            const auto occurrence=snapshot.occurrence(core::Grade{1},i);
            const auto weighted=occurrence.weighted_map(),contribution=occurrence.contribution_map();
            if (!first)std::cout<<',';
            first=false;
            std::cout<<"{\"key\":"<<occurrence.key.value<<",\"active\":"<<(occurrence.active?"true":"false")
                     <<",\"coefficient\":\""<<occurrence.coefficient.get_str()<<"\",\"underlying\":";
            rows(occurrence.underlying.coefficients());std::cout<<",\"weighted\":";rows(weighted.coefficients());
            std::cout<<",\"contribution\":";rows(contribution.coefficients());std::cout<<'}';
        }std::cout<<"]}"<<'\n';
    }
}
