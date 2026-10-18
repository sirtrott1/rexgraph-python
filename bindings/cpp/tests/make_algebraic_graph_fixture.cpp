#include "rfc/storage/algebraic_archive_graph_codec.hpp"
#include <iostream>

namespace {
using namespace rfc;
void element(const numeric::AlgebraicElement& value) {
    std::cout<<'[';
    for (std::size_t k=0;k<value.coordinates().size();++k) {
        if (k) std::cout<<',';
        std::cout<<'"'<<value.coordinates()[k].get_str(16)<<'"';
    }
    std::cout<<']';
}
void vector(const local::AlgebraicModuleVector& value) {
    std::cout<<'[';
    for (std::size_t j=0;j<value.coordinates().size();++j) {
        if (j) std::cout<<',';
        element(value.coordinates()[j]);
    }
    std::cout<<']';
}
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    try {
        const auto graph=rfc::storage::read_algebraic_archive_graph(argv[1]);
        std::cout<<"{\"archives\":{";
        bool first=true;
        for (const auto& row:graph.contents().archives) {
            if (!first) std::cout<<',';
            first=false; std::cout<<'"'<<row.key<<"\":";
            vector(std::get<rfc::local::AlgebraicModuleVector>(row.value.at(10)));
        }
        std::cout<<"},\"carriers\":{"; first=true;
        for (const auto& row:graph.contents().carriers) {
            if (!first) std::cout<<',';
            first=false; std::cout<<'"'<<row.key<<"\":[";
            const auto& m=row.value.source(); bool basis_first=true;
            for (std::size_t j=0;j<m.rank();++j) for (std::size_t k=0;k<m.field().degree();++k) {
                if (!basis_first) std::cout<<',';
                basis_first=false;
                std::vector<rfc::numeric::AlgebraicElement> values(m.rank(),m.field().scalar(0));
                rfc::numeric::RationalVector coordinates(m.field().degree()); coordinates[k]=1;
                values[j]=m.field().element(coordinates);vector(row.value.apply(m.vector(std::move(values))));
            }
            std::cout<<']';
        }
        std::cout<<"},\"measurements\":{";first=true;
        for (const auto& row:graph.contents().measurements) {
            if (!first) std::cout<<',';
            first=false;std::cout<<'"'<<row.key<<"\":[";
            const auto& proof=graph.measurement(row.key);
            const auto& value=std::get<rfc::local::AlgebraicModuleVector>(graph.archive(row.source_archive).at(10));
            element(proof.domains().scalar_comparison().apply(proof.source().quadrance(value)));
            std::cout<<',';element(proof.target().quadrance(proof.carrier().apply(value)));std::cout<<']';
        }
        std::cout<<"}}\n";
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
