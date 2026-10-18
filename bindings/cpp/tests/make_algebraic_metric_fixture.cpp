#include "rfc/storage/algebraic_archive_graph_history.hpp"
#include "rfc/storage/native_series_codec.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace rfc;
int main(int argc,char** argv) {try {
    if (argc!=3) return 2;
    const std::filesystem::path input(argv[1]),output(argv[2]);
    std::filesystem::create_directories(output);
    for (unsigned i=1;i<=15;++i) {
        const auto name="profile-"+std::to_string(i);
        const auto original=storage::read_algebraic_archive(input/(name+".rfcxa"));
        auto c=original.contents();const auto& bundle=original.bundle();const auto& f=bundle.field();
        if (!c.metric_domain) c.metric_domain=field::AlgebraicMetricDomain::real(
            geometry::SelectedRealAlgebraicEmbedding::isolate(f,1,numeric::Rational{3,2}),numeric::AlgebraicInvolution::identity(f));
        const auto& domain=*c.metric_domain;const auto one=f.scalar(numeric::Rational{1}),zero=f.scalar(numeric::Rational{0});
        const auto t=f.generator()+one;
        for (std::size_t k=0;k<bundle.source().grade_count();++k) {
            const auto grade=core::Grade{static_cast<std::uint16_t>(k)};
            for (std::size_t cell=0;cell<bundle.source().grade(grade).cell_count();++cell) {
                const auto module=bundle.stalk({grade,static_cast<std::uint32_t>(cell)});const auto rank=module.rank();
                std::vector<numeric::AlgebraicElement> d(rank*rank,zero),g(rank*rank,zero);
                for (std::size_t row=0;row<rank;++row) {
                    d[row*rank+row]=f.scalar(numeric::Rational(numeric::Integer(k+cell+row+2)));
                    g[row*rank+row]=one;
                    for (std::size_t col=row+1;col<rank;++col) g[row*rank+col]=t.power(col-row);
                }
                const auto diagonal=field::AlgebraicModuleMetric::certify(field::AlgebraicMeasurementForm::declare(
                    domain,local::AlgebraicModuleMap::declare(module,module,std::move(d))));
                const auto basis=local::AlgebraicModuleMap::declare(module,module,std::move(g));
                const auto metric=field::AlgebraicModuleMetric::certify(diagonal.measurement().pullback(basis));
                c.records.push_back({5000+100*k+cell,metric});
            }
        }
        const auto archive=storage::AlgebraicArchive::declare(bundle,std::move(c));
        storage::write_algebraic_archive(archive,output/(name+".rfcxa"));
        storage::AlgebraicArchiveGraphContents contents;
        contents.fields={{1,f}};contents.bundles={{1,bundle}};contents.archives={{9,archive}};
        const auto& declared=archive.contents();
        contents.involutions={{1,*declared.involution}};contents.embeddings={{1,*declared.embedding}};contents.domains={{1,*declared.metric_domain}};
        if (declared.fibers) contents.fibers={{1,*declared.fibers}};
        const auto graph=storage::AlgebraicGraphSnapshot::anchor_only(storage::AlgebraicArchiveGraph::declare(std::move(contents)),1);
        storage::write_algebraic_graph_snapshot(graph,output/(name+".rfcxh"));
        const storage::NativeTemporalSeries series(storage::with_algebraic_graph_snapshot(
            storage::TemporalEndpoint(graph.source(),31),{9631,1,2},graph),9603);
        const auto data=storage::encode_native_series(series);std::ofstream out(output/(name+".rfcxs"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
        if (!out) throw std::runtime_error("metric history fixture write failed");
        std::filesystem::copy_file(input/(name+".jsonl"),output/(name+".jsonl"),std::filesystem::copy_options::overwrite_existing);
    }
    std::cout<<"15 explicit complete metric fixtures written\n";
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
