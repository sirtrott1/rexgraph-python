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
        const auto archive=storage::read_algebraic_archive(input/(name+".rfcxa"));
        storage::AlgebraicArchiveGraphContents contents;
        contents.fields={{1,archive.bundle().field()}};contents.bundles={{1,archive.bundle()}};
        contents.archives={{9,archive}};const auto& c=archive.contents();
        if (c.involution) contents.involutions={{1,*c.involution}};
        if (c.embedding) contents.embeddings={{1,*c.embedding}};
        if (c.metric_domain) contents.domains={{1,*c.metric_domain}};
        if (c.fibers) contents.fibers={{1,*c.fibers}};
        const auto graph=storage::AlgebraicGraphSnapshot::anchor_only(storage::AlgebraicArchiveGraph::declare(std::move(contents)),1);
        storage::write_algebraic_graph_snapshot(graph,output/(name+".rfcxh"));
        const storage::AlgebraicGraphChannel channel{9631,1,2};
        const storage::NativeTemporalSeries series(storage::with_algebraic_graph_snapshot(storage::TemporalEndpoint(graph.source(),31),channel,graph),9603);
        const auto data=storage::encode_native_series(series);std::ofstream out(output/(name+".rfcxs"),std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
        if (!out) throw std::runtime_error("grade history fixture write failed");
        std::filesystem::copy_file(input/(name+".jsonl"),output/(name+".jsonl"),std::filesystem::copy_options::overwrite_existing);
    }
    std::cout<<"15 grade history fixtures written\n";
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
