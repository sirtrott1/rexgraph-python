#include "rfc/storage/algebraic_archive_graph_history.hpp"
#include "rfc/storage/native_series_codec.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace rfc;
void save(const std::filesystem::path& path,const storage::NativeTemporalSeries& series) {
    const auto data=storage::encode_native_series(series);std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
    if (!out) throw std::runtime_error("fixture write failed");
}
int main(int argc,char** argv) {try {
    if (argc!=3) return 2;
    std::filesystem::create_directories(argv[2]);const std::filesystem::path root(argv[2]),input(argv[1]);
    const storage::AlgebraicGraphChannel channel{9611,1,2};
    const auto first=storage::read_algebraic_graph_snapshot(input/"revision-40.rfcxh");
    auto endpoint=storage::with_algebraic_graph_snapshot(storage::TemporalEndpoint(first.source(),40),channel,first);
    endpoint.declarations->values.push_back({20,channel.chunk_type(),0,0,{},{},composite::DeclaredValue::integer(42)});
    endpoint.declarations->values.push_back({21,channel.type(),0,0,{},{},storage::algebraic_archive_history_detail::chunks(std::vector<std::byte>(32,std::byte{'X'}))});
    storage::NativeTemporalSeries history(endpoint,9601);save(root/"40.rfcxs",history);
    storage::write_algebraic_graph_snapshot(first,root/"40.rfcxh");
    for (std::uint64_t i=1;i<3;++i) {
        const auto snapshot=storage::read_algebraic_graph_snapshot(input/("revision-"+std::to_string(40+i)+".rfcxh"));
        auto next=storage::with_algebraic_graph_snapshot(endpoint,channel,snapshot);next.sequence=40+i;
        history.append(storage::make_temporal_segment(endpoint,next));endpoint=std::move(next);
        save(root/(std::to_string(40+i)+".rfcxs"),history);
        storage::write_algebraic_graph_snapshot(snapshot,root/(std::to_string(40+i)+".rfcxh"));
    }
    auto corrupt=endpoint;
    for (auto& row:corrupt.declarations->values) if (row.key==2)
        row.value=storage::algebraic_archive_history_detail::chunks(std::vector<std::byte>(32,std::byte{'X'}));
    corrupt.sequence=43;history.append(storage::make_temporal_segment(endpoint,corrupt));save(root/"future-corrupt.rfcxs",history);
    auto foreign=endpoint;core::RelationalComplexBuilder builder(endpoint.source.signature());builder.declare_witness_identity(999);
    foreign.source=builder.freeze();save(root/"foreign.rfcxs",storage::NativeTemporalSeries(foreign,9601));
    std::cout<<"graph history fixtures written\n";
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
