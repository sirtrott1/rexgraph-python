#include "rfc/storage/algebraic_archive_history.hpp"
#include "rfc/storage/native_series_codec.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace rfc;
void write_series(const std::filesystem::path& path,const storage::NativeTemporalSeries& history) {
    const auto bytes=storage::encode_native_series(history);
    std::ofstream stream(path,std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("native history fixture write failed");
}
storage::AlgebraicArchive revision(const storage::AlgebraicArchive& base, numeric::Rational offset) {
    auto contents=base.contents();
    auto& vector=std::get<local::AlgebraicModuleVector>(contents.records.front().value);
    auto coordinates=vector.coordinates();
    coordinates[0]=coordinates[0]+base.bundle().field().scalar(offset);
    vector=vector.carrier().vector(std::move(coordinates));
    return storage::AlgebraicArchive::declare(base.bundle(),std::move(contents));
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    try {
        const std::filesystem::path input(argv[1]),output(argv[2]);
        std::filesystem::create_directories(output);
        for (const auto* name: {"rational","real_cubic","complex_quadratic","complex_quartic"}) {
            const auto base=storage::read_algebraic_archive(input/(std::string(name)+".rfcxa"));
            const storage::AlgebraicArchiveChannel channel{701,1,100};
            auto endpoint=storage::with_algebraic_archive(storage::TemporalEndpoint(base.bundle().source(),40),channel,base);
            endpoint=storage::with_algebraic_archive(endpoint,{701,1,101},base);
            endpoint.declarations->values.push_back({200,channel.chunk_type(),0,0,{},{},composite::DeclaredValue::integer(42)});
            const std::vector<std::byte> invalid(32,std::byte{'X'});
            endpoint.declarations->values.push_back({201,channel.type(),0,0,{},{},storage::algebraic_archive_history_detail::chunks(invalid)});
            storage::NativeTemporalSeries history(endpoint,9001);
            write_series(output/(std::string(name)+"-40.rfcxs"),history);
            for (const auto offset: {7,99}) {
                auto next=storage::with_algebraic_archive(endpoint,channel,revision(base,numeric::Rational{offset}));
                next.sequence=endpoint.sequence+1;
                history.append(storage::make_temporal_segment(endpoint,next));
                endpoint=std::move(next);
                write_series(output/(std::string(name)+"-"+std::to_string(endpoint.sequence)+".rfcxs"),history);
            }
            storage::TemporalEndpoint legacy(base.bundle().source(),40);
            legacy.declarations.emplace();legacy.declarations->namespace_key=701;
            legacy.declarations->registry=composite::DeclarationRegistry({composite::Declaration::integer(channel.chunk_type())});
            legacy.declarations->values.push_back({100,channel.chunk_type(),0,0,{},{},
                composite::DeclaredValue::integer(storage::algebraic_archive_history_detail::pack(storage::encode_algebraic_archive(base)))});
            write_series(output/(std::string(name)+"-legacy.rfcxs"),storage::NativeTemporalSeries(legacy,9001));
            core::RelationalComplexBuilder builder(base.bundle().source().signature());
            builder.declare_witness_identity(999);builder.declare_grade(core::Grade{3});
            auto foreign=endpoint;foreign.source=builder.freeze();
            write_series(output/(std::string(name)+"-foreign.rfcxs"),storage::NativeTemporalSeries(foreign,9001));
            if (std::string(name)=="complex_quartic") {
                auto contents=base.contents();
                for (std::uint64_t key=1000;key<1128;++key) contents.records.push_back({key,base.at(10)});
                const auto large=storage::AlgebraicArchive::declare(base.bundle(),std::move(contents));
                const auto large_endpoint=storage::with_algebraic_archive(storage::TemporalEndpoint(base.bundle().source(),40),channel,large);
                write_series(output/"large.rfcxs",storage::NativeTemporalSeries(large_endpoint,9001));
            }
            std::cout<<"{\"profile\":\""<<name<<"\",\"first_sequence\":40,\"last_sequence\":42}\n";
        }
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
