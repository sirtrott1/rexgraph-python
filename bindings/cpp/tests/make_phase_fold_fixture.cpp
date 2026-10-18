#include <rfc/storage/phase_fold_history.hpp>
#include <rfc/storage/native_series_codec.hpp>
#include <fstream>
#include <iostream>

using namespace rfc;
void write(const std::filesystem::path& path,const storage::NativeTemporalSeries& history) {
    const auto bytes=storage::encode_native_series(history);std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("phase/fold fixture write failed");
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    try {
        const std::filesystem::path input(argv[1]),output(argv[2]);std::filesystem::create_directories(output);
        for (const auto* profile:{"rational","algebraic","fold","rational-walk","algebraic-walk"}) {
            const auto base=storage::read_phase_fold_archive(input/(std::string(profile)+".rfcxw"));
            const storage::PhaseFoldChannel channel{801,1,100};
            auto endpoint=storage::with_phase_fold_archive(storage::TemporalEndpoint(base.source(),70),channel,base);
            endpoint=storage::with_phase_fold_archive(endpoint,{801,1,101},base);
            endpoint.declarations->values.push_back({200,channel.chunk_type(),0,0,{},{},composite::DeclaredValue::integer(42)});
            const std::vector<std::byte> opaque(32,std::byte{'X'});
            endpoint.declarations->values.push_back({201,channel.type(),0,0,{},{},storage::algebraic_archive_history_detail::chunks(opaque)});
            storage::NativeTemporalSeries history(endpoint,9101);write(output/(std::string(profile)+"-70.rfcxs"),history);
            const auto changed=base.proposals().empty() ? base.observed_prefix(base.operation_count()-1) : base.apply(7);
            for (const auto& archive:{changed,base.observed_prefix(0)}) {
                auto next=storage::with_phase_fold_archive(endpoint,channel,archive);next.sequence=endpoint.sequence+1;
                history.append(storage::make_temporal_segment(endpoint,next));endpoint=std::move(next);
                write(output/(std::string(profile)+"-"+std::to_string(endpoint.sequence)+".rfcxs"),history);
            }
            core::RelationalComplexBuilder builder(base.source().signature());builder.declare_witness_identity(999);builder.declare_grade(core::Grade{3});
            auto foreign=endpoint;foreign.source=builder.freeze();write(output/(std::string(profile)+"-foreign.rfcxs"),storage::NativeTemporalSeries(foreign,9101));
            for (const auto prefix:{std::size_t{0},base.operation_count()})
                storage::write_phase_fold_archive(base.observed_prefix(prefix),output/(std::string(profile)+"-prefix-"+std::to_string(prefix)+".rfcxw"));
            if (!base.proposals().empty()) storage::write_phase_fold_archive(changed,output/(std::string(profile)+"-applied.rfcxw"));
            std::cout<<"{\"profile\":\""<<profile<<"\",\"operations\":"<<base.operation_count()<<"}\n";
        }
        const auto base=storage::read_phase_fold_archive(input/"rational.rfcxw");
        const auto& word=std::get<temporal::RationalPhaseHistory>(base.word());
        auto large=temporal::RationalPhaseHistory::begin(word.source(),word.cell(),word.phase_at(0));
        for (std::size_t i=0;i<64;++i) large=large.append(word.operations()[i%word.operations().size()]);
        storage::write_phase_fold_archive(storage::PhaseFoldArchive::declare(large),output/"large.rfcxw");
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
