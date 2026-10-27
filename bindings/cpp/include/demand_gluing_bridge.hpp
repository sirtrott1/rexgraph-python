#pragma once
#include <rfc/inference/demand_gluing.hpp>
#include <atomic>
#include <sstream>
namespace rexgraph::python {
namespace dg=rfc::inference;namespace ds=rfc::storage;
inline std::span<const std::byte> demand_span(const std::string& s){return std::as_bytes(std::span(s.data(),s.size()));}
inline std::string demand_bytes(std::span<const std::byte> b){return {reinterpret_cast<const char*>(b.data()),b.size()};}
inline ds::PayloadManifest demand_manifest(const std::string& bytes){
 if(bytes.size()>(8u<<20))throw std::length_error("demand original manifest receiving admission");
 ds::rfcx_detail::Reader r(demand_span(bytes));if(r.u64()!=ds::payload_manifest_detail::magic || r.u64()!=1)throw std::invalid_argument("unknown original payload manifest");
 const auto n=r.size_value();if(n>4096)throw std::length_error("demand original source receiving admission");std::vector<ds::PayloadSourceRevision> sources;
 for(std::size_t j=0;j<n;++j)sources.push_back(ds::payload_manifest_detail::source(r));
 return ds::PayloadManifest::decode(demand_span(bytes),sources);
}
struct DemandPatchData {std::size_t index{};std::vector<std::uint64_t> cells,occurrences;std::uint64_t available{};};
inline std::string demand_cover(const std::string& bytes,const std::vector<std::uint64_t>& meta,const std::vector<DemandPatchData>& rows){
 if(meta.size()!=7)throw std::invalid_argument("seven demand producer/clock selectors required");
 auto manifest=demand_manifest(bytes);ds::DemandCoverDeclaration d{meta[0],meta[1],meta[2],meta[3],meta[4],meta[5],meta[6],{}};
 for(const auto& row:rows){const auto& original=manifest.payloads().at(row.index);if(row.cells.size()!=original.cells.size())throw std::invalid_argument("complete demand cell map required");ds::DemandPatch p;p.payload=original.identity;p.available_at=row.available;p.occurrences=row.occurrences;
  for(std::size_t i=0;i<row.cells.size();++i){p.cells.push_back({original.cells[i].grade,row.cells[i]});}
  d.patches.push_back(std::move(p));}
 auto cover=ds::DemandCover::create(manifest,d);return demand_bytes(cover.bytes());
}
struct DemandArchiveInput {std::uint64_t name_space{},source{},revision{},payload{};std::string bytes;};
inline std::string demand_publish(const std::string& path,const std::vector<DemandArchiveInput>& input){
 if(input.size()>256)throw std::length_error("demand archive producer count admission");
 std::size_t total{};for(const auto& row:input){total=rfc::machine::checked_size_add(total,row.bytes.size());if(total>(256u<<20))throw std::length_error("demand archive input receiving admission");}
 std::vector<ds::DeclaredRfcxPayload> values;std::vector<ds::PayloadSourceRevision> sources;std::vector<ds::PayloadDescriptor> descriptors;
 for(const auto& row:input){ds::RfcxPayloadLimits limits;const auto bytes=demand_span(row.bytes);const auto plan=ds::rfcx_payload_detail::inspect(bytes,limits);if(plan.decoded_bytes>(256u<<20) || plan.scratch_bytes>(512u<<20))throw std::length_error("demand archive producer preparation admission");ds::MappedRfcx image(bytes);auto source=image.materialize_canonical();
  auto value=ds::declare_rfcx_payload(row.name_space,row.source,row.revision,row.payload,source,limits);if(!std::ranges::equal(value.bytes,bytes))throw std::invalid_argument("original canonical RFCX producer bytes required");sources.push_back(value.descriptor.identity.source);descriptors.push_back(value.descriptor);values.push_back(std::move(value));}
 auto manifest=ds::PayloadManifest::create(sources,descriptors);std::vector<std::span<const std::byte>> payloads;for(const auto& value:values)payloads.push_back(value.bytes);ds::write_payload_archive(path,manifest,payloads);return demand_bytes(manifest.bytes());
}
class DemandCancel {std::atomic_bool cancelled_{false};public:void cancel(){cancelled_=true;}void check()const{if(cancelled_)throw rfc::runtime::ExecutionCancelled();}};
class DemandBridge {
 ds::PayloadArchiveProvider provider_;rfc::database::NativePayloadCache cache_;dg::DemandGluing plan_;std::shared_ptr<DemandCancel> cancel_;
 static dg::DemandExecutionLimits execution(const std::vector<std::size_t>& v){if(v.size()!=19)throw std::invalid_argument("nineteen explicit demand receiving limits required");return {v[1],v[2],v[3],v[4],v[5],v[6],{v[7],v[8],v[9],v[10]}};}
 static rfc::database::NativePayloadLimits cache_limits(const std::vector<std::size_t>& v){(void)execution(v);rfc::database::NativePayloadLimits out;out.max_entries=v[11];out.raw.max_payload_bytes=v[12];out.max_decoded_bytes=v[13];out.max_coefficient_bits=v[14];out.max_scratch_bytes=v[15];out.max_io_bytes=v[16];out.max_metadata_bytes=v[17];out.preparation.max_file_bytes=v[18];return out;}
 DemandBridge(ds::PayloadArchiveProvider p,ds::DemandCover c,const std::vector<std::size_t>& v,std::shared_ptr<DemandCancel> token):provider_(std::move(p)),cache_(provider_,cache_limits(v)),plan_(cache_,std::move(c),execution(v)),cancel_(std::move(token)){}
 static void text(ds::rfcx_detail::Writer& w,const std::string& s){w.u64(s.size());for(char c:s)w.u8(static_cast<std::uint8_t>(c));}
 static std::string text(ds::rfcx_detail::Reader& r){const auto n=r.size_value();if(n>65536)throw std::length_error("demand scalar text admission");std::string s;for(std::size_t j=0;j<n;++j)s.push_back(static_cast<char>(r.u8()));return s;}
 static rfc::numeric::Rational rational(ds::rfcx_detail::Reader& r){return rfc::numeric::canonical_rational(rfc::numeric::Rational{text(r)});}
 static void rational(ds::rfcx_detail::Writer& w,const rfc::numeric::Rational& q){std::ostringstream s;s<<q;text(w,s.str());}
 static void vector(ds::rfcx_detail::Writer& w,const rfc::numeric::RationalVector& v){w.u64(v.size());for(const auto& q:v)rational(w,q);}
 static void matrix(ds::rfcx_detail::Writer& w,const rfc::numeric::ExactMatrix& m){w.u64(m.rows());w.u64(m.cols());w.u64(m.nonzeros());for(const auto& [ij,q]:m.entries()){w.u64(ij.first);w.u64(ij.second);rational(w,q);}}
public:
 static std::shared_ptr<DemandBridge> open(const std::string& path,const std::string& bytes,const std::string& image,const std::vector<std::size_t>& v,const std::shared_ptr<DemandCancel>& token){
  (void)execution(v);if(!token)throw std::invalid_argument("demand cancellation owner required");token->check();auto m=demand_manifest(bytes);ds::DemandCoverLimits l;l.max_index_bytes=v[0];auto c=ds::DemandCover::decode(demand_span(image),m,l);
  auto p=ds::PayloadArchiveProvider::open(path,m,{},[token]{token->check();});return std::shared_ptr<DemandBridge>(new DemandBridge(std::move(p),std::move(c),v,token));}
 std::string execute(const std::string& request){
  cancel_->check();if(request.size()>(16u<<20))throw std::length_error("demand request admission");ds::rfcx_detail::Reader r(demand_span(request));
  auto checkpoint=[token=cancel_]{token->check();};plan_.qualify_index(checkpoint);const auto before_io=provider_.usage().payload_io_bytes;const auto before_admitted=cache_.usage().admitted_io_bytes;const auto mode=r.u8();dg::DemandExecutionResult result;
  if(mode==1){const auto identity=r.u64();const auto grade=r.u16();const auto width=r.size_value();const auto word=text(r);const bool channels=r.boolean();const auto n=r.size_value();if(n>65536 || !width || width>16)throw std::length_error("demand field input admission");
   std::map<std::uint64_t,rfc::numeric::RationalVector> fields;for(std::size_t j=0;j<n;++j){const auto key=r.u64();rfc::numeric::RationalVector values;for(std::size_t k=0;k<width;++k)values.push_back(rational(r));if(!fields.emplace(key,std::move(values)).second)throw std::invalid_argument("duplicate demand field cell");}
   auto field=plan_.field(identity,grade,width,std::move(fields));result=plan_.word(field,word,channels,checkpoint);
  }else if(mode==2){const bool global=r.boolean(),boolean=r.boolean();auto n=r.size_value();if(n>65536)throw std::length_error("demand seed input admission");std::vector<std::uint64_t> seeds;for(std::size_t j=0;j<n;++j)seeds.push_back(r.u64());n=r.size_value();if(n>65536)throw std::length_error("demand rhs input admission");std::map<std::uint64_t,rfc::numeric::Rational> rhs;
   for(std::size_t j=0;j<n;++j){const auto key=r.u64();auto q=rational(r);if(!rhs.emplace(key,std::move(q)).second)throw std::invalid_argument("duplicate demand rhs");}result=plan_.family(std::move(seeds),rhs,global,boolean,checkpoint);
  }else throw std::invalid_argument("unknown demand query profile");
  if(r.remaining())throw std::invalid_argument("trailing demand query data");
  ds::rfcx_detail::Writer w;w.u64(0x3152454458434652ULL);w.u16(result.grade);w.u64(result.width);rational(w,result.moment);w.u64(result.field.size());for(const auto& [key,value]:result.field){w.u64(key);vector(w,value);}w.u64(result.channels.size());for(const auto& [key,q]:result.channels){w.u64(key);rational(w,q);}
  for(const auto* v:{&result.variables,&result.edges}){w.u64(v->size());for(auto x:*v)w.u64(x);}vector(w,result.particular);vector(w,result.contradiction);matrix(w,result.equations);matrix(w,result.kernel);w.u64(result.boolean_members.size());for(const auto& v:result.boolean_members)vector(w,v);w.u8(result.consistent);w.u8(result.boolean_complete);w.u8(result.trace.global);
  w.u64(result.trace.loaded.size());for(std::size_t j=0;j<result.trace.loaded.size();++j){ds::payload_manifest_detail::key(w,result.trace.loaded[j]);w.u64(result.trace.generations[j]);}
  const auto usage=cache_.usage();const auto raw=cache_.raw_usage();const auto io=provider_.usage();
  for(auto count:{result.trace.patch_visits,result.trace.occurrence_visits,result.trace.scalar_steps,result.trace.work_envelope_bytes,result.trace.output_bytes,result.trace.fill_entries,result.trace.output_coefficient_bits,plan_.cover().index_bytes(),usage.metadata_bytes,usage.resident_decoded_bytes,usage.resident_coefficient_bits,usage.peak_scratch_bytes,usage.admitted_io_bytes-before_admitted,raw.resident_payload_bytes,io.metadata_io_bytes,io.payload_io_bytes-before_io,io.archive_bytes,result.trace.index_preparation.payload_io_bytes,result.trace.index_preparation.patches,result.trace.index_preparation.work_envelope_bytes,io.payload_io_bytes,usage.admitted_io_bytes})w.u64(count);
  cancel_->check();return demand_bytes(w.bytes());
 }
 bool evict(std::size_t i){return cache_.evict(provider_.manifest().payloads().at(i).identity);}
};
} // namespace rexgraph::python
