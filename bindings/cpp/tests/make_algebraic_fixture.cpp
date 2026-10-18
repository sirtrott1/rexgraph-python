#include "rfc/storage/algebraic_archive_codec.hpp"
#include "rfc/core/builder.hpp"
#include <filesystem>
#include <iostream>

using namespace rfc;
using Q=numeric::Rational;
using E=numeric::AlgebraicElement;
using Map=local::AlgebraicModuleMap;
using Domain=field::AlgebraicMetricDomain;
using Form=field::AlgebraicMeasurementForm;
using Metric=field::AlgebraicModuleMetric;
struct Case { const char* name; Domain domain; };
std::vector<Case> cases() {
    const auto rational=numeric::CertifiedAlgebraicField::declare({-1,1},numeric::IrreducibleReduction{3});
    const auto cubic=numeric::CertifiedAlgebraicField::declare({-2,0,0,1},numeric::EisensteinShift{2,0});
    const auto quadratic=numeric::CertifiedAlgebraicField::declare({1,0,1},numeric::IrreducibleReduction{3});
    const auto quartic=numeric::CertifiedAlgebraicField::declare({1,0,0,0,1},numeric::EisensteinShift{2,1});
    const auto cyclotomic=numeric::CertifiedAlgebraicField::declare_cyclotomic(12);
    return {
        {"rational",Domain::real(geometry::SelectedRealAlgebraicEmbedding::isolate(rational,0,2),numeric::AlgebraicInvolution::identity(rational))},
        {"real_cubic",Domain::real(geometry::SelectedRealAlgebraicEmbedding::isolate(cubic,1,Q{3,2}),numeric::AlgebraicInvolution::identity(cubic))},
        {"complex_quadratic",Domain::hermitian(geometry::SelectedComplexAlgebraicEmbedding::isolate(quadratic,{0,1},Q{1,4}),
                                             numeric::AlgebraicInvolution::declare(quadratic,-quadratic.generator()))},
        {"complex_quartic",Domain::hermitian(geometry::SelectedComplexAlgebraicEmbedding::isolate(quartic,{Q{11,16},Q{11,16}},Q{1,8}),
                                           numeric::AlgebraicInvolution::declare(quartic,quartic.generator().inverse()))},
        {"complex_cyclotomic",Domain::hermitian(geometry::SelectedComplexAlgebraicEmbedding::isolate(cyclotomic,{Q{7,8},Q{1,2}},Q{1,32}),
                                              numeric::AlgebraicInvolution::declare(cyclotomic,cyclotomic.generator().inverse()))},
    };
}
core::RelationalComplex source() {
    core::RelationSignature signature;
    const auto type=signature.add_type("algebraic-binding-fixture",core::BoundaryPolicy::DeclaredExact,0,10);
    core::RelationalComplexBuilder builder(signature);
    builder.add_grade_one(core::RelationKey{100},type,{{10,true,false,Q{1}},{11,true,true,Q{1}},
        {10,true,false,Q{2}},{10,true,true,Q{2}},{12,false,false,Q{7}}});
    builder.add_grade_one(core::RelationKey{101},type,{{10,true,false,Q{1}},{11,true,true,Q{1}}});
    builder.add_grade_one(core::RelationKey{102},type,{{11,true,false,Q{1}},{12,true,true,Q{1}}});
    builder.add_higher(core::Grade{2},core::RelationKey{200},type,{{0,true,false,Q{2}},{1,true,true,Q{2}}});
    builder.add_higher(core::Grade{2},core::RelationKey{201},type,{});
    builder.declare_grade(core::Grade{3}); return builder.freeze();
}
storage::AlgebraicArchive archive(const Case& sample, bool deformed=false) {
    const auto original=source(); const auto& f=sample.domain.field();
    std::vector<std::vector<std::size_t>> ranks;
    for (std::size_t k=0;k<original.grade_count();++k)
        ranks.emplace_back(original.grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count(),2);
    ranks[0][1]=3; const auto bundle=local::AlgebraicStalkBundle::declare(original,f,ranks);
    const auto u=bundle.stalk({core::Grade{0},0}),physical=bundle.stalk({core::Grade{0},1}),v=bundle.stalk({core::Grade{0},2});
    const auto one=f.scalar(Q{1}),zero=f.scalar(Q{0}),alpha=f.generator();
    const auto t=one+alpha,conjugate=sample.domain.involution().apply(t);
    auto metric=[&](const local::AlgebraicModule& carrier,Q weight) {
        const auto d=f.scalar(weight),e=f.scalar(Q{3});
        return Metric::certify(Form::declare(sample.domain,Map::declare(carrier,carrier,{d,d*t,conjugate*d,conjugate*d*t+e})));
    };
    numeric::Integer big{1}; big<<=512;
    const auto x=u.vector({alpha+f.scalar(Q{1,3}),f.scalar(Q{big,7})});
    const auto mu=metric(u,Q{2}),mv=metric(v,Q{4}); const auto map=Map::declare(u,v,{one,alpha,one+alpha,zero});
    std::vector<local::AlgebraicOccurrenceMap> maps;
    for (std::size_t k=1;k<original.grade_count();++k) {
        const auto grade=core::Grade{static_cast<std::uint16_t>(k)}; const auto& arena=original.grade(grade);
        for (std::size_t i=0;i<arena.incidence_count();++i) {
            const auto upper=bundle.stalk({grade,arena.upper_index[i]});
            const auto lower=bundle.stalk({core::Grade{static_cast<std::uint16_t>(k-1)},arena.lower_index[i]});
            std::vector<E> entries(upper.rank()*lower.rank(),zero);
            if (!arena.aspects.get(original.signature().existence_aspect(),i)) std::fill(entries.begin(),entries.end(),one);
            else for (std::size_t j=0;j<std::min(upper.rank(),lower.rank());++j) entries[j*upper.rank()+j]=one;
            if (deformed && k==2 && i==0) entries[0]=entries[0]+one;
            maps.push_back({grade,i,Map::declare(upper,lower,std::move(entries))});
        }
    }
    storage::AlgebraicArchiveContents contents;
    contents.metric_domain=sample.domain;
    contents.fibers=local::AlgebraicRelationalFiberSystem::declare(bundle,maps,
        deformed ? local::AlgebraicChainPolicy::DeclaredDeformation : local::AlgebraicChainPolicy::RequireFlat);
    contents.records={{10,x},{11,u.vector({zero,zero})},{20,mu.measurement().as_covector(x)},
        {30,map},{40,mu.measurement()},{50,mu},{60,mv},{70,field::algebraic_metric_adjoint(map,mu,mv)},
        {80,Form::declare(sample.domain,Map::declare(u,u,{one,zero,zero,-one}))},
        {90,Map::declare(u,physical,{one,zero,zero,one,one,one})},
        {91,Map::declare(physical,u,{one,zero,zero,zero,one,zero})}};
    contents.reconstructions={{100,90,91}}; return storage::AlgebraicArchive::declare(bundle,std::move(contents));
}
void coordinates(const std::vector<E>& values) {
    std::cout<<'[';
    for (std::size_t i=0;i<values.size();++i) {
        if (i) std::cout<<',';
        std::cout<<'[';
        for (std::size_t j=0;j<values[i].coordinates().size();++j) {
            if (j) std::cout<<',';
            std::cout<<'"'<<values[i].coordinates()[j]<<'"';
        }
        std::cout<<']';
    }
    std::cout<<']';
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    try {
        const std::filesystem::path output(argv[1]);std::filesystem::create_directories(output);
        const auto profiles=cases();
        const auto deformed=archive(profiles[1],true);
        storage::write_algebraic_archive(deformed,output/"deformed.rfcxa");
        const auto plain=archive(profiles[0]);
        storage::AlgebraicArchiveContents contents;contents.records={{10,std::get<local::AlgebraicModuleVector>(plain.at(10))}};
        storage::write_algebraic_archive(storage::AlgebraicArchive::declare(plain.bundle(),std::move(contents)),output/"plain.rfcxa");
        for (const auto& sample:profiles) {
            const auto value=archive(sample);storage::write_algebraic_archive(value,output/(std::string(sample.name)+".rfcxa"));
            const auto& x=std::get<local::AlgebraicModuleVector>(value.at(10));
            const auto& metric=std::get<Metric>(value.at(50)); const auto& map=std::get<Map>(value.at(30));
            const auto pair=value.reconstruction(100); const auto image=pair.accession().apply(x);
            const auto y=map.apply(x),solved=metric.solve(x),reconstructed=pair.reconstruction().apply(image);
            const auto dual=metric.measurement().as_covector(x);
            std::cout<<"{\"profile\":\""<<sample.name<<"\",\"map\":";coordinates(y.coordinates());
            std::cout<<",\"solve\":";coordinates(solved.coordinates());
            std::cout<<",\"quadrance\":";coordinates({metric.quadrance(x)});
            std::cout<<",\"covector\":";coordinates(dual.coefficients());
            std::cout<<",\"evaluate\":";coordinates({std::get<local::AlgebraicCovector>(value.at(20)).evaluate(x)});
            std::cout<<",\"accession\":";coordinates(image.coordinates());
            std::cout<<",\"reconstruct\":";coordinates(reconstructed.coordinates());
            std::cout<<"}\n";
        }
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
