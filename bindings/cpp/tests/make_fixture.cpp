#include "rfc/field/execution_tower.hpp"
#include "rfc/storage/native_series_codec.hpp"
#include "rfc/learning/integer_network.hpp"
#include "rfc/learning/native_exact_trainer.hpp"
#include "rfc/learning/integer_perceptron.hpp"
#include "rfc/algebra/morphism.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace rfc;
using namespace rfc::storage;
using Q=numeric::Rational;
TemporalEndpoint endpoint(bool positive=false,bool empty=false) {
    core::RelationSignature signature;
    const auto type=signature.add_type("bridge-fixture",core::BoundaryPolicy::DeclaredExact,0,10);
    core::RelationalComplexBuilder builder(signature);
    builder.add_grade_one(core::RelationKey{100},type,{{10,true,false,Q{1}},{11,true,true,Q{1}},
        {10,true,false,Q{2}},{10,true,true,Q{2}},{12,false,false,Q{7}}});
    builder.add_grade_one(core::RelationKey{101},type,{{10,true,false,Q{1}},{11,true,true,Q{1}}});
    builder.add_grade_one(core::RelationKey{102},type,{{11,true,false,Q{1}},{12,true,true,Q{1}}});
    builder.add_higher(core::Grade{2},core::RelationKey{200},type,{{0,true,false,Q{2}},{1,true,true,Q{2}}});
    builder.add_higher(core::Grade{2},core::RelationKey{201},type,{});
    if(empty)builder.declare_grade(core::Grade{3});
    TemporalEndpoint result(builder.freeze());
    for(unsigned k=0;k<result.source.grade_count();++k) {
        const auto n=result.source.grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count();
        result.carriers.push_back({10+k,TemporalCarrierKind::SourceGrade,core::Grade{static_cast<std::uint16_t>(k)}, {},0,numeric::IndexDomain::linear(n)});
        TemporalTensor field; field.key=1000+k; field.role=TemporalTensorRole::Field; field.output=10+k;
        std::vector<numeric::SparseRationalEntry> values;
        if(n)values={{0,Q{1,3}},{n-1,Q{-2,5}}};
        field.tensor={numeric::IndexDomain::linear(n),numeric::StructuralSupport::full(n),
            numeric::SparseRationalVector(n,std::move(values))};
        result.tensors.push_back(field);
        if(!n)continue;
        TemporalTensor metric; metric.key=2000+k; metric.role=TemporalTensorRole::Metric; metric.input=10+k; metric.output=10+k;
        numeric::ExactMatrix matrix(n,n); for(std::size_t i=0;i<n;++i)matrix.set(i,i,Q{static_cast<unsigned long>(i+2)});
        if(positive && n>1){matrix.set(0,1,Q{1});matrix.set(1,0,Q{1});}
        metric.tensor=temporal_matrix_tensor(matrix);result.tensors.push_back(metric);
    }
    result.carriers.push_back({21,TemporalCarrierKind::SourceGrade,core::Grade{1},{},0,numeric::IndexDomain::linear(3)});
    auto other=result.tensors[2];other.key=3001;other.output=21;result.tensors.push_back(other);
    auto partial=result.tensors[2];partial.key=3002;
    partial.tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::sparse(3,{0}),numeric::SparseRationalVector(3,{{0,Q{1,3}}})};
    result.tensors.push_back(partial);
    auto wide=result.tensors[2];wide.key=3003;numeric::Integer big{1};big<<=512;
    wide.tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::full(3),numeric::SparseRationalVector(3,{{0,Q{big}},{2,Q{-2,5}}})};
    result.tensors.push_back(wide);
    wide.key=3004;big=1;big<<=16384;
    wide.tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::full(3),numeric::SparseRationalVector(3,{{0,Q{big,7}}})};
    result.tensors.push_back(wide);
    validate_temporal_endpoint(result);return result;
}
void save(const std::filesystem::path& path,const std::vector<std::byte>& bytes) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if(!file)throw std::runtime_error("fixture write failed");
}
void reference(const TemporalEndpoint& source,const std::string& name) {
    std::vector<field::ExactMetric> metrics;
    for(unsigned k=0;k<3;++k)metrics.emplace_back(temporal_tensor_matrix(source.tensors[k*2+1]));
    const auto oracle=field::ExactExecutionTower::weighted(source.source,std::move(metrics));
    for(unsigned grade=0;grade<3;++grade) for(auto kind:{field::OperatorKind::Boundary,field::OperatorKind::Adjoint,
        field::OperatorKind::HodgeDown,field::OperatorKind::HodgeUp,field::OperatorKind::Hodge,field::OperatorKind::Green,field::OperatorKind::HarmonicProjector}) {
        if(!grade && (kind==field::OperatorKind::Boundary || kind==field::OperatorKind::Adjoint))continue;
        const auto input_grade=kind==field::OperatorKind::Adjoint?grade-1:grade;
        const auto& tensor=source.tensors[input_grade*2].tensor;
        numeric::RationalVector input(tensor.domain().logical_size(),Q{0});
        for(const auto& entry:tensor.values().entries())input[entry.index]=entry.value;
        const auto lambda=kind==field::OperatorKind::Green?Q{1,2}:Q{0};
        const auto values=oracle.apply({kind,core::Grade{static_cast<std::uint16_t>(grade)},lambda},input);
        std::cout<<"{\"source\":\""<<name<<"\",\"kind\":"<<static_cast<unsigned>(kind)<<",\"grade\":"<<grade<<",\"input\":"<<(1000+input_grade)<<",\"values\":[";
        for(std::size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<'"'<<values[i]<<'"';}std::cout<<"]}\n";
    }
}
NativeAccessionRequest field_request(std::uint64_t key) {
    NativeAccessionRequest request;request.kind=NativeAccessionKind::Field;request.subject=key;return request;
}
inference::NativeExactTowerRequest learning_request(const NativeAccessionDirectory& directory,bool positive) {
    inference::NativeExactTowerRequest request;request.grade_carriers={10,11,12};
    request.metrics=positive?inference::NativeMetricSelection::NamedUnit:inference::NativeMetricSelection::DeclaredDiagonal;
    if(positive)return request;
    for(unsigned k=0;k<3;++k){auto metric=field_request(2000+k);metric.kind=NativeAccessionKind::Tensor;
        request.metric_tensors.push_back(directory.pin(metric));}
    return request;
}
void vector_json(std::ostream& output,const numeric::RationalVector& values) {
    output<<'[';for(std::size_t i=0;i<values.size();++i){if(i)output<<',';output<<'"'<<values[i]<<'"';}output<<']';
}
void transport_reference(const std::filesystem::path& out) {
    auto source=endpoint();
    core::RelationSignature signature;
    const auto type=signature.add_type("bridge-fixture",core::BoundaryPolicy::DeclaredExact,0,10);
    core::RelationalComplexBuilder builder(signature);
    for(const auto key:{10012,10010,10011})builder.declare_witness_identity(key);
    builder.add_grade_one(core::RelationKey{10101},type,{{10010,true,false,Q{1}},{10011,true,true,Q{1}}});
    builder.add_grade_one(core::RelationKey{10102},type,{{10011,true,false,Q{1}},{10012,true,true,Q{1}}});
    builder.add_grade_one(core::RelationKey{10100},type,{{10010,true,false,Q{1}},{10011,true,true,Q{1}},
        {10010,true,false,Q{2}},{10010,true,true,Q{2}},{10012,false,false,Q{7}}});
    builder.add_higher(core::Grade{2},core::RelationKey{10201},type,{});
    builder.add_higher(core::Grade{2},core::RelationKey{10200},type,{{2,true,false,Q{2}},{0,true,true,Q{2}}});
    const auto target=builder.freeze();
    const algebra::RelationalMorphism morphism(source.source,target,{{1,2,0},{2,0,1},{1,0}});
    std::ofstream json(out/"transport.json");json<<"{\"occurrences\":[";
    const auto& maps=morphism.occurrence_maps();
    for(std::size_t k=0;k<maps.size();++k) {
        if(k)json<<',';
        json<<'[';
        for(std::size_t i=0;i<maps[k].size();++i){if(i)json<<',';json<<maps[k][i];}
        json<<']';
    }
    json<<"],\"fields\":[";
    for(unsigned k=0;k<3;++k) {
        if(k)json<<',';
        const auto& tensor=source.tensors[2*k].tensor;
        numeric::RationalVector values(tensor.domain().logical_size(),Q{0});
        for(const auto& entry:tensor.values().entries())values[entry.index]=entry.value;
        vector_json(json,morphism.push_field(core::Grade{static_cast<std::uint16_t>(k)},values));
    }
    json<<"]}";
}
void learning_reference(const std::filesystem::path& out,bool positive) {
    auto source=endpoint(positive);
    source.carriers.push_back({40,TemporalCarrierKind::Explicit,core::Grade{0},{},900,numeric::IndexDomain::linear(4)});
    source.carriers.push_back({41,TemporalCarrierKind::Explicit,core::Grade{0},{},901,numeric::IndexDomain::linear(1)});
    TemporalTensor tensor;tensor.key=4000;tensor.role=TemporalTensorRole::Field;tensor.output=40;
    tensor.tensor={numeric::IndexDomain::linear(4),numeric::StructuralSupport::full(4),
        numeric::SparseRationalVector(4,{{0,Q{2}},{1,Q{-3}},{2,Q{4}},{3,Q{5}}})};source.tensors.push_back(tensor);
    tensor.key=4001;tensor.output=41;
    tensor.tensor={numeric::IndexDomain::linear(1),numeric::StructuralSupport::full(1),numeric::SparseRationalVector(1,{{0,Q{2,3}}})};source.tensors.push_back(tensor);
    tensor.key=4002;tensor.output=11;
    tensor.tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::full(3),numeric::SparseRationalVector(3,{{1,Q{1,2}},{2,Q{1,5}}})};source.tensors.push_back(tensor);
    tensor.key=4003;
    tensor.tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::full(3),numeric::SparseRationalVector(3,{{0,Q{-1,7}},{1,Q{2,3}}})};source.tensors.push_back(tensor);
    validate_temporal_endpoint(source);
    const std::string name=positive?"learning-unit":"learning-diagonal";
    save(out/(name+".rfcxs"),encode_native_series(NativeTemporalSeries(source,88)));
    NativeAccessionDirectory directory(source,88);
    const auto input=directory.pin(field_request(1001));
    const auto tower=inference::PreparedNativeExactTower::prepare(input,learning_request(directory,positive));
    using C=learning::ExactMetricChannelKind;
    const std::vector<learning::ExactMetricChannel> channels{{C::Identity,0},{C::HodgeDown,0},{C::HodgeUp,0},{C::Hodge,0}};
    const learning::NativeExactTrainingContract contract{learning::NativeTrainingObjective::HalfMetricSquaredError,
        {"bridge-training",900,901},700,9900,9901,9902,2,0,9903};
    auto trained=learning::NativeExactTrainer::prepare(tower,core::Grade{1},channels,
        directory.pin(field_request(4000)),directory.pin(field_request(4001)),
        {{input,input,learning::NativeSamplePartition::Training,3,4,5},
         {directory.pin(field_request(4002)),directory.pin(field_request(4003)),learning::NativeSamplePartition::Training,6,7,8}},contract);
    std::ofstream json(out/(name+".json"));json<<'[';
    for(unsigned step=0;step<=6;++step) {
        if(step)json<<',';
        NativeAccessionDirectory parameters(*trained.history().tail().pin(),700);
        const auto current_input=parameters.pin(field_request(1001));
        const auto current_tower=inference::PreparedNativeExactTower::prepare(current_input,learning_request(parameters,positive));
        const learning::PreparedNativeMetricLayer layer(current_tower,core::Grade{1},channels,
            parameters.pin(field_request(4000)),parameters.pin(field_request(4001)),contract.model);
        json<<"{\"step\":"<<step<<",\"parameters\":";vector_json(json,trained.parameters());
        json<<",\"values\":";vector_json(json,layer.forward(current_input).values());
        json<<",\"loss\":";if(trained.last_loss())json<<'"'<<*trained.last_loss()<<'"';else json<<"null";json<<'}';
        if(step==4)save(out/(name+"-trained.rfcxs"),encode_native_series(trained.history()));
        if(step<6)trained=trained.advance(Q{1,1000000},10,20);
    }
    json<<']';
    learning::IntegerRelationalNetwork network(source.source,core::Grade{1});
    using I=learning::OperatorChannelKind;
    network.add_layer(learning::CompiledIntegerRelationalLayer(source.source,core::Grade{1},
        {I::Identity,I::HodgeDown,I::HodgeUp,I::Hodge},{2,-1,1,0},3,learning::IntegerActivation::ReLU));
    network.add_layer(learning::CompiledIntegerRelationalLayer(source.source,core::Grade{1},
        {I::Identity,I::Hodge},{1,-1},-2,learning::IntegerActivation::Identity));
    const auto integer=network.forward(field::MachineLatticeField(source.source,core::Grade{1},{1,-2,3}));
    std::ofstream ints(out/(name+"-integer.json"));ints<<'[';
    for(std::size_t i=0;i<integer.values().size();++i){if(i)ints<<',';ints<<integer.values()[i];}ints<<']';
}
void integer_batch_reference(const std::filesystem::path& out) {
    const auto source=endpoint();
    using C=learning::OperatorChannelKind;
    const std::vector<C> channels{C::Identity,C::HodgeDown,C::HodgeUp,C::Hodge,C::HodgeDown};
    const std::vector<long> gains{2,-1,1,1,0};
    std::ofstream json(out/"integer-batch.json");json<<'[';bool first=true;
    for(unsigned k=0;k<3;++k)for(unsigned activation=0;activation<3;++activation)for(unsigned b:{1,2,8}) {
        const core::Grade grade{static_cast<std::uint16_t>(k)};
        const auto n=source.source.grade(grade).cell_count();
        std::vector<numeric::Integer> values(n*b);
        for(unsigned sample=0;sample<b;++sample) {
            std::vector<numeric::Integer> input(n),output(n,3);
            for(std::size_t i=0;i<n;++i)input[i]=static_cast<long>((i*b+sample)%7)-3;
            for(std::size_t c=0;c<channels.size();++c) {
                const auto response=learning::apply_operator_channel(channels[c],field::ExactLatticeField(source.source,grade,input));
                for(std::size_t i=0;i<n;++i)output[i]+=gains[c]*response.values()[i];
            }
            for(std::size_t i=0;i<n;++i) {
                if(activation==1 && output[i]<0)output[i]=0;
                if(activation==2)output[i]=output[i]>=0?1:-1;
                values[i*b+sample]=2*output[i]-1;
            }
        }
        if(!first)json<<',';first=false;
        json<<"{\"grade\":"<<k<<",\"activation\":"<<activation<<",\"batch\":"<<b<<",\"values\":[";
        for(std::size_t i=0;i<values.size();++i){if(i)json<<',';json<<'"'<<values[i]<<'"';}json<<"]}";
    }
    json<<']';
}
void rational_batch_reference(const std::filesystem::path& out) {
    const auto source=endpoint();
    std::ofstream json(out/"rational-batch.json");json<<'[';bool first=true;
    for(bool unit:{true,false}) {
        std::vector<field::ExactMetric> metrics;
        for(unsigned k=0;k<3;++k) {
            if(unit)metrics.push_back(field::ExactMetric::identity(source.source.grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count()));
            else metrics.push_back(field::ExactMetric::diagonal(k==2?numeric::RationalVector{Q{2},Q{3}}:numeric::RationalVector{Q{2},Q{3},Q{4}}));
        }
        const auto reference=field::ExactExecutionTower::weighted(source.source,std::move(metrics));
        using K=field::OperatorKind;
        for(unsigned k=0;k<3;++k)for(auto kind:{K::Boundary,K::Adjoint,K::HodgeDown,K::HodgeUp,K::Hodge}) {
            if(!k && (kind==K::Boundary || kind==K::Adjoint))continue;
            const auto input_grade=kind==K::Adjoint?k-1:k;
            std::vector<std::uint64_t> keys{1000+input_grade};
            if(input_grade==1)keys.push_back(3003);
            for(const auto key:keys) {
                const auto& tensor=std::find_if(source.tensors.begin(),source.tensors.end(),
                    [key](const auto& t){return t.key==key;})->tensor;
                numeric::RationalVector input(tensor.domain().logical_size(),Q{0});
                for(const auto& entry:tensor.values().entries())input[entry.index]=entry.value;
                const auto result=reference.apply({kind,core::Grade{static_cast<std::uint16_t>(k)},0},input);
                if(!first)json<<',';first=false;
                json<<"{\"metric\":\""<<(unit?"unit":"diagonal")<<"\",\"kind\":"<<static_cast<unsigned>(kind)
                    <<",\"grade\":"<<k<<",\"input\":"<<key<<",\"values\":";
                vector_json(json,result);json<<'}';
            }
        }
    }
    json<<']';
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    const std::filesystem::path out(argv[1]);std::filesystem::create_directories(out);
    transport_reference(out);
    integer_batch_reference(out);
    rational_batch_reference(out);
    for(bool positive:{false,true}) {
        auto initial=endpoint(positive);NativeTemporalSeries history(initial,88);
        const std::string name=positive?"positive":"diagonal";
        save(out/(name+".rfcxs"),encode_native_series(history));reference(initial,name);
        auto next=initial;next.sequence=1;
        next.tensors[2].tensor={numeric::IndexDomain::linear(3),numeric::StructuralSupport::full(3),numeric::SparseRationalVector(3,{{1,Q{5}}})};
        history.append(make_temporal_segment(initial,next));
        save(out/(name+"-history.rfcxs"),encode_native_series(history));
        learning_reference(out,positive);
    }
    NativeTemporalSeries empty(endpoint(false,true),88);
    save(out/"empty.rfcxs",encode_native_series(empty));
    auto constructed=endpoint();
    constructed.tensors.erase(std::remove_if(constructed.tensors.begin(),constructed.tensors.end(),
        [](const auto& t){return t.key==3003 || t.key==3004;}),constructed.tensors.end());
    std::stable_partition(constructed.tensors.begin(),constructed.tensors.end(),
        [](const auto& t){return t.role==TemporalTensorRole::Field;});
    constructed.carriers.push_back({40,TemporalCarrierKind::Explicit,core::Grade{0},{},900,numeric::IndexDomain::linear(2)});
    constructed.carriers.push_back({41,TemporalCarrierKind::Explicit,core::Grade{0},{},901,numeric::IndexDomain::linear(1)});
    TemporalTensor p;p.key=4000;p.role=TemporalTensorRole::Field;p.output=40;
    p.tensor={numeric::IndexDomain::linear(2),numeric::StructuralSupport::full(2),numeric::SparseRationalVector(2,{{0,Q{1}},{1,Q{-2}}})};
    const auto position=std::find_if(constructed.tensors.begin(),constructed.tensors.end(),[](const auto& t){return t.role==TemporalTensorRole::Metric;});
    const auto offset=static_cast<std::size_t>(position-constructed.tensors.begin());
    constructed.tensors.insert(constructed.tensors.begin()+static_cast<std::ptrdiff_t>(offset),p);
    p.key=4001;p.output=41;
    p.tensor={numeric::IndexDomain::linear(1),numeric::StructuralSupport::full(1),numeric::SparseRationalVector(1,{{0,Q{3}}})};
    constructed.tensors.insert(constructed.tensors.begin()+static_cast<std::ptrdiff_t>(offset+1),p);
    validate_temporal_endpoint(constructed);
    save(out/"constructed.rfcxs",encode_native_series(NativeTemporalSeries(constructed,88)));
    learning::ExactIntegerPerceptron perceptron({1,-2},3);
    const std::vector<learning::SparseIntSample> samples{{{{0,2},{1,1}},1},{{{0,-2},{1,-1}},-1},{{{0,1},{0,1},{1,2}},1}};
    std::ofstream oracle(out/"perceptron.json");oracle<<'[';
    std::size_t mistakes=0;
    for(unsigned step=0;step<=6;++step) {
        if(step)oracle<<',';oracle<<"{\"step\":"<<step<<",\"parameters\":[";
        for(const auto weight:perceptron.weights())oracle<<weight<<',';
        oracle<<perceptron.bias()<<"],\"mistakes\":"<<mistakes<<",\"scores\":[";
        for(std::size_t i=0;i<samples.size();++i){if(i)oracle<<',';oracle<<perceptron.score(samples[i].features);}
        oracle<<"],\"values\":[";
        for(std::size_t i=0;i<samples.size();++i){if(i)oracle<<',';oracle<<static_cast<int>(perceptron.predict(samples[i].features));}
        oracle<<"]}";
        if(step<6)mistakes=perceptron.train_epoch(samples,2);
    }
    oracle<<']';
}
