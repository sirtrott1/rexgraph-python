#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rfc::storage { struct TemporalEndpoint; class NativeSeriesEndpoint; }

namespace rexgraph::python {
struct SourceInfo {
    std::uint64_t name_space{}, sequence{}, first_sequence{};
    std::string archive, endpoint;
    std::vector<std::vector<std::string>> coordinates;
    std::vector<std::uint64_t> carrier_keys, tensor_keys, tensor_carriers;
    std::vector<unsigned> carrier_grades, carrier_kinds, tensor_roles, tensor_has_output;
};
struct FieldInfo {
    std::uint64_t key{}, carrier{};
    std::vector<std::string> values;
    std::vector<std::size_t> support;
};
struct ExecutionResult {
    std::vector<std::string> values;
    std::uint64_t input_carrier{}, output_carrier{}, input_field{}, name_space{}, sequence{};
    std::uint16_t output_grade{};
    std::string profile, implementation, error_message, error_resource;
    std::size_t integer_bits{}, scalar_work{}, error_required{}, error_limit{};
    unsigned error{};
};
class Cancellation {
public:
    Cancellation();
    void cancel();
    bool cancelled() const;
private:
    struct State;
    std::shared_ptr<State> state_;
    std::shared_ptr<const std::atomic_bool> flag() const;
    friend struct AlgebraicExecutionAccess;
    friend class Plan;
    friend class IntegerNetwork;
    friend class Trainer;
    friend class Perceptron;
    friend class Transport;
    friend class Source;
};
class Field;
class Plan;
class IntegerInput;
class IntegerNetwork;
class IntegerBatch;
struct IntegerBatchBuild {
    std::shared_ptr<IntegerBatch> batch;
    ExecutionResult status;
};
struct IntegerBatchResult {
    ExecutionResult status;
    std::vector<std::int64_t> values;
    std::size_t cells{}, samples{}, workspace_capacity{}, workspace_slots{};
};
struct RationalBatchResult {
    ExecutionResult status;
    std::vector<std::uint64_t> input_fields;
    std::size_t cells{}, samples{}, workspace_capacity{}, workspace_slots{};
};
class Trainer;
struct IntegerLayer {
    std::vector<unsigned> channels;
    std::vector<std::int64_t> gains;
    std::int64_t bias{};
    unsigned activation{};
};
struct TrainingSample {
    std::uint64_t input{}, target{}, event_time{}, knowledge_time{}, label_available_time{};
    unsigned partition{};
};
struct TrainingRequest {
    std::uint16_t grade{};
    std::vector<unsigned> channels;
    std::vector<TrainingSample> samples;
    std::string model;
    std::uint64_t gains{}, bias{}, channel_coordinates{}, bias_coordinates{}, history_namespace{},
        metadata_carrier{}, metadata_tensor{}, time_coordinates{}, metadata_coordinates{},
        initial_parameter_knowledge_time{}, publication_cutoff{};
    std::size_t metadata_capacity{4096};
};
struct TrainingInfo {
    std::string archive, loss;
    std::vector<std::string> parameters;
    std::uint64_t step{}, history_namespace{};
    std::uint64_t parameter_knowledge_time{};
    bool has_loss{};
};
struct TrainingExecutionInfo {
    std::string profile;
    std::vector<std::string> response_channels;
    std::vector<std::vector<std::size_t>> parameter_coordinates;
    std::size_t capacity{}, cached_samples{}, retained_slots{}, hits{}, misses{};
    std::size_t scalar_work{}, integer_bits{}, rational_slots{};
};
struct TrainingUpdate {
    std::shared_ptr<Trainer> trainer;
    ExecutionResult status;
};
struct SourceType { std::string name; std::size_t min_arity{}, max_arity{}; };
struct SourceOccurrence { std::uint64_t key{}, lower{}; bool exists{}, negative{}; std::string magnitude; std::vector<unsigned> aspects; };
struct SourceRelation { std::uint16_t grade{}; std::uint64_t key{}; unsigned type{}; std::vector<SourceOccurrence> occurrences; };
struct SourceCarrier { std::uint64_t key{}, coordinates{}; unsigned kind{}; std::uint16_t grade{}; std::size_t size{}; };
struct SourceField { std::uint64_t key{}, carrier{}; std::vector<std::size_t> indices,support; std::vector<std::string> values; bool full{}; };
struct SourceMetric { std::uint64_t key{}, carrier{}; std::vector<std::string> diagonal; };
struct SourceDeclaration {
    std::uint64_t name_space{}; std::uint16_t max_grade{}, semantic_scale{};
    std::vector<std::string> role_aspects;
    std::vector<SourceType> types; std::vector<std::uint64_t> witnesses;
    std::vector<SourceRelation> relations; std::vector<SourceCarrier> carriers;
    std::vector<SourceField> fields; std::vector<SourceMetric> metrics;
};
struct SparseFeature { unsigned index{}; std::int64_t value{}; };
struct PerceptronInfo { std::vector<std::int64_t> weights; std::int64_t bias{}; std::size_t mistakes{}; };
struct PackedPerceptronInfo { std::vector<std::int64_t> parameters; std::size_t mistakes{}; };
class Perceptron;
class PerceptronRows {
public:
    std::size_t rows() const;
    std::size_t entries() const;
    std::size_t features() const;
private:
    struct State; std::shared_ptr<const State> state_;
    explicit PerceptronRows(std::shared_ptr<const State>);
    friend class Perceptron;
};
struct PerceptronRowsBuild { std::shared_ptr<PerceptronRows> rows; ExecutionResult status; };
struct PerceptronUpdate { std::shared_ptr<Perceptron> model; ExecutionResult status; };
struct PerceptronOutput {
    std::vector<std::int64_t> labels,scores;
    ExecutionResult status;
    std::size_t requested_workers{}, worker_slots{}, launched_threads{}, tiles{}, rows_per_tile{};
};
class Perceptron {
public:
    Perceptron(const std::vector<std::int64_t>&,std::int64_t,const std::vector<std::size_t>&);
    PerceptronInfo info() const;
    PackedPerceptronInfo packed_info() const;
    PerceptronRowsBuild prepare_rows(const std::vector<std::vector<SparseFeature>>&,
        const std::shared_ptr<Cancellation>&) const;
    PerceptronOutput forward_prepared(const std::shared_ptr<PerceptronRows>&,
        std::size_t, std::size_t, const std::shared_ptr<Cancellation>&) const;
    PerceptronOutput forward(const std::vector<std::vector<SparseFeature>>&,const std::shared_ptr<Cancellation>&) const;
    PerceptronOutput forward_cpu(const std::vector<std::vector<SparseFeature>>&,
        std::size_t, std::size_t, const std::shared_ptr<Cancellation>&) const;
    PerceptronUpdate advance(const std::vector<std::vector<SparseFeature>>&,const std::vector<std::int64_t>&,
        std::int64_t,const std::shared_ptr<Cancellation>&) const;
private:
    struct State; std::shared_ptr<State> state_;
    explicit Perceptron(std::shared_ptr<State>);
    friend class Source;
};
struct CarrierLink {
    std::uint64_t source{}, target{};
    std::vector<unsigned> indices;
};
struct TransportInfo { std::vector<std::vector<std::size_t>> occurrences; };
class Transport;
struct TransportBuild { std::shared_ptr<Transport> transport; ExecutionResult status; };
class Transport {
public:
    TransportInfo info() const;
    ExecutionResult push(std::uint64_t, const std::vector<std::string>&,
        const std::shared_ptr<Cancellation>&) const;
    ExecutionResult check_tensor(std::uint64_t, std::uint64_t,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    std::shared_ptr<State> state_;
    explicit Transport(std::shared_ptr<State>);
    friend class Source;
};
class Source {
public:
    static std::shared_ptr<Source> load(const char*, std::size_t, std::size_t,
        std::size_t max_dimension=1000000,std::size_t max_records=1000000,std::size_t max_logical_size=0);
    static std::shared_ptr<Source> build(const SourceDeclaration&, std::size_t);
    SourceInfo info() const;
    SourceCarrier carrier(std::uint64_t) const;
    TransportBuild transport(const std::shared_ptr<Source>&,
        const std::vector<std::vector<unsigned>>&, const std::vector<CarrierLink>&,
        const std::vector<std::size_t>&, const std::shared_ptr<Cancellation>&) const;
    std::shared_ptr<Source> at(std::uint64_t) const;
    std::shared_ptr<Field> field(std::uint64_t) const;
    std::shared_ptr<Plan> prepare(const std::vector<std::uint64_t>&,
        const std::vector<std::uint64_t>&, unsigned, bool, std::size_t,
        const std::vector<std::size_t>&) const;
    std::shared_ptr<IntegerInput> integer_input(std::uint16_t, std::uint64_t,
        const std::vector<std::int64_t>&) const;
    IntegerBatchBuild integer_batch(std::uint16_t, std::uint64_t,
        std::vector<std::int64_t>, std::size_t, const std::vector<std::size_t>&) const;
    std::shared_ptr<IntegerNetwork> integer_network(std::uint16_t, std::uint64_t,
        const std::vector<IntegerLayer>&, const std::vector<std::size_t>&) const;
    PerceptronUpdate perceptron(std::uint64_t, std::uint64_t,
        const std::vector<std::size_t>&, const std::shared_ptr<Cancellation>&) const;
    std::shared_ptr<Trainer> trainer(const std::shared_ptr<Plan>&, const TrainingRequest&,
        const std::string&, std::size_t) const;
private:
    struct State;
    explicit Source(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    std::shared_ptr<const rfc::storage::TemporalEndpoint> endpoint() const;
    rfc::storage::NativeSeriesEndpoint native_endpoint() const;
    std::size_t retained_revisions() const;
    friend class AlgebraicGraphRevision;
    friend class AlgebraicArchive;
    friend class PhaseFoldArchive;
    friend class ChainMap;
    friend class IntegerInput;
    friend class IntegerNetwork;
    friend class IntegerBatch;
    friend class Trainer;
    friend class Transport;
};
class Field {
public:
    std::uint64_t key() const;
    std::uint64_t carrier() const;
    FieldInfo info() const;
private:
    struct State;
    explicit Field(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
    friend class Plan;
    friend class Trainer;
};
class Plan {
public:
    ExecutionResult execute(const std::shared_ptr<Field>&, unsigned, std::uint16_t,
        const std::string&, const std::shared_ptr<Cancellation>&) const;
    RationalBatchResult execute_batch(const std::vector<std::shared_ptr<Field>>&,
        unsigned, std::uint16_t, const std::string&, const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    explicit Plan(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
    friend class Trainer;
};
class IntegerInput {
public:
    std::vector<std::int64_t> values() const;
    std::uint64_t carrier() const;
    std::uint16_t grade() const;
private:
    struct State;
    explicit IntegerInput(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
    friend class IntegerNetwork;
};
class IntegerNetwork {
public:
    ExecutionResult forward(const std::shared_ptr<IntegerInput>&,
        const std::shared_ptr<Cancellation>&) const;
    IntegerBatchResult forward_batch(const std::shared_ptr<IntegerBatch>&,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    explicit IntegerNetwork(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
};
class IntegerBatch {
public:
    std::vector<std::int64_t> values() const;
    std::uint64_t carrier() const;
    std::uint16_t grade() const;
    std::size_t cells() const;
    std::size_t samples() const;
private:
    struct State;
    explicit IntegerBatch(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
    friend class IntegerNetwork;
};
class Trainer {
public:
    TrainingInfo info() const;
    std::uint64_t parameter_knowledge_time() const noexcept;
    TrainingExecutionInfo training_execution() const;
    TrainingUpdate advance(const std::string&, std::uint64_t, std::uint64_t, std::size_t,
        const std::shared_ptr<Cancellation>&) const;
    ExecutionResult forward(const std::shared_ptr<Field>&,
        const std::shared_ptr<Cancellation>&) const;
    RationalBatchResult forward_batch(const std::vector<std::shared_ptr<Field>>&,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    explicit Trainer(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class Source;
};
std::string engine_digest();
std::string engine_version();
std::string binding_digest();
} // namespace rexgraph::python
