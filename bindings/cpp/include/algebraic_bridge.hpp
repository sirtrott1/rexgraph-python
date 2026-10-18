#pragma once
#include "bridge.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rexgraph::python {
struct AlgebraicCarrierInfo {
    bool present{};
    std::uint16_t grade{};
    std::uint32_t index{};
    std::size_t rank{}, degree{};
};
struct AlgebraicValueInfo {
    unsigned kind{};
    std::size_t rows{}, columns{}, coefficient_count{};
    AlgebraicCarrierInfo source, target;
    std::vector<std::vector<std::string>> coefficients;
};
struct AlgebraicArchiveInfo {
    std::string archive, source, certificate_shift;
    std::size_t degree{};
    unsigned certificate_kind{}, embedding_kind{}, chain_policy{};
    std::uint64_t certificate_prime{}, certificate_order{};
    bool has_involution{}, has_metric{}, has_fibers{}, flat{};
    std::vector<std::string> polynomial, involution, embedding;
    std::vector<std::size_t> source_sizes;
    std::vector<std::uint64_t> record_keys, reconstruction_keys;
    std::vector<unsigned> record_kinds;
};
class AlgebraicValue;
class AlgebraicGraph;
class AlgebraicSnapshot;
class AlgebraicRecord;
class AlgebraicWindow;
class AlgebraicQuery;
class AlgebraicGraphRevision;
class AlgebraicGraphDatabase;
class AlgebraicGradeTensor;
class AlgebraicGradeWindow;
class AlgebraicMetricField;
class AlgebraicMetricValues;
class AlgebraicArchive {
public:
    static std::vector<std::size_t> default_limits();
    static std::shared_ptr<AlgebraicArchive> load(const char*, std::size_t,
                                                const std::vector<std::size_t>&);
    static std::shared_ptr<AlgebraicArchive> from_source(const std::shared_ptr<Source>&,
        const std::vector<std::uint64_t>&, const std::vector<std::size_t>&);
    AlgebraicArchiveInfo info() const;
    std::shared_ptr<AlgebraicValue> record(std::uint64_t) const;
    std::shared_ptr<AlgebraicValue> vector(std::uint64_t, unsigned,
                                          const std::vector<std::vector<std::string>>&) const;
    std::shared_ptr<AlgebraicValue> scalar(const std::vector<std::string>&) const;
    std::shared_ptr<AlgebraicValue> reconstruction(std::uint64_t, unsigned,
                                                  const std::shared_ptr<AlgebraicValue>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicArchive(std::shared_ptr<const State>);
    friend class AlgebraicGraph;
    friend class AlgebraicSnapshot;
};
class AlgebraicValue {
public:
    AlgebraicValueInfo info(std::size_t) const;
    std::shared_ptr<AlgebraicValue> operation(unsigned, const std::shared_ptr<AlgebraicValue>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicValue(std::shared_ptr<const State>);
    friend class AlgebraicArchive;
    friend class AlgebraicGraph;
    friend class AlgebraicSnapshot;
    friend class AlgebraicRecord;
    friend class AlgebraicWindow;
    friend class AlgebraicQuery;
    friend class AlgebraicGradeTensor;
    friend class AlgebraicGradeWindow;
    friend class AlgebraicMetricField;
    friend class AlgebraicMetricValues;
};
struct AlgebraicGraphInfo {
    std::string image;
    std::vector<std::vector<std::uint64_t>> keys;
};
struct AlgebraicComparisonInfo {
    std::uint64_t source_field{}, target_field{}, source_bundle{}, target_bundle{},
        source_domain{}, target_domain{}, scalar_key{}, carrier_key{},
        source_archive{}, target_archive{}, source_record{}, target_record{},
        accession_key{}, reconstruction_key{};
    AlgebraicCarrierInfo source, target;
    std::size_t coefficient_count{}, action_rows{}, action_columns{};
    std::vector<std::vector<std::string>> coefficients;
    std::vector<std::string> action;
};
class AlgebraicGraph {
public:
    static std::vector<std::size_t> default_limits();
    static std::shared_ptr<AlgebraicGraph> load(const char*, std::size_t,
                                               const std::vector<std::size_t>&);
    AlgebraicGraphInfo info() const;
    std::vector<std::uint64_t> archive_refs(std::uint64_t) const;
    std::shared_ptr<AlgebraicArchive> archive(std::uint64_t) const;
    AlgebraicComparisonInfo comparison(unsigned, std::uint64_t, std::size_t) const;
    std::shared_ptr<AlgebraicValue> compare(unsigned, std::uint64_t, unsigned,
        const std::shared_ptr<AlgebraicValue>&, std::uint64_t) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicGraph(std::shared_ptr<const State>);
    friend class AlgebraicGraphRevision;
};
struct AlgebraicGraphSourceInfo {
    std::uint64_t bundle{};
    unsigned kind{};
    std::vector<std::vector<std::size_t>> cells, occurrences;
};
struct AlgebraicGraphRevisionInfo {
    std::string image;
    std::uint64_t name_space{}, sequence{}, anchor{};
    std::vector<AlgebraicGraphSourceInfo> sources;
    std::vector<std::vector<std::uint64_t>> archives;
};
class AlgebraicGraphRevision {
public:
    static std::vector<std::size_t> default_limits();
    static std::shared_ptr<AlgebraicGraphRevision> capture(const std::shared_ptr<Source>&,
        const std::vector<std::uint64_t>&, const std::vector<std::size_t>&);
    AlgebraicGraphRevisionInfo info() const;
    std::shared_ptr<AlgebraicGraph> graph() const;
    bool same_owner(const std::shared_ptr<AlgebraicGraphRevision>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicGraphRevision(std::shared_ptr<const State>);
    friend class AlgebraicGraphDatabase;
};
struct AlgebraicGraphDatabaseBuild {
    std::shared_ptr<AlgebraicGraphDatabase> database;
    ExecutionResult status;
};
struct AlgebraicGraphArchiveSelection {
    std::shared_ptr<AlgebraicSnapshot> snapshot;
    ExecutionResult status;
};
class AlgebraicGraphDatabase {
public:
    static AlgebraicGraphDatabaseBuild select(const std::shared_ptr<AlgebraicGraphRevision>&,
        std::uint64_t, std::uint64_t, const std::vector<std::size_t>&, const std::shared_ptr<Cancellation>&);
    AlgebraicGraphArchiveSelection archive(std::uint64_t, std::uint64_t, std::uint64_t,
        const std::vector<std::size_t>&, const std::shared_ptr<Cancellation>&) const;
    std::vector<std::size_t> requirements() const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicGraphDatabase(std::shared_ptr<const State>);
};
struct AlgebraicRecordInfo {
    std::uint64_t key{};
    std::vector<std::size_t> domain;
    AlgebraicValueInfo value;
};
struct AlgebraicOccurrenceInfo {
    std::uint64_t key{};
    std::string coefficient;
    bool active{};
    std::shared_ptr<AlgebraicValue> underlying, weighted, contribution;
};
class AlgebraicRecord {
public:
    AlgebraicRecordInfo info(std::size_t) const;
    std::shared_ptr<AlgebraicValue> value() const;
    std::shared_ptr<AlgebraicValue> coordinate(const std::vector<std::size_t>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicRecord(std::shared_ptr<const State>);
    friend class AlgebraicSnapshot;
    friend class AlgebraicWindow;
    friend class AlgebraicQueryResult;
};
class AlgebraicSnapshot {
public:
    static std::shared_ptr<AlgebraicSnapshot> publish(const std::shared_ptr<AlgebraicArchive>&,
        std::uint64_t, std::uint64_t);
    bool same_publication(const std::shared_ptr<AlgebraicSnapshot>&) const;
    std::vector<std::uint64_t> graph_origin() const;
    std::shared_ptr<AlgebraicRecord> record(std::uint64_t) const;
    AlgebraicOccurrenceInfo occurrence(std::uint16_t, std::size_t) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicSnapshot(std::shared_ptr<const State>);
    friend class AlgebraicMetricField;
    friend class AlgebraicQuery;
    friend class AlgebraicGradeTensor;
    friend class AlgebraicGraphDatabase;
};
struct AlgebraicWindowInfo {
    std::vector<std::size_t> domain, source_domain, lower, upper, requirements;
    std::size_t supported_count{};
};
struct AlgebraicWindowValues {
    std::vector<std::vector<std::string>> coefficients;
    std::vector<std::size_t> source_indices;
};
struct AlgebraicWindowBuild { std::shared_ptr<AlgebraicWindow> window; ExecutionResult status; };
struct AlgebraicSum { std::shared_ptr<AlgebraicValue> value; ExecutionResult status; };
class AlgebraicWindow {
public:
    static AlgebraicWindowBuild prepare(const std::shared_ptr<AlgebraicRecord>&,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&,
        std::uint64_t, std::uint64_t, const std::vector<std::size_t>&, const std::shared_ptr<Cancellation>&);
    AlgebraicWindowInfo info() const;
    AlgebraicWindowValues values(std::size_t) const;
    std::size_t source_index(std::size_t) const;
    std::vector<std::size_t> window_index(std::size_t) const;
    std::shared_ptr<AlgebraicValue> coordinate(std::size_t) const;
    AlgebraicSum sum(std::uint64_t, std::uint64_t, const std::vector<std::size_t>&,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicWindow(std::shared_ptr<const State>);
};
struct AlgebraicPredicate {
    std::vector<std::size_t> coordinate;
    unsigned operation{};
    std::shared_ptr<AlgebraicValue> literal;
};
struct AlgebraicGradeInfo {
    std::uint16_t grade{};
    bool derived{};
    std::size_t cells{}, coordinates{};
    std::vector<std::uint64_t> keys;
    std::vector<std::size_t> offsets, requirements;
};
struct AlgebraicGradeValues {
    std::vector<std::vector<std::string>> coefficients;
    std::vector<std::vector<std::size_t>> coordinates;
};
struct AlgebraicGradeBuild {std::shared_ptr<AlgebraicGradeTensor> tensor;ExecutionResult status;};
class AlgebraicGradeTensor {
public:
    static AlgebraicGradeBuild select(const std::shared_ptr<AlgebraicSnapshot>&,std::uint16_t,
        const std::vector<std::uint64_t>&,std::uint64_t,std::uint64_t,
        const std::vector<std::size_t>&,const std::shared_ptr<Cancellation>&);
    AlgebraicGradeBuild boundary(std::uint64_t,std::uint64_t,const std::vector<std::size_t>&,
        const std::shared_ptr<Cancellation>&) const;
    AlgebraicGradeInfo info() const;
    AlgebraicGradeValues values(std::size_t) const;
    std::vector<std::size_t> source_coordinate(std::size_t) const;
    std::size_t source_index(std::uint16_t,std::uint32_t,std::size_t) const;
    std::shared_ptr<AlgebraicValue> coordinate(std::size_t) const;
    std::shared_ptr<AlgebraicValue> stalk(std::uint32_t) const;
    bool same_owner(const std::shared_ptr<AlgebraicGradeTensor>&) const;
    bool matches_source(const std::shared_ptr<AlgebraicSnapshot>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicGradeTensor(std::shared_ptr<const State>);
    friend class AlgebraicGradeWindow;
    friend class AlgebraicMetricField;
};
struct AlgebraicMetricBuild {std::shared_ptr<AlgebraicMetricField> field;ExecutionResult status;};
struct AlgebraicMetricAction {std::shared_ptr<AlgebraicMetricValues> values;ExecutionResult status;};
class AlgebraicMetricField {
public:
    static AlgebraicMetricBuild prepare(const std::shared_ptr<AlgebraicSnapshot>&,
        const std::vector<std::uint64_t>&,std::size_t,std::uint64_t,std::uint64_t,
        const std::shared_ptr<Cancellation>&);
    AlgebraicMetricAction apply(const std::shared_ptr<AlgebraicGradeTensor>&,unsigned,
        std::uint64_t,std::uint64_t,const std::shared_ptr<Cancellation>&) const;
    AlgebraicSum scalar(const std::shared_ptr<AlgebraicGradeTensor>&,
        const std::shared_ptr<AlgebraicGradeTensor>&,unsigned,std::uint64_t,std::uint64_t,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicMetricField(std::shared_ptr<const State>);
};
class AlgebraicMetricValues {
public:
    AlgebraicGradeInfo info() const;
    AlgebraicGradeValues values(std::size_t) const;
    std::shared_ptr<AlgebraicValue> stalk(std::uint32_t) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicMetricValues(std::shared_ptr<const State>);
    friend class AlgebraicMetricField;
};
struct AlgebraicGradeWindowBuild {std::shared_ptr<AlgebraicGradeWindow> window;ExecutionResult status;};
class AlgebraicGradeWindow {
public:
    static AlgebraicGradeWindowBuild prepare(const std::shared_ptr<AlgebraicGradeTensor>&,
        const std::vector<std::size_t>&,const std::vector<std::size_t>&,const std::vector<std::size_t>&,
        std::uint64_t,std::uint64_t,const std::vector<std::size_t>&,const std::shared_ptr<Cancellation>&);
    AlgebraicWindowInfo info() const;
    AlgebraicGradeValues values(std::size_t) const;
    std::size_t source_index(std::size_t) const;
    std::vector<std::size_t> source_coordinate(std::size_t) const;
    std::vector<std::size_t> window_index(std::size_t) const;
    std::shared_ptr<AlgebraicValue> coordinate(std::size_t) const;
    AlgebraicSum sum(std::uint64_t,std::uint64_t,const std::vector<std::size_t>&,
        const std::shared_ptr<Cancellation>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicGradeWindow(std::shared_ptr<const State>);
};

class AlgebraicQueryResult;
class AlgebraicQueryCache;
struct AlgebraicQueryBuild { std::shared_ptr<AlgebraicQuery> query; ExecutionResult status; };
struct AlgebraicQueryExecution { std::shared_ptr<AlgebraicQueryResult> result; ExecutionResult status; };
struct AlgebraicQueryInfo { std::size_t candidate_count{}; std::vector<std::size_t> requirements; };
struct AlgebraicQueryResultInfo {
    std::vector<std::uint64_t> keys;
    std::uint64_t name_space{}, max_sequence{};
    bool cache_hit{};
    std::vector<std::size_t> cpu;
};
class AlgebraicQuery {
public:
    static AlgebraicQueryBuild prepare(const std::shared_ptr<AlgebraicSnapshot>&, int,
        const std::shared_ptr<AlgebraicValue>&, const std::shared_ptr<AlgebraicValue>&,
        const std::vector<AlgebraicPredicate>&, std::uint64_t, std::uint64_t,
        const std::vector<std::size_t>&, std::size_t, const std::shared_ptr<Cancellation>&);
    AlgebraicQueryInfo info() const;
    AlgebraicQueryExecution execute(std::uint64_t, std::uint64_t, const std::vector<std::size_t>&,
        std::size_t, std::size_t, std::size_t, const std::shared_ptr<Cancellation>&,
        const std::shared_ptr<AlgebraicQueryCache>&) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicQuery(std::shared_ptr<const State>);
};
class AlgebraicQueryResult {
public:
    AlgebraicQueryResultInfo info() const;
    std::shared_ptr<AlgebraicRecord> record(std::size_t) const;
private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit AlgebraicQueryResult(std::shared_ptr<const State>);
    friend class AlgebraicQuery;
};
class AlgebraicQueryCache {
public:
    explicit AlgebraicQueryCache(std::size_t);
    void clear();
    std::vector<std::size_t> counts() const;
private:
    struct State;
    std::shared_ptr<State> state_;
    friend class AlgebraicQuery;
};
} // namespace rexgraph::python
