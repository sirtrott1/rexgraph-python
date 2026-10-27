#pragma once
#include <atomic>
#include <rfc/inference/c_declaration_archive.hpp>
#include <sstream>

namespace rexgraph::python {
using CQ = rfc::numeric::Rational;
using CM = rfc::numeric::ExactMatrix;
using CG = rfc::numeric::ExactActionGuard;
using CK = rfc::storage::CDeclarationKind;
using CA = rfc::inference::CDeclarationArchive;
inline std::string c_bytes(std::span<const std::byte> b) {
  return {reinterpret_cast<const char *>(b.data()), b.size()};
}
inline std::span<const std::byte> c_span(const std::string &b) {
  return std::as_bytes(std::span(b.data(), b.size()));
}
inline CQ c_rational(const std::string &s) {
  if (s.size() > 65536)
    throw std::length_error("exact scalar text limit");
  return rfc::numeric::canonical_rational(CQ{s, 16});
}
struct CRecordData {
  std::uint64_t kind{}, key{}, revision{};
  std::vector<std::size_t> dependencies;
  std::string payload;
};
struct CMetaData {
  std::uint64_t producer_namespace{}, producer{}, revision{},
      evidence_namespace{}, evidence_cutoff{};
  std::string purpose;
};
struct CItem {
  std::string name, kind;
  std::size_t rows{}, cols{};
  std::vector<std::string> values;
  std::vector<std::uint64_t> integers;
  std::string bytes;
};
struct CInspection {
  std::string type;
  std::vector<CItem> items;
};
class CCancel {
  std::atomic_bool flag_{false};

public:
  void cancel() { flag_ = true; }
  void check() const {
    if (flag_)
      throw std::runtime_error("C workflow cancelled");
  }
};
inline rfc::storage::CDeclarationLimits
c_limits(const std::vector<std::size_t> &v) {
  if (v.size() != 6)
    throw std::invalid_argument("six C declaration limits required");
  return {v[0], v[1], v[2], v[3], v[4], v[5]};
}
inline rfc::numeric::ExactActionLimits
c_exact(const std::vector<std::size_t> &v) {
  if (v.size() != 4)
    throw std::invalid_argument("four exact-action limits required");
  return {v[0], v[1], v[2], v[3]};
}
class CPayloadBridge {
  rfc::storage::CDeclarationPayload p_;

public:
  explicit CPayloadBridge(std::size_t maximum) : p_(maximum) {}
  void integer(std::uint64_t q) { p_.u64(q); }
  void flag(bool q) { p_.flag(q); }
  void rational(const std::string &q) { p_.rational(c_rational(q)); }
  void vector(const std::vector<std::string> &v) {
    rfc::numeric::RationalVector q;
    for (const auto &s : v)
      q.push_back(c_rational(s));
    p_.vector(q);
  }
  void matrix(std::size_t rows, std::size_t cols,
              const std::vector<std::string> &v) {
    if (rfc::machine::checked_size_mul(rows, cols) != v.size())
      throw std::invalid_argument(
          "matrix requires its complete declared carrier");
    CM m(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j)
        m.set(i, j, c_rational(v[i * cols + j]));
    p_.matrix(m);
  }
  void blob(const std::string &s) { p_.blob(c_span(s)); }
  void text(const std::string &s) { p_.string(s); }
  std::string take() {
    auto b = p_.take();
    return c_bytes(b);
  }
};
class CWorkflowBridge {
  CA archive_;
  rfc::storage::CDeclarationLimits limits_;
  rfc::numeric::ExactActionLimits exact_;
  std::shared_ptr<CCancel> cancel_;
  CWorkflowBridge(CA a, rfc::storage::CDeclarationLimits l,
                  rfc::numeric::ExactActionLimits e,
                  std::shared_ptr<CCancel> cancel)
      : archive_(std::move(a)), limits_(l), exact_(e),
        cancel_(std::move(cancel)) {}
  static rfc::storage::CDeclarationMetadata metadata(const CMetaData &m) {
    return {m.producer_namespace, m.producer,        m.revision,
            m.evidence_namespace, m.evidence_cutoff, m.purpose};
  }

public:
  static std::shared_ptr<CWorkflowBridge>
  decode(const std::string &bytes, const std::vector<std::size_t> &limits,
         const std::vector<std::size_t> &exact,
         const std::shared_ptr<CCancel> &cancel) {
    auto l = c_limits(limits);
    auto e = c_exact(exact);
    CG g(e, [cancel] { cancel->check(); });
    auto a = rfc::inference::decode_c_declaration_archive(c_span(bytes), g, l);
    return std::shared_ptr<CWorkflowBridge>(
        new CWorkflowBridge(std::move(a), l, e, cancel));
  }
  static std::shared_ptr<CWorkflowBridge>
  prepare(const CMetaData &meta, const std::vector<CRecordData> &rows,
          const std::vector<std::size_t> &exports,
          const std::vector<std::size_t> &limits,
          const std::vector<std::size_t> &exact,
          const std::shared_ptr<CCancel> &cancel) {
    auto l = c_limits(limits);
    auto e = c_exact(exact);
    if (rows.size() > l.max_records)
      throw std::length_error("C declaration record count");
    std::vector<rfc::storage::CDeclarationRecord> records;
    for (const auto &r : rows) {
      if (r.kind < 1 || r.kind > 66)
        throw std::invalid_argument("unknown C constructor");
      auto b = c_span(r.payload);
      records.push_back({static_cast<CK>(r.kind),
                         r.key,
                         r.revision,
                         r.dependencies,
                         {b.begin(), b.end()}});
    }
    CG g(e, [cancel] { cancel->check(); });
    auto a = CA::prepare(metadata(meta), std::move(records), exports, g, l);
    return std::shared_ptr<CWorkflowBridge>(
        new CWorkflowBridge(std::move(a), l, e, cancel));
  }
  std::string encode() const {
    auto b = rfc::inference::encode_c_declaration_archive(archive_, limits_);
    return c_bytes(b);
  }
  std::vector<CRecordData> records() const {
    std::vector<CRecordData> out;
    for (const auto &r : archive_.declarations())
      out.push_back({static_cast<std::uint64_t>(r.kind), r.key, r.revision,
                     r.dependencies, c_bytes(r.payload)});
    return out;
  }
  std::vector<std::size_t> exports() const { return archive_.exports(); }
  CMetaData meta() const {
    const auto &m = archive_.metadata();
    return {m.producer_namespace, m.producer,        m.revision,
            m.evidence_namespace, m.evidence_cutoff, m.purpose};
  }
  bool same(const std::shared_ptr<CWorkflowBridge> &other) const {
    return this == other.get();
  }
  class Inspect {
    CInspection out_;
    std::size_t remaining_;
    std::size_t bit_remaining_ = 1u << 26;
    std::string rational(const CQ &q) {
      const auto bits =
          rfc::machine::checked_size_add(mpz_sizeinbase(q.get_num_mpz_t(), 2),
                                         mpz_sizeinbase(q.get_den_mpz_t(), 2));
      if (bits > bit_remaining_)
        throw std::length_error("C observation coefficient bit limit");
      bit_remaining_ -= bits;
      return q.get_str(16);
    }
    void charge(std::size_t n) {
      if (n > remaining_)
        throw std::length_error("complete C value inspection limit");
      remaining_ -= n;
    }

  public:
    Inspect(std::string type, std::size_t max)
        : out_{std::move(type), {}}, remaining_(max) {}
    void text(std::string name, std::string s) {
      charge(s.size());
      out_.items.push_back(
          {std::move(name), "text", 0, 0, {}, {}, std::move(s)});
    }
    void bytes(std::string name, std::string s) {
      charge(s.size());
      out_.items.push_back(
          {std::move(name), "bytes", 0, 0, {}, {}, std::move(s)});
    }
    void scalar(std::string name, const CQ &q) {
      charge(1);
      out_.items.push_back(
          {std::move(name), "rational", 1, 1, {rational(q)}, {}, {}});
    }
    void flag(std::string name, bool b) {
      charge(1);
      out_.items.push_back(
          {std::move(name), "boolean", 1, 1, {}, {b ? 1u : 0u}, {}});
    }
    template <class T> void integers(std::string name, const T &xs) {
      charge(xs.size());
      CItem i;
      i.name = std::move(name);
      i.kind = "integers";
      for (auto x : xs)
        i.integers.push_back(static_cast<std::uint64_t>(x));
      out_.items.push_back(std::move(i));
    }
    template <class T> void vector(std::string name, const T &xs) {
      charge(xs.size());
      CItem i;
      i.name = std::move(name);
      i.kind = "vector";
      for (const auto &x : xs)
        i.values.push_back(rational(CQ{x}));
      out_.items.push_back(std::move(i));
    }
    template <class T> void matrix(std::string name, const T &m) {
      charge(rfc::machine::checked_size_mul(m.rows(), m.cols()));
      CItem i;
      i.name = std::move(name);
      i.kind = "matrix";
      i.rows = m.rows();
      i.cols = m.cols();
      for (std::size_t r = 0; r < m.rows(); ++r)
        for (std::size_t c = 0; c < m.cols(); ++c)
          i.values.push_back(rational(CQ{m.get(r, c)}));
      out_.items.push_back(std::move(i));
    }
    void matrices(std::string name, const std::vector<CM> &ms) {
      for (std::size_t k = 0; k < ms.size(); ++k)
        matrix(name + "." + std::to_string(k), ms[k]);
    }
    template <class T> void formal(std::string name, const T &m) {
      integers(name + ".shape", std::vector<std::size_t>{m.rows, m.cols});
      for (std::size_t i = 0; i < m.values.size(); ++i) {
        vector(name + "." + std::to_string(i) + ".numerator",
               m.values[i].numerator());
        vector(name + "." + std::to_string(i) + ".denominator",
               m.values[i].denominator());
      }
    }
    CInspection take() { return out_; }
  };
  CInspection inspect(std::size_t at, std::size_t maximum) const {
    using namespace rfc;
    using namespace temporal;
    using namespace field;
    using namespace geometry;
    using namespace inference;
    const auto &record = archive_.declarations().at(at);
    Inspect out("native-C-value-v1", maximum);
    out.integers("constructor", std::vector<std::uint64_t>{
                                    static_cast<std::uint64_t>(record.kind),
                                    record.key, record.revision});
    out.integers("dependencies", record.dependencies);
    CG g(exact_, [this] { cancel_->check(); });
    cancel_->check();
    switch (record.kind) {
    case CK::Source: {
      const auto &x = archive_.get<core::OwnedRelationalSource>(at);
      out.bytes("source", c_bytes(storage::encode_rfcx(x.source()).bytes));
      std::vector<std::size_t> sizes;
      for (std::size_t k = 0; k < x.source().grade_count(); ++k)
        sizes.push_back(
            x.source().grade({static_cast<std::uint16_t>(k)}).cell_count());
      out.integers("grades", sizes);
      break;
    }
    case CK::Characters: {
      const auto &x =
          archive_.get<topology::PreparedCyclicCharacterDecomposition>(at);
      out.integers("homology", x.homology_dimensions());
      for (const auto &s : x.sectors()) {
        auto name = "sector." + std::to_string(s.order());
        out.integers(name, s.homology_dimensions());
        out.integers(name + ".weight", std::vector<std::size_t>{s.weight()});
      }
      break;
    }
    case CK::CyclicChain: {
      const auto &x =
          archive_.get<topology::PreparedCyclicEquivariantChain>(at);
      out.integers("order", std::vector<std::uint64_t>{x.order()});
      for (std::size_t k = 0; k < x.voltages().size(); ++k)
        out.integers("voltages." + std::to_string(k), x.voltages()[k]);
      break;
    }
    case CK::Branch: {
      const auto &x = archive_.get<local::PreparedBranchChain>(at);
      out.matrix("basis", x.basis());
      out.matrix("boundary", x.boundary());
      out.vector("polynomial", x.characteristic_polynomial());
      break;
    }
    case CK::PeriodConstruction: {
      const auto &x =
          archive_.get<PreparedIntegralPeriodConstruction>(at).lattice();
      out.matrix("gram", x.gram());
      out.matrix("cup", x.cup_form());
      out.matrix("cycles", x.integral_cycles());
      out.matrix("dual", x.dual_cochains());
      break;
    }
    case CK::PeriodLattice: {
      const auto &x = archive_.get<PreparedIntegralPeriodLattice>(at);
      out.matrix("gram", x.gram());
      out.matrix("cup", x.cup_form());
      out.matrix("cycles", x.integral_cycles());
      out.matrix("dual", x.dual_cochains());
      break;
    }
    case CK::Symplectic:
    case CK::SurfaceSymplectic: {
      const auto &x = archive_.get<PreparedIntegralSymplecticBasis>(at);
      out.matrix("basis", x.basis());
      out.matrix("inverse", x.inverse());
      out.matrix("cycles", x.cycles());
      break;
    }
    case CK::Surface: {
      const auto &x = archive_.get<PreparedSquareTiledSurface>(at);
      out.integers("right", x.right());
      out.integers("up", x.up());
      out.integers("subdivision", std::vector<std::uint64_t>{x.subdivision()});
      break;
    }
    case CK::SurfacePeriods: {
      const auto &x = archive_.get<PreparedSquareTiledPeriods>(at);
      out.matrix("gram", x.construction().lattice().gram());
      out.matrix("cup", x.construction().lattice().cup_form());
      out.matrix("gluing", x.gluing_cycles());
      break;
    }
    case CK::Normalization: {
      const auto &x = archive_.get<PreparedFiniteJacobianNormalization>(at);
      out.matrix("star_square", x.star_square());
      for (std::size_t k = 0; k < x.sectors().size(); ++k) {
        auto n = "sector." + std::to_string(k);
        const auto &s = x.sectors()[k];
        out.vector(n + ".frequency",
                   s.declaration.squared_frequency.coordinates());
        out.vector(n + ".root", s.declaration.positive_root.coordinates());
        out.integers(n + ".dimension", std::vector<std::size_t>{s.dimension});
      }
      break;
    }
    case CK::EllipticFactors: {
      const auto &x =
          archive_.get<PreparedFiniteEllipticFactors>(at).certificate();
      out.scalar("index", CQ{x.index()});
      for (std::size_t k = 0; k < x.factors().size(); ++k) {
        auto n = "factor." + std::to_string(k);
        const auto &f = x.factors()[k];
        out.scalar(n + ".polarization", CQ{f.intersection});
        out.scalar(n + ".frequency", f.squared_frequency);
        out.matrix(n + ".lattice", f.saturated_lattice);
        out.vector(n + ".modulus_real", f.modulus.real.coordinates());
        out.vector(n + ".modulus_imaginary", f.modulus.imaginary.coordinates());
      }
      break;
    }
    case CK::Coordinate: {
      const auto &x = archive_.get<UnivariateCoordinate>(at);
      out.text("name", x.name());
      out.integers("revision", std::vector<std::uint64_t>{x.revision()});
      break;
    }
    case CK::Domain: {
      const auto &x = archive_.get<UnivariateDomain>(at);
      out.scalar("lower", x.lower());
      out.scalar("upper", x.upper());
      out.flag("lower_closed", x.lower_closed());
      out.flag("upper_closed", x.upper_closed());
      for (std::size_t k = 0; k < x.original_guards().size(); ++k)
        out.vector("guard." + std::to_string(k), x.original_guards()[k]);
      break;
    }
    case CK::Expression: {
      const auto &x = archive_.get<OwnedRationalExpression>(at);
      out.vector("numerator", x.original_numerator());
      out.vector("denominator", x.original_denominator());
      break;
    }
    case CK::Carrier: {
      const auto &x = archive_.get<UnivariateCarrier>(at);
      out.integers("dimension", std::vector<std::size_t>{x.dimension()});
      break;
    }
    case CK::CovariantFamily: {
      const auto &x = archive_.get<PreparedCovariantFamily>(at);
      out.flag("metric_compatible", x.metric_compatible());
      for (std::size_t i = 0; i < x.boundaries().size(); ++i)
        out.formal("boundary." + std::to_string(i), x.boundaries()[i]);
      for (std::size_t i = 0; i < x.metrics().size(); ++i)
        out.formal("metric." + std::to_string(i), x.metrics()[i]);
      for (std::size_t i = 0; i < x.domain().original_guards().size(); ++i)
        out.vector("guard." + std::to_string(i),
                   x.domain().original_guards()[i]);
      break;
    }
    case CK::FamilyMap: {
      out.formal("map", archive_.get<CovariantMapFamily>(at).formal());
      break;
    }
    case CK::LayerFamily: {
      const auto &x = archive_.get<PreparedLayerFamily>(at);
      out.formal("offset", x.offset());
      out.formal("kernel", x.kernel());
      out.integers("constraint_rank",
                   std::vector<std::size_t>{x.constraint_rank()});
      break;
    }
    case CK::JetBase:
    case CK::JetReframe: {
      const auto &x = archive_.get<OwnedJetBase>(at);
      out.matrices("metrics", x.metrics());
      out.matrices("boundaries", x.boundaries());
      out.matrices("frames", x.frames());
      for (std::size_t k = 0; k < x.fields().size(); ++k)
        out.vector("field." + std::to_string(k), x.fields()[k]);
      break;
    }
    case CK::Jet:
    case CK::FamilyJet: {
      const auto &x = archive_.get<PreparedCovariantJet>(at);
      out.matrices("hodge", x.hodge());
      out.matrices("hodge_dot", x.hodge_dots());
      out.matrices("metric_residual", x.metric_residuals());
      break;
    }
    case CK::RectangularJet:
    case CK::FamilyMapJet: {
      const auto &x = archive_.get<RectangularCovariantJet>(at);
      out.matrix("value", x.value());
      out.matrix("dot", x.dot());
      out.matrix("covariant", x.covariant());
      break;
    }
    case CK::Response:
    case CK::GreenResponse:
    case CK::FamilyResponse: {
      const auto &x = archive_.get<PreparedResponseJet>(at);
      out.matrix("gauge", x.gauge());
      out.matrix("gauge_dot", x.gauge_dot());
      break;
    }
    case CK::DerivativeField: {
      const auto &x = archive_.get<OwnedDerivativeField>(at);
      out.vector("value", x.value());
      out.vector("dot", x.dot());
      break;
    }
    case CK::ResponseApply: {
      const auto &x = archive_.get<PreparedResponseJet::Result>(at);
      out.vector("value", x.value());
      out.vector("dot", x.dot());
      out.vector("source_driver", x.source_driver());
      out.vector("operator_driver", x.operator_driver());
      out.vector("gauge_driver", x.gauge_driver());
      break;
    }
    case CK::WordApply: {
      const auto &x = archive_.get<PreparedDerivativeWord::Result>(at);
      out.vector("value", x.value());
      out.vector("dot", x.dot());
      for (std::size_t k = 0; k < x.positions().size(); ++k)
        out.vector("position." + std::to_string(k), x.positions()[k]);
      break;
    }
    case CK::CrossPairing: {
      const auto &x = archive_.get<CrossPairingJet>(at);
      out.matrix("form", x.form());
      out.matrix("dot", x.dot());
      break;
    }
    case CK::SurfaceCocycle: {
      out.vector("values", archive_.get<OwnedSurfaceCocycle>(at).values());
      break;
    }
    case CK::SurfaceHolonomy: {
      const auto &x = archive_.get<PreparedSurfaceHolonomy>(at);
      out.vector("periods", x.class_periods());
      out.vector("bezout", x.bezout());
      out.vector("psi", x.psi());
      break;
    }
    case CK::SurfaceInheritance: {
      const auto &x = archive_.get<PreparedSurfaceInheritance>(at);
      out.integers("dimensions", x.dimensions());
      for (std::size_t k = 0; k < 3; ++k)
        out.matrix("sector." + std::to_string(k), x.frames()[k]);
      break;
    }
    case CK::RotatingSystem: {
      const auto &x = archive_.get<OwnedConsistentLinearSystem>(at);
      out.vector("rhs", x.rhs());
      out.vector("solution", x.solution());
      break;
    }
    case CK::RotatingCoefficient: {
      const auto &x = archive_.get<PreparedRotatingCoefficient>(at);
      out.scalar("kappa", x.parameters().kappa);
      out.scalar("q", x.parameters().q);
      break;
    }
    case CK::RefinementLevel:
    case CK::RefinementReframe: {
      const auto &x = archive_.get<OwnedRefinementLevel>(at);
      out.matrices("metrics", x.metrics());
      out.matrices("boundaries", x.boundaries());
      out.matrices("hodge", x.hodge());
      out.matrices("frames", x.frames());
      out.matrices("frame_changes", x.frame_changes());
      break;
    }
    case CK::RefinementMap:
    case CK::RefinementCompose: {
      const auto &x = archive_.get<PreparedRefinementMap>(at);
      out.matrices("maps", x.maps());
      out.matrices("metric_defect", x.metric_defects());
      out.matrices("adjoint_defect", x.adjoint_defects());
      out.matrices("hodge_defect", x.hodge_defects());
      break;
    }
    case CK::RefinementDiagram: {
      const auto &x = archive_.get<PreparedRefinementDiagram>(at);
      out.integers("path_count", std::vector<std::size_t>{x.paths().size()});
      for (std::size_t k = 0; k < x.paths().size(); ++k) {
        out.integers("path." + std::to_string(k), x.paths()[k].maps);
        out.matrices("composite." + std::to_string(k), x.paths()[k].composite);
      }
      break;
    }
    case CK::RefinementField: {
      out.vector("value", archive_.get<RefinementField>(at).values());
      break;
    }
    case CK::RefinementEnergy: {
      const auto &x = archive_.get<PreparedRefinementMap::EnergyReceipt>(at);
      out.scalar("total", x.total());
      break;
    }
    case CK::RefinementSplit: {
      const auto &x = archive_.get<PreparedRefinementRetraction::Split>(at);
      out.vector("coarse", x.coarse().values());
      out.vector("represented", x.represented().values());
      out.vector("private", x.private_field().values());
      out.matrix("gram", x.gram());
      break;
    }
    case CK::RefinementResponse: {
      out.matrix("defect",
                 archive_.get<PreparedRefinementResponse>(at).defect());
      break;
    }
    case CK::RefinementResponseApply: {
      const auto &x =
          archive_.get<PreparedRefinementResponse::Result>(at).data();
      out.vector("value", x.aggregate);
      out.vector("operator", x.operator_term);
      out.vector("innovation", x.innovation_term);
      break;
    }
    case CK::RefinementAccumulate: {
      const auto &x = archive_.get<PreparedRefinementDiagram::Accumulation>(at);
      out.matrix("gram", x.gram());
      out.vector("terminal", x.terminal().values());
      break;
    }
    case CK::LimitingContract: {
      const auto &x =
          archive_.get<PreparedRefinementDiagram::LimitContract>(at);
      out.text("name", x.name());
      out.text("theorem", x.theorem());
      out.flag("continuum_certified", x.certifies_continuum_limit());
      for (std::size_t k = 0; k < x.unresolved_hypotheses().size(); ++k)
        out.text("unresolved." + std::to_string(k),
                 x.unresolved_hypotheses()[k]);
      break;
    }
    case CK::NativeField: {
      const auto &x = archive_.get<NativeFieldValue>(at);
      out.vector("value", x.sparse_values().dense());
      break;
    }
    case CK::Krylov: {
      out.vector("polynomial",
                 archive_.get<PreparedNativeKrylov>(at).polynomial());
      break;
    }
    case CK::CayleyOrbit: {
      const auto &x = archive_.get<PreparedNativeCayleyOrbit>(at);
      out.integers("kind", std::vector<std::uint64_t>{
                               static_cast<std::uint64_t>(x.kind())});
      out.vector("polynomial", x.krylov().polynomial());
      out.vector("unresolved", x.unresolved_polynomial());
      if (x.period())
        out.integers("period", std::vector<std::size_t>{*x.period()});
      break;
    }
    default:
      out.text("inspection_profile", "declaration-only");
      break;
    }
    return out.take();
  }
  CInspection evaluate(std::size_t at, const std::string &operation,
                       const std::string &parameter, std::size_t grade,
                       std::size_t power, std::size_t maximum) const {
    using namespace rfc;
    using namespace temporal;
    using namespace geometry;
    Inspect out("native-C-observation-v1", maximum);
    CG g(exact_, [this] { cancel_->check(); });
    cancel_->check();
    auto q = c_rational(parameter);
    g.value(q);
    if (operation != "moment" && (grade || power))
      throw std::invalid_argument(
          "unused grade/power outside the moment profile");
    if ((operation == "moment" || operation == "half_turn") && q != 0)
      throw std::invalid_argument(
          "this observation has no affine parameter argument");
    if (operation == "domain_contains") {
      const auto &d = archive_.get<UnivariateDomain>(at);
      out.flag("contains", d.contains(d.at(q, g), g));
    } else if (operation == "layer_at") {
      const auto &layer = archive_.get<PreparedLayerFamily>(at);
      const auto &row = archive_.declarations().at(at);
      const auto map = row.dependencies.at(0);
      const auto family = archive_.declarations().at(map).dependencies.at(0);
      const auto &d = archive_.get<PreparedCovariantFamily>(family).domain();
      auto point = OwnedJetPoint::declare(d, d.at(q, g), g);
      const auto jet = layer.jet(point, g);
      out.vector("offset", jet.offset());
      out.vector("offset_dot", jet.offset_dot());
      out.matrix("kernel", jet.kernel());
      out.matrix("kernel_dot", jet.kernel_dot());
      out.integers("constraint_rank",
                   std::vector<std::size_t>{jet.constraint_rank()});
    } else if (operation == "rank") {
      const auto &x = archive_.get<PreparedUnivariateFamily>(at);
      out.integers("rank",
                   std::vector<std::size_t>{x.rank_at(x.domain().at(q, g), g)});
    } else if (operation == "expression") {
      const auto &x = archive_.get<OwnedRationalExpression>(at);
      out.scalar("value", x.evaluate(x.domain().at(q, g), g));
    } else if (operation == "moment") {
      const auto &x = archive_.get<PreparedCovariantJet>(at);
      auto v = x.moment(grade, power, g);
      out.scalar("value", v.value);
      out.scalar("derivative", v.derivative);
      out.vector("positions", v.operator_positions);
    } else if (operation == "rotation" || operation == "half_turn") {
      const auto &x = archive_.get<PreparedRotatingCoefficient>(at);
      auto s = operation == "half_turn" ? x.half_turn(g)
                                        : x.sample(x.domain().at(q, g), g);
      out.scalar("obstruction", s.obstruction());
      out.vector("solution_real", s.solution_real());
      out.vector("solution_imaginary", s.solution_imaginary());
    } else
      throw std::invalid_argument("unsupported C observation profile");
    return out.take();
  }
};
} // namespace rexgraph::python
