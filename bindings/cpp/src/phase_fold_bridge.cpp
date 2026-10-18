#include "phase_fold_bridge.hpp"
#include "algebraic_bridge.hpp"
#include <rfc/storage/phase_fold_history.hpp>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace rexgraph::python {
using namespace rfc;
namespace {
storage::PhaseFoldLimits policy(const std::vector<std::size_t>& v) {
    if (v.size()!=35) throw std::invalid_argument("phase/fold archive requires 35 admission limits");
    storage::PhaseFoldLimits p;
    p.wire.max_file_bytes=v[0];p.wire.max_records=v[1];p.wire.max_rational_coordinates=v[2];p.wire.max_integer_bytes=v[3];
    p.wire.field={v[4],v[5],v[6],v[7],v[8]};p.wire.modules={v[9],v[10],v[11],v[12]};
    p.wire.fibers={v[13],v[14]};p.wire.real_root={v[15]};p.wire.complex_root={v[16],v[17],v[18]};p.wire.metric={v[19],v[20]};
    p.history={v[21],v[22],v[23],v[24],v[25],v[26]};p.cover={v[27],v[28],v[29]};p.panels={v[30],v[31],v[32],v[33]};p.max_proposals=v[34];
    return p;
}
std::string bytes(const std::vector<std::byte>& v) {return {reinterpret_cast<const char*>(v.data()),v.size()};}
std::vector<std::string> coordinates(const numeric::RationalPhase& v) {return {v.real().get_str(16),v.imaginary().get_str(16)};}
std::vector<std::string> coordinates(const numeric::AlgebraicUnitPhase& v) {
    std::vector<std::string> out;for (const auto& q:v.value().coordinates()) out.push_back(q.get_str(16));return out;
}
numeric::RationalVector exact(const std::vector<std::string>& v) {
    numeric::RationalVector out;
    for (const auto& s:v) {numeric::Rational q;if (q.set_str(s,16)!=0 || q.get_den()==0) throw std::invalid_argument("invalid phase coordinate");q.canonicalize();out.push_back(q);}
    return out;
}
}
struct PhaseFoldArchive::State {
    storage::PhaseFoldArchive archive;
    storage::PhaseFoldLimits limits;
    std::string image, source;
    std::shared_ptr<Source> history_owner;
    State(storage::PhaseFoldArchive a,storage::PhaseFoldLimits p):archive(std::move(a)),limits(p),
        image(bytes(storage::encode_phase_fold_archive(archive,p))),source(bytes(storage::encode_rfcx(archive.source()).bytes)) {}
};
PhaseFoldArchive::PhaseFoldArchive(std::shared_ptr<const State> p):state_(std::move(p)) {}
std::vector<std::size_t> PhaseFoldArchive::default_limits() {
    auto v=AlgebraicArchive::default_limits();const storage::PhaseFoldLimits p;
    const auto h=p.history;const auto c=p.cover;const auto a=p.panels;
    v.insert(v.end(),{h.max_steps,h.max_coordinate_slots,h.max_integer_bits,h.max_source_cells,h.max_source_occurrences,h.max_interaction_terms,
                     c.max_vertices,c.max_path_steps,c.max_integer_bits,a.max_cells,a.max_occurrences,a.max_creases,a.max_integer_bits,p.max_proposals});return v;
}
std::shared_ptr<PhaseFoldArchive> PhaseFoldArchive::load(const char* data,std::size_t size,const std::vector<std::size_t>& v) {
    if (!data) throw std::invalid_argument("phase/fold archive requires a byte buffer");
    const auto p=policy(v);auto a=storage::decode_phase_fold_archive({reinterpret_cast<const std::byte*>(data),size},p);
    return std::shared_ptr<PhaseFoldArchive>(new PhaseFoldArchive(std::make_shared<const State>(std::move(a),p)));
}
std::shared_ptr<PhaseFoldArchive> PhaseFoldArchive::from_source(const std::shared_ptr<Source>& source,
    const std::vector<std::uint64_t>& channel,const std::vector<std::size_t>& v) {
    if (!source || channel.size()!=3) throw std::invalid_argument("phase/fold history requires an endpoint and three channel names");
    const auto p=policy(v);auto a=storage::read_phase_fold_archive(*source->endpoint(),{channel[0],channel[1],channel[2]},p);
    auto state=std::make_shared<State>(std::move(a),p);state->history_owner=source;
    return std::shared_ptr<PhaseFoldArchive>(new PhaseFoldArchive(std::move(state)));
}
std::string PhaseFoldArchive::image() const {return state_->image;}
std::string PhaseFoldArchive::source_image() const {return state_->source;}
PhaseFoldInfo PhaseFoldArchive::info(std::size_t maximum, bool proposals) const {
    PhaseFoldInfo out;const auto& a=state_->archive;
    out.kind=static_cast<unsigned>(a.word().index());out.operations=a.operation_count();out.proposals=a.proposals().size();
    for (std::size_t k=0;k<a.source().grade_count();++k) out.source_sizes.push_back(a.source().grade(core::Grade{static_cast<std::uint16_t>(k)}).cell_count());
    auto retain_phase=[&](const auto& phase) {
        using T=std::decay_t<decltype(phase)>;
        if constexpr (std::same_as<T,numeric::AlgebraicUnitPhase>) {
            const auto& involution=phase.involution();out.degree=involution.field().degree();
            for (const auto& q:involution.field().minimal_polynomial()) out.polynomial.push_back(q.get_str(16));
            for (const auto& q:involution.generator_image().coordinates()) out.involution.push_back(q.get_str(16));
        } else out.degree=1;
        const auto c=coordinates(phase);out.endpoint_coordinates=c.size();out.shape={c.size()};
        out.endpoint.assign(c.begin(),c.begin()+static_cast<std::ptrdiff_t>(std::min(maximum,c.size())));
    };
    std::visit([&](const auto& h) {
        using T=std::decay_t<decltype(h)>;
        if constexpr (std::same_as<T,temporal::RationalFoldHistory>) {
            out.degree=1;out.shape={h.poses().size(),4,4};out.endpoint_coordinates=machine::checked_size_mul(h.poses().size(),16);
            if (maximum) for (const auto& pose:h.poses()) {
                const auto matrix=pose.affine();
                for (std::size_t i=0;i<4 && out.endpoint.size()<maximum;++i)
                    for (std::size_t j=0;j<4 && out.endpoint.size()<maximum;++j) out.endpoint.push_back(matrix.get(i,j).get_str(16));
                if (out.endpoint.size()>=maximum) break;
            }
            out.observation_coordinates=out.endpoint_coordinates;
            for (const auto& op:h.operations()) out.observation_coordinates=machine::checked_size_add(out.observation_coordinates,machine::checked_size_add(2,op.moving_panels.size()));
            for (std::size_t i=0;i<std::min(maximum,h.operations().size());++i) {
                const auto& op=h.operations()[i];out.operation_views.push_back({op.identity,op.active,false,op.crease,0,op.moving_panels,coordinates(op.phase)});
            }
        } else if constexpr (std::same_as<T,storage::RationalPhaseWalk> || std::same_as<T,storage::AlgebraicPhaseWalk>) {
            retain_phase(h.endpoint());out.branch=h.lift().cover().branch_vertex();
            out.observation_coordinates=machine::checked_size_add(out.endpoint_coordinates,h.lift().positions().size());
            for (std::size_t i=0;i<std::min(maximum,h.lift().positions().size());++i) out.positions.push_back(h.lift().positions()[i].get_str(16));
            for (std::size_t i=0;i<std::min(maximum,h.lift().path().steps().size());++i) {
                const auto& op=h.lift().path().steps()[i];out.operation_views.push_back({0,true,op.forward,0,op.relation,{}, {}});
            }
        } else {
            retain_phase(h.endpoint());out.has_cell=true;out.grade=h.cell().grade.value;out.index=h.cell().index;
            out.observation_coordinates=machine::checked_size_add(out.endpoint_coordinates,machine::checked_size_mul(h.operations().size(),out.endpoint_coordinates));
            for (std::size_t i=0;i<std::min(maximum,h.operations().size());++i) {
                const auto& op=h.operations()[i];out.operation_views.push_back({op.identity,op.active,false,0,0,{},coordinates(op.phase)});
            }
        }
    },a.word());
    out.observation_coordinates=machine::checked_size_add(out.observation_coordinates,
        machine::checked_size_add(out.polynomial.size(),out.involution.size()));
    for (std::size_t i=0;proposals && i<std::min(maximum,a.proposals().size());++i) {
        const auto& p=a.proposals()[i];out.proposal_views.push_back({p.key,p.position,p.active,std::visit([](const auto& phase){return coordinates(phase);},p.replacement)});
    }
    return out;
}
std::shared_ptr<PhaseFoldArchive> PhaseFoldArchive::prefix(std::size_t count) const {
    return std::shared_ptr<PhaseFoldArchive>(new PhaseFoldArchive(std::make_shared<const State>(state_->archive.observed_prefix(count),state_->limits)));
}
std::shared_ptr<PhaseFoldArchive> PhaseFoldArchive::propose(std::uint64_t key,std::size_t position,const std::vector<std::string>& values,bool active) const {
    const auto q=exact(values);const auto& word=state_->archive.word();
    auto archive=[&]() {
        if (const auto* h=std::get_if<temporal::AlgebraicPhaseHistory>(&word)) {
            const auto& involution=h->phase_at(0).involution();
            if (q.size()!=involution.field().degree()) throw std::invalid_argument("phase requires complete polynomial coordinates");
            return state_->archive.propose(key,position,numeric::AlgebraicUnitPhase(involution,involution.field().element(q)),active);
        }
        if (q.size()!=2) throw std::invalid_argument("rational phase requires exact real and imaginary coordinates");
        return state_->archive.propose(key,position,numeric::RationalPhase(q[0],q[1]),active);
    }();
    return std::shared_ptr<PhaseFoldArchive>(new PhaseFoldArchive(std::make_shared<const State>(std::move(archive),state_->limits)));
}
std::shared_ptr<PhaseFoldArchive> PhaseFoldArchive::apply(std::uint64_t key) const {
    return std::shared_ptr<PhaseFoldArchive>(new PhaseFoldArchive(std::make_shared<const State>(state_->archive.apply(key),state_->limits)));
}
bool PhaseFoldArchive::same_carrier(const std::shared_ptr<PhaseFoldArchive>& other) const {
    if (!other || state_->archive.word().index()!=other->state_->archive.word().index()) return false;
    return std::visit([&](const auto& h) {
        using T=std::decay_t<decltype(h)>;const auto& b=std::get<T>(other->state_->archive.word());
        if constexpr (std::same_as<T,storage::RationalPhaseWalk> || std::same_as<T,storage::AlgebraicPhaseWalk>) return h.lift().cover().same_owner(b.lift().cover());
        else return h.same_carrier(b);
    },state_->archive.word());
}
}
