#include <random>
#include <cmath>
#include <sstream>

#include "libslic3r/Algorithm/LineSplit.hpp"
#include "libslic3r/Arachne/utils/ExtrusionJunction.hpp"
#include "libslic3r/Arachne/utils/ExtrusionLine.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/PerimeterGenerator.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/PrintConfig.hpp"

#include "FuzzySkin.hpp"

#include "libnoise/noise.h"

// #define DEBUG_FUZZY

using namespace Slic3r;

namespace Slic3r::Feature::FuzzySkin {

// Produces a random value between 0 and 1. Thread-safe.
static double random_value() {
    thread_local std::random_device rd;
    // Hash thread ID for random number seed if no hardware rng seed is available
    thread_local std::mt19937 gen(rd.entropy() > 0 ? rd() : std::hash<std::thread::id>()(std::this_thread::get_id()));
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

class UniformNoise: public noise::module::Module {
    public:
        UniformNoise(): Module (GetSourceModuleCount ()) {};

        virtual int GetSourceModuleCount() const { return 0; }
        virtual double GetValue(double x, double y, double z) const { return random_value() * 2 - 1; }
};

// ---------------------------------------------------------------------------
// RippleNoise — deterministic sine-wave displacement along path arc length.
//
// Unlike all other noise types, the ripple pattern is NOT driven by world-space
// (x,y,z) coordinates.  Instead it is driven by the cumulative arc-length
// travelled along the print path, which gives a uniform wave period regardless
// of the geometry.  Because the libnoise Module interface only exposes a
// stateless GetValue(x,y,z) call we cannot use it for ripple.  Instead,
// fuzzy_polyline() and fuzzy_extrusion_line() each have an early-exit branch
// that handles NoiseType::Ripple directly before the generic noise path.
//
// The phase offset is computed from the layer index as follows:
//
//   period_index   = floor(layer_id / layers_between_ripple_offset)
//   phase_offset   = period_index * ripple_offset   [mm along path]
//
// This means layers 0 … period-1 are identical, layers period … 2*period-1
// are shifted by one full ripple_offset, and so on.  Setting
// layers_between_ripple_offset = 1 shifts on every layer; setting it to N makes N
// consecutive layers share the same pattern before jumping.
// ---------------------------------------------------------------------------
static double compute_ripple_spacing(const FuzzySkinConfig& cfg, double perimeter_length_mm)
{
    if (cfg.ripples_per_layer <= 0)
        return 0.0;
    return perimeter_length_mm / (cfg.ripples_per_layer * 4.0);
}

static double ripple_phase_offset_mm(const FuzzySkinConfig& cfg, double fitted_spacing)
{
    if (cfg.ripple_offset == 0.0 || cfg.layers_between_ripple_offset <= 0)
        return 0.0;
    if (fitted_spacing <= 0.0)
        return 0.0;

    const int    effective_layer = std::max(cfg.layer_id, 0);
    const int    period_index    = effective_layer / std::max(cfg.layers_between_ripple_offset, 1);
    const double full_period_mm  = fitted_spacing * 2.0;
    const double raw_offset      = period_index * cfg.ripple_offset * full_period_mm;
    return fmod(raw_offset, full_period_mm);
}

/// Sine-wave displacement value in [-1, 1] for a given arc-length position.
/// point_distance is used as the half-period (full wavelength = 2 * point_distance).
static double ripple_value(double arc_length_mm, double half_period_mm, double phase_offset_mm)
{
    if (half_period_mm <= 0.0)
        return 0.0;
    return std::sin((arc_length_mm + phase_offset_mm) * M_PI / half_period_mm);
}

static std::unique_ptr<noise::module::Module> get_noise_module(const FuzzySkinConfig& cfg)
{
    if (cfg.noise_type == NoiseType::Perlin) {
        auto perlin_noise = noise::module::Perlin();
        perlin_noise.SetFrequency(1 / cfg.noise_scale);
        perlin_noise.SetOctaveCount(cfg.noise_octaves);
        perlin_noise.SetPersistence(cfg.noise_persistence);
        return std::make_unique<noise::module::Perlin>(perlin_noise);
    } else if (cfg.noise_type == NoiseType::Billow) {
        auto billow_noise = noise::module::Billow();
        billow_noise.SetFrequency(1 / cfg.noise_scale);
        billow_noise.SetOctaveCount(cfg.noise_octaves);
        billow_noise.SetPersistence(cfg.noise_persistence);
        return std::make_unique<noise::module::Billow>(billow_noise);
    } else if (cfg.noise_type == NoiseType::RidgedMulti) {
        auto ridged_multi_noise = noise::module::RidgedMulti();
        ridged_multi_noise.SetFrequency(1 / cfg.noise_scale);
        ridged_multi_noise.SetOctaveCount(cfg.noise_octaves);
        return std::make_unique<noise::module::RidgedMulti>(ridged_multi_noise);
    } else if (cfg.noise_type == NoiseType::Voronoi) {
        auto voronoi_noise = noise::module::Voronoi();
        voronoi_noise.SetFrequency(1 / cfg.noise_scale);
        voronoi_noise.SetDisplacement(1.0);
        return std::make_unique<noise::module::Voronoi>(voronoi_noise);
    } else {
        // Classic (uniform random) and Ripple both fall here for the module
        // placeholder.  Ripple never actually calls GetValue on this module —
        // it is handled in the dedicated ripple branch before GetValue is used.
        return std::make_unique<UniformNoise>();
    }
}

#define M_2PI (M_PI * 2.) // can move it to a common functions section


// Thanks Cura developers for this function.
void fuzzy_polyline(Points& poly, bool closed, coordf_t slice_z, const FuzzySkinConfig& cfg)
{
    if (poly.size() < 2)
        return;

    if (cfg.noise_type == NoiseType::Ripple) {
        if (poly.size() < 3)
            return;

        const double amplitude = unscale_(cfg.thickness);
        const double N         = static_cast<double>(cfg.ripples_per_layer);
        if (N <= 0.0)
            return;

        const double fill_step_mm = unscale_(cfg.point_distance);
        if (fill_step_mm < 1e-6)
            return;

        // --- Per-layer-group phase shift ---
        const int    effective_layer = std::max(cfg.layer_id, 0);
        const int    period_index    = effective_layer / std::max(cfg.layers_between_ripple_offset, 1);
        const double phase_shift_rad = (cfg.ripple_offset == 0.0 || cfg.layers_between_ripple_offset <= 0) ?
                                           0.0 :
                                           fmod(period_index * cfg.ripple_offset * M_2PI, M_2PI);

        // --- Find anchor: leftmost Y=0 crossing, fallback to closest-|y| vertex ---
        Vec2d        anchor_world(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        bool         found_crossing = false;
        const size_t np             = poly.size();
        for (size_t i = 0; i < np; ++i) {
            const Point& pa = poly[i];
            const Point& pb = poly[(i + 1) % np];
            const double ya = unscale_(pa.y());
            const double yb = unscale_(pb.y());
            if ((ya <= 0.0 && yb >= 0.0) || (ya >= 0.0 && yb <= 0.0)) {
                double t             = (std::abs(yb - ya) < 1e-9) ? 0.0 : ya / (ya - yb);
                t                    = std::max(0.0, std::min(1.0, t));
                const double x_cross = unscale_(pa.x()) + t * (unscale_(pb.x()) - unscale_(pa.x()));
                if (!found_crossing || x_cross < anchor_world.x()) {
                    anchor_world   = Vec2d(x_cross, 0.0);
                    found_crossing = true;
                }
            }
        }
        if (!found_crossing) {
            double best_abs_y = std::numeric_limits<double>::max();
            for (const Point& p : poly) {
                const double ay = std::abs(unscale_(p.y()));
                if (ay < best_abs_y) {
                    best_abs_y   = ay;
                    anchor_world = Vec2d(unscale_(p.x()), unscale_(p.y()));
                }
            }
        }

        // --- Compute total perimeter length (mm) ---
        double perimeter_mm = 0.0;
        for (size_t i = 0; i < np; ++i)
            perimeter_mm += (poly[(i + 1) % np] - poly[i]).cast<double>().norm();
        perimeter_mm = unscale_(perimeter_mm); // convert from scaled units to mm

        if (perimeter_mm < 1e-6)
            return;

        // --- Find arc-length of anchor along the perimeter ---
        // Walk edges until we find the one containing anchor_world, accumulate length.
        // We re-walk the edges looking for the closest point on the polyline to anchor_world.
        double anchor_arc_mm = 0.0;
        double best_dist_sq  = std::numeric_limits<double>::max();
        double accum_mm      = 0.0;
        for (size_t i = 0; i < np; ++i) {
            const Vec2d  pa_mm(unscale_(poly[i].x()), unscale_(poly[i].y()));
            const Vec2d  pb_mm(unscale_(poly[(i + 1) % np].x()), unscale_(poly[(i + 1) % np].y()));
            const Vec2d  seg     = pb_mm - pa_mm;
            const double seg_len = seg.norm();
            if (seg_len < 1e-9) {
                accum_mm += seg_len;
                continue;
            }

            // Project anchor onto this segment, clamp to [0,1]
            const double t       = std::max(0.0, std::min(1.0, (anchor_world - pa_mm).dot(seg) / (seg_len * seg_len)));
            const Vec2d  closest = pa_mm + seg * t;
            const double dist_sq = (closest - anchor_world).squaredNorm();
            if (dist_sq < best_dist_sq) {
                best_dist_sq  = dist_sq;
                anchor_arc_mm = accum_mm + t * seg_len;
            }
            accum_mm += seg_len;
        }

        // --- Walk perimeter, emit displaced points ---
        // Phase at arc-length s:  φ(s) = N * 2π * (s - anchor_arc_mm) / perimeter_mm + phase_shift_rad
        // sin(φ) = 1 at anchor (s = anchor_arc_mm, shift=0) → peak at anchor every layer.
        // We want peak at anchor → need sin(φ) = 1 → φ = π/2 at anchor.
        // φ(anchor) = 0 + phase_shift = phase_shift.  Add π/2 to make it a peak:
        const double phase_at_zero = M_PI_2 + phase_shift_rad;

        auto arc_phase = [&](double arc_mm) -> double {
            const double rel = arc_mm - anchor_arc_mm;
            return N * M_2PI * rel / perimeter_mm + phase_at_zero;
        };

        Points out;
        out.reserve(static_cast<size_t>(perimeter_mm / fill_step_mm) + np * 2);

        accum_mm = 0.0;
        for (size_t i = 0; i < np; ++i) {
            const Point& p0      = poly[i];
            const Point& p1      = poly[(i + 1) % np];
            const Vec2d  seg     = (p1 - p0).cast<double>();
            const double seg_len = seg.norm();
            if (seg_len < EPSILON)
                continue;

            const double seg_len_mm = unscale_(seg_len);
            const Vec2d  seg_unit   = seg / seg_len;
            const Vec2d  seg_perp   = perp(seg_unit); // points "outward" (left of travel)

            // How far into this segment does the current fill_step position land?
            // We sample at every fill_step_mm along the full perimeter.
            // Find first sample point >= accum_mm within this segment.
            const double seg_start_mm = accum_mm;
            const double seg_end_mm   = accum_mm + seg_len_mm;

            // First sample index whose arc position falls inside this segment
            const double first_s = std::ceil(seg_start_mm / fill_step_mm) * fill_step_mm;

            for (double s = first_s; s < seg_end_mm; s += fill_step_mm) {
                const double t    = (s - seg_start_mm) / seg_len_mm;
                const double phi  = arc_phase(s);
                const double disp = std::sin(phi) * amplitude; // mm

                // Position along segment in scaled coords
                const Point pt_sc = p0 + (seg * t).cast<coord_t>();
                out.emplace_back(pt_sc + (seg_perp * scale_(disp)).cast<coord_t>());
            }

            accum_mm = seg_end_mm;
        }

        while (out.size() < 3) {
            out.emplace_back(poly[poly.size() - 2]);
        }

#ifdef DEBUG_FUZZY
        {
            const double       disp_at_anchor = std::sin(arc_phase(anchor_arc_mm));
            std::ostringstream dbg;
            dbg << std::fixed << std::setprecision(4);
            dbg << "[Ripple-ArcLen] layer=" << cfg.layer_id << "  pts=" << poly.size() << "  z=" << slice_z << "\n";
            dbg << "  anchor_world=(" << anchor_world.x() << ", " << anchor_world.y() << ")"
                << "  anchor_arc_mm=" << anchor_arc_mm << "  perimeter_mm=" << perimeter_mm << "\n";
            dbg << "  N=" << N << "  phase_at_zero=" << phase_at_zero << "  phase_shift_rad=" << phase_shift_rad
                << "  period_index=" << period_index << "\n";
            dbg << "  disp_at_anchor=" << disp_at_anchor << "  (should be 1.0 when shift=0)\n";
            dbg << "  out_pts=" << out.size() << "\n";
            OutputDebugStringA(dbg.str().c_str());
        }
#endif

        if (out.size() >= 3)
            poly = std::move(out);
        return;
    }
    // Generic noise branch unchanged
    std::unique_ptr<noise::module::Module> noise                   = get_noise_module(cfg);
    const double                           min_dist_between_points = cfg.point_distance * 3. / 4.;
    const double                           range_random_point_dist = cfg.point_distance / 2.;
    double                                 dist_left_over          = random_value() * (min_dist_between_points / 2.);
    Point*                                 p0                      = &poly.back();
    Points                                 out;
    out.reserve(poly.size());
    for (Point& p1 : poly) {
        if (!closed) {
            closed = true;
            p0     = &p1;
            continue;
        }
        Vec2d  p0p1      = (p1 - *p0).cast<double>();
        double p0p1_size = p0p1.norm();
        double p0pa_dist = dist_left_over;
        for (; p0pa_dist < p0p1_size; p0pa_dist += min_dist_between_points + random_value() * range_random_point_dist) {
            Point  pa = *p0 + (p0p1 * (p0pa_dist / p0p1_size)).cast<coord_t>();
            double r  = noise->GetValue(unscale_(pa.x()), unscale_(pa.y()), slice_z) * cfg.thickness;
            out.emplace_back(pa + (perp(p0p1).cast<double>().normalized() * r).cast<coord_t>());
        }
        dist_left_over = p0pa_dist - p0p1_size;
        p0             = &p1;
    }
    while (out.size() < 3) {
        size_t point_idx = poly.size() - 2;
        out.emplace_back(poly[point_idx]);
        if (point_idx == 0)
            break;
        -- point_idx;
    }
    if (out.size() >= 3)
        poly = std::move(out);
}
// internal fuzzy skin parameters
struct FuzzySkinParams
{
    Arachne::ExtrusionJunctions& out;
    FuzzySkinMode                current_mode;
    const FuzzySkinConfig&       cfg;
    double                       z;
    bool                         one_way_expansion; // one_way_expansionmetric skin (true = Displacement+, Combined)
    noise::module::Module&       noise;
    coord_t                      params[3] = {0, 0, 0};
    bool                         draw_fill = true;
    bool                         draw_line = true;
};

auto constrainPI = [](double x) { // can move it to a common functions section
    while (x < 0.)
        x += M_2PI;
    return fmod(x + M_PI, M_2PI) - M_PI;
};

template<typename T> double get_noise_value(noise::module::Module& noise, T& point, double z, bool one_way_expansion)
{
    return one_way_expansion ? noise.GetValue(unscale_(point.x()), unscale_(point.y()), z) * .5 + .5 :
                               noise.GetValue(unscale_(point.x()), unscale_(point.y()), z);
};

// Defining the function for drawing a new fuzzy line
// vector - a variable that determines the direction of the passing line.
//          The fuzzy range is set in the perpendicular direction as usual.
//          The length of the vector must be normalized to 1. If the value is different, the offset coefficient is set.
// j      - Extrusion Junction point with coordinates, thickness and index. The index is a serial number of the wall.
//          The index changes for complex lines or void fillers, if they are generated (< 0 or > 99999).
//          Complex lines will be printed with the inner walls parameters before printing the outer wall.
// p      - additional parameters of Fuzzy Skin

void out_point(Vec2d vector, Arachne::ExtrusionJunction* j, FuzzySkinParams& p)
{
    const Vec2d   _perp(perp(vector));
    const coord_t _semithick = j->w / 2;
    double        dist       = get_noise_value(p.noise, j->p, p.z, p.one_way_expansion);
    coord_t       _counter;
    switch (p.current_mode) {
    case FuzzySkinMode::Displacement: // classical algorithm, no any changed, two-way expansion
        p.out.emplace_back(j->p + (_perp * dist * p.cfg.thickness).cast<coord_t>(), j->w, j->perimeter_index);
        break;
    case FuzzySkinMode::Extrusion: p.out.emplace_back(j->p, dist * (vector.norm() * p.cfg.thickness + j->w), j->perimeter_index); break;
    case FuzzySkinMode::Combined:
        dist *= p.cfg.thickness + j->w;
        p.out.emplace_back(j->p + (_perp * (dist / 2 - _semithick)).cast<coord_t>(), dist, j->perimeter_index);
        break;
    }
};

// The procedure for filtering lines through a shaded polygon
Arachne::ExtrusionJunctions fuzzy_filter_by_polygon(Arachne::ExtrusionJunctions out)
{
    Arachne::ExtrusionJunctions out2;
    coord_t                     w(out.front().w);
    coord_t                     pi(out.front().perimeter_index);
    Polyline                    _pl;

    for (auto& point : out) // convert topolyline
        _pl.points.emplace_back(point.p);
    _pl.simplify(scaled(EPSILON));

    ExPolygons _expgs = union_ex(to_polygons({_pl}));
    ExPolygon  _expg;
    for (auto& expg : _expgs)
        if (_expg.contour.length() < expg.contour.length())
            _expg = std::move(expg);

    for (auto& point : _expg.contour)
        out2.emplace_back(Arachne::ExtrusionJunction(point, w, pi));

    if (out2.size() > 2) {
        out2.push_back(out2.front());
        std::reverse(out2.begin(), out2.end());
    }

    return out2;
};

// ---------------------------------------------------------------------------
// fuzzy_extrusion_line — Arachne path variant
// The ripple branch mirrors the logic in fuzzy_polyline but operates on
// ExtrusionJunction vectors so that line width (j->w) is preserved.
// ---------------------------------------------------------------------------

// Thanks Cura developers for this function.
void fuzzy_extrusion_line(Arachne::ExtrusionJunctions& ext_lines, coordf_t slice_z, const FuzzySkinConfig& cfg, bool closed)
{
    size_t  c_size   = ext_lines.size();
    coord_t _overall = 0;

    bool _is_loop = ext_lines.front().p.distance_to(ext_lines.back().p) < cfg.wall_width; // determinate a closed line

    if (_is_loop) { // filter and close loop if necessary
        // Randomize start point for minimise some undesired occurs — skipped for
        // ripple so the deterministic phase is always anchored at the same seam.
        if (cfg.noise_type != NoiseType::Ripple) {
            ext_lines.pop_back();
            std::rotate(ext_lines.begin(), ext_lines.begin() + random_value() * (c_size - 1), ext_lines.end());
            ext_lines.push_back(ext_lines.front());
        }
    }

    for (int _it = 1; _it < (int) c_size; _it++) // calculate polyline length
        _overall += (ext_lines[_it].p - ext_lines[_it - 1].p).norm();

    if (c_size < (_is_loop ? 3 : 2) || _overall < cfg.point_distance || // return if the line too short or has undesired index
        (ext_lines[0].perimeter_index > 0 &&
         (cfg.mode == FuzzySkinMode::Combined)))
        return;
    // -----------------------------------------------------------------------
    // Generic noise branch
    // -----------------------------------------------------------------------
    Arachne::ExtrusionJunctions out;
    size_t                      _cap = _overall * 3 / cfg.point_distance + c_size * 8;
    out.reserve(_cap);

    const bool   _is_random = cfg.noise_type == NoiseType::Classic;
    const double _distance  = cfg.point_distance;
    const bool   _one_way   = cfg.mode == FuzzySkinMode::Combined;
    std::unique_ptr<noise::module::Module> noise = get_noise_module(cfg);
    FuzzySkinParams                        _params{out, cfg.mode, cfg, slice_z, _one_way, *noise};

    const double min_dist_between_points = _is_random ?
                                               _distance * 3. / 4. :
                                               _distance; // hardcoded: the point distance may vary between 3/4 and 5/4 the supplied value
    const double range_random_point_dist =
        _is_random ? _distance / 2. : 0.; // _is_random ? ... - disabled for the test function of precise corners when using 3D noise
    double dist_left_over = random_value() *
                            (min_dist_between_points / 2.); // the distance to be traversed on the line before making the first new point
    coord_t minimal_line = cfg.minimal_line; // Minimal linewidth for this condition for height and spacing: parsms * float(1. - 0.25 * PI)
    _overall = std::rand() * _distance; // Use this var for length calc. Determinate first random point for Fur and Displacement+ infill

    // Get angles map for negative values for loops filter and positive ones for corner former
    std::vector<double> corners(c_size, 0.);
    double              _last_angle = 0;
    for (int _i = 0; _i < (int) c_size - 1; _i++) {
        Point  _p     = (ext_lines[_i + 1].p - ext_lines[_i].p);
        double _angle = atan2(_p.y(), _p.x());
        corners[_i]   = constrainPI(_angle - _last_angle);
        _last_angle   = _angle;
    }

    if (_is_loop) {
        corners[0] = corners[c_size - 1] = constrainPI(corners[0] - _last_angle);
    } else {
        corners[0] = corners[c_size - 1] = 0.;
    }

    switch (_params.current_mode) {
    case FuzzySkinMode::Displacement: // classical algorithm, no any changed, two-way expansion
        _params.one_way_expansion = false;
        break;
    }

    // Fuzzy line segment former
    Arachne::ExtrusionJunction* p0 = &ext_lines.front();
    Point                       p0p1, pa;
    Vec2d                       _vect, _perp; // line vector
    for (int _it = 1; _it < (int) c_size; _it++) {
        Arachne::ExtrusionJunction& p1(ext_lines[_it]);
        const double                _thick = p1.w; // thickness alias
        size_t                      _index = p1.perimeter_index;
        p0p1                               = (p1.p - p0->p);
        pa                                 = p0->p;                                   // 'a' is the (next) new point between p0 and p1
        _vect                              = Vec2d(p0p1.cast<double>().normalized()); // line vector

        // Bypass processing an unclosed line if the first element is too short to filter out an unwanted corner
        if (!_is_loop && _it == 1 && (p0p1.norm() < cfg.point_distance))
            continue;

        // Orca: only skip the first point for closed path, open path should not skip any point
        if (closed && p0->p == p1.p) { // Connect endpoints.
            out_point(_vect, &p1, _params);
            continue;
        }

        _perp            = perp(_vect); // perpendicular vector of p0p1
        double p0p1_size = p0p1.norm();
        double p0pa_dist = dist_left_over;

        double theta = constrainPI(atan2(_vect.y(), _vect.x()) + corners[_it]);
        Vec2d  _vect2(cos(theta), sin(theta)); // vector of p1p2

        if (p0pa_dist > p0p1_size) { // for short lines need to build own point
            p0pa_dist -= p0p1_size;
            _params.params[0] = _overall + p0pa_dist;
            Arachne::ExtrusionJunction j(p1.p, _thick, _index);
            out_point(_vect, &j, _params);
        } else
            for (; p0pa_dist < p0p1_size; p0pa_dist += min_dist_between_points + random_value() * range_random_point_dist) {
                pa                = p0->p + (p0p1 * (p0pa_dist / p0p1_size)).cast<coord_t>();
                _params.params[0] = _overall + p0pa_dist;
                Arachne::ExtrusionJunction j(pa, _thick, _index);
                out_point(_vect, &j, _params);
            }

        // Stop processing an unclosed line if the last element is too short to filter out an unwanted corner
        if (!_is_loop && (_it == (int) (c_size - 1)) && ((ext_lines[_it + 1].p - p1.p).norm() < cfg.point_distance))
            break;

        _overall += p0p1_size;

        // Draw a needed point outside perimeter at visible corner
        double r_end = get_noise_value(*noise, p1.p, slice_z, _params.one_way_expansion) * cfg.thickness;
        if ((corners[_it] < 0.) != (r_end < 0. && !_params.one_way_expansion)) { // for outside corners only
            if (abs(corners[_it]) > M_PI_4 / 2.) {                               // for weight angles only
                double _rs, _re;
                Vec2d  _vect_d;
                Vec2d  sec_point;
                Vec2d  p1p = p1.p.cast<double>();
            } else {
                theta             = constrainPI(atan2(_vect.y(), _vect.x()) + corners[_it] / 2.);
                _params.params[0] = _overall + p0pa_dist;
                out_point(Vec2d(cos(theta), sin(theta)) / abs(cos(corners[_it] / 2.)), &p1, _params);
            }
        }

        dist_left_over = p0pa_dist - p0p1_size;
        p0             = &p1;
    }

    for (auto& point : out) // filter thick lines
        point.w = std::max(point.w, minimal_line);

#ifdef DEBUG_FUZZY
    {
        int      i = 0;
        Polyline _tpl;

        for (auto& point : ext_lines) // filter thick lines
            _tpl.points.emplace_back(point.p);

        BoundingBox bbox = get_extents(_tpl);
        bbox.offset(scale_(1.));
        ::Slic3r::SVG svg(debug_out_path("fuzzy_skin_z%.2f_%d.svg", slice_z, _params.one_way_expansion ? 0 : 1).c_str(), bbox);
        svg.draw(_tpl, "red");

        _tpl.clear();
        for (auto& point : out) // filter thick lines
            _tpl.points.emplace_back(point.p);
        svg.draw(_tpl, "orange");

        Polyline _pl(_tpl.points);
        _pl.simplify(scaled(EPSILON));
        svg.draw(_pl, "blue");

        _tpl.clear();
        for (auto& point : fuzzy_filter_by_polygon(out)) // filter thick lines
            _tpl.points.emplace_back(point.p);
        svg.draw(_tpl, "green");

        _tpl.clear();
        for (auto& point : out2) // filter thick lines
            _tpl.points.emplace_back(point.p);
        svg.draw(_tpl, "yellow");

        svg.Close();
        i++;
    }
#endif

    while (out.size() < 3) {
        size_t point_idx = ext_lines.size() - 2;
        out.emplace_back(ext_lines[point_idx].p, ext_lines[point_idx].w, ext_lines[point_idx].perimeter_index);
        if (point_idx == 0)
            break;
        --point_idx;
    }

    if (out.size() >= 3)
        ext_lines = std::move(out);
}

void group_region_by_fuzzify(PerimeterGenerator& g)
{
    g.regions_by_fuzzify.clear();
    g.has_fuzzy_skin = false;
    g.has_fuzzy_hole = false;

    std::unordered_map<FuzzySkinConfig, SurfacesPtr> regions;
    for (auto region : *g.compatible_regions) {
        const auto&   region_config = region->region().config();
        const coord_t _wall_width(scaled<coord_t>((region_config.outer_wall_line_width.value ? region_config.outer_wall_line_width.value :
                                                                                               g.object_config->line_width.value)) *
                                  g.print_config->filament_flow_ratio.get_at(g.print_config->master_extruder_id.value));
        const FuzzySkinConfig cfg{region_config.fuzzy_skin,
                                  scaled<coord_t>(region_config.fuzzy_skin_thickness.value),
                                  scaled<coord_t>(region_config.fuzzy_skin_point_distance.value),
                                  region_config.fuzzy_skin_first_layer,
                                  region_config.fuzzy_skin_noise_type,
                                  region_config.fuzzy_skin_scale,
                                  region_config.fuzzy_skin_octaves,
                                  region_config.fuzzy_skin_persistence,
                                  region_config.fuzzy_skin_mode,
                                  region_config.fuzzy_skin_ripples_per_layer,
                                  region_config.fuzzy_skin_ripple_offset,
                                  region_config.fuzzy_skin_layers_between_ripple_offset,
                                  _wall_width,
                                  scaled<coord_t>(std::max(float(g.layer_height), g.ext_perimeter_flow.spacing()) *
                                                  0.215), // param * (1. - 0.25 * PI)
                                  g.layer_id};
        auto&                 surfaces = regions[cfg];
        for (const auto& surface : region->slices.surfaces) {
            surfaces.push_back(&surface);
        }

        if (cfg.type != FuzzySkinType::None && cfg.type != FuzzySkinType::Disabled_fuzzy) {
            g.has_fuzzy_skin = true;
            if (cfg.type != FuzzySkinType::External) {
                g.has_fuzzy_hole = true;
            }
        }
    }

    if (regions.size() == 1) { // optimization
        g.regions_by_fuzzify[regions.begin()->first] = {};
        return;
    }

    for (auto& it : regions) {
        g.regions_by_fuzzify[it.first] = offset_ex(it.second, ClipperSafetyOffset);
    }
}

bool should_fuzzify(const FuzzySkinConfig& config, const int layer_id, const size_t loop_idx, const bool is_contour)
{
    const auto fuzziy_type = config.type;

    if (fuzziy_type == FuzzySkinType::None || fuzziy_type == FuzzySkinType::Disabled_fuzzy) {
        return false;
    }
    if (!config.fuzzy_first_layer && layer_id <= 0) {
        // Do not fuzzy first layer unless told to
        return false;
    }

    const bool fuzzify_contours = loop_idx == 0 || fuzziy_type == FuzzySkinType::AllWalls;
    const bool fuzzify_holes    = fuzzify_contours && (fuzziy_type == FuzzySkinType::All || fuzziy_type == FuzzySkinType::AllWalls);

    return is_contour ? fuzzify_contours : fuzzify_holes;
}

Polygon apply_fuzzy_skin(const Polygon& polygon, const PerimeterGenerator& perimeter_generator, const size_t loop_idx, const bool is_contour)
{
    Polygon fuzzified;

    const auto  slice_z = perimeter_generator.slice_z;
    const auto& regions = perimeter_generator.regions_by_fuzzify;
    if (regions.size() == 1) { // optimization
        const auto& config  = regions.begin()->first;
        const bool  fuzzify = should_fuzzify(config, perimeter_generator.layer_id, loop_idx, is_contour);
        if (!fuzzify) {
            return polygon;
        }

        fuzzified = polygon;
        fuzzy_polyline(fuzzified.points, true, slice_z, config);
        return fuzzified;
    }

    // Find all affective regions
    std::vector<std::pair<const FuzzySkinConfig&, const ExPolygons&>> fuzzified_regions;
    fuzzified_regions.reserve(regions.size());
    for (const auto& region : regions) {
        if (should_fuzzify(region.first, perimeter_generator.layer_id, loop_idx, is_contour)) {
            fuzzified_regions.emplace_back(region.first, region.second);
        }
    }
    if (fuzzified_regions.empty()) {
        return polygon;
    }

#ifdef DEBUG_FUZZY
    {
        int i = 0;
        for (const auto& r : fuzzified_regions) {
            BoundingBox bbox = get_extents(perimeter_generator.slices->surfaces);
            bbox.offset(scale_(1.));
            ::Slic3r::SVG svg(debug_out_path("fuzzy_traverse_loops_%d_%d_%d_region_%d.svg", perimeter_generator.layer_id,
                                             is_contour ? 0 : 1, loop_idx, i)
                                  .c_str(),
                              bbox);
            svg.draw_outline(perimeter_generator.slices->surfaces);
            svg.draw_outline(polygon, "green");
            svg.draw(r.second, "red", 0.5);
            svg.draw_outline(r.second, "red");
            svg.Close();
            i++;
        }
    }
#endif

    // Split the loops into lines with different config, and fuzzy them separately
    fuzzified = polygon;
    for (const auto& r : fuzzified_regions) {
        auto splitted = Algorithm::split_line(fuzzified, r.second, true);
        if (splitted.empty()) {
            // No intersection, skip
            continue;
        }

        // Fuzzy splitted polygon
        if (std::all_of(splitted.begin(), splitted.end(), [](const Algorithm::SplitLineJunction& j) { return j.clipped; })) {
            // The entire polygon is fuzzified
            fuzzy_polyline(fuzzified.points, true, slice_z, r.first);
        } else {
            // Start from a non-clipped junction so wrapped clipped segments do
            // not need an artificial reconnection across the seam.
            const auto first_non_clipped = std::find_if(splitted.begin(), splitted.end(),
                                                        [](const Algorithm::SplitLineJunction& j) { return !j.clipped; });
            if (first_non_clipped != splitted.begin()) {
                std::rotate(splitted.begin(), first_non_clipped, splitted.end());
            }
            Points segment;
            segment.reserve(splitted.size());
            fuzzified.points.clear();

            const auto fuzzy_current_segment = [&segment, &fuzzified, &r, slice_z]() {
                fuzzified.points.push_back(segment.front());
                const auto back = segment.back();
                fuzzy_polyline(segment, false, slice_z, r.first);
                fuzzified.points.insert(fuzzified.points.end(), segment.begin(), segment.end());
                fuzzified.points.push_back(back);
                segment.clear();
            };

            for (const auto& p : splitted) {
                if (p.clipped) {
                    segment.push_back(p.p);
                } else {
                    if (segment.empty()) {
                        fuzzified.points.push_back(p.p);
                    } else {
                        segment.push_back(p.p);
                        fuzzy_current_segment();
                    }
                }
            }
            if (!segment.empty()) {
                // Close the loop
                segment.push_back(splitted.front().p);
                fuzzy_current_segment();
            }
        }
    }

    return fuzzified;
}

void apply_fuzzy_skin(Arachne::ExtrusionLine* extrusion, const PerimeterGenerator& perimeter_generator, const bool is_contour)
{
    const auto  slice_z = perimeter_generator.slice_z;
    const auto& regions = perimeter_generator.regions_by_fuzzify;
    if (regions.size() == 1) { // optimization
        const auto& config  = regions.begin()->first;
        const bool  fuzzify = should_fuzzify(config, perimeter_generator.layer_id, extrusion->inset_idx, is_contour);
        if (fuzzify)
            fuzzy_extrusion_line(extrusion->junctions, slice_z, config);
    } else {
        // Find all affective regions
        std::vector<std::pair<const FuzzySkinConfig&, const ExPolygons&>> fuzzified_regions;
        fuzzified_regions.reserve(regions.size());
        for (const auto& region : regions) {
            if (should_fuzzify(region.first, perimeter_generator.layer_id, extrusion->inset_idx, is_contour)) {
                fuzzified_regions.emplace_back(region.first, region.second);
            }
        }
        if (!fuzzified_regions.empty()) {
 
#ifdef DEBUG_FUZZY
            {
                int i = 0;
                for (const auto& r : fuzzified_regions) {
                    BoundingBox bbox = get_extents(perimeter_generator.slices->surfaces);
                    bbox.offset(scale_(1.));
                    ::Slic3r::SVG svg(debug_out_path("fuzzy_traverse_loops_%d_%d_%d_region_%d.svg", perimeter_generator.layer_id,
                                                     is_contour ? 0 : 1, extrusion->inset_idx, i)
                                          .c_str(),
                                      bbox);

                    // Convert extrusion line to polygon for visualization
                    Polygon extrusion_polygon;
                    extrusion_polygon.points.reserve(extrusion->junctions.size());
                    for (const auto& junction : extrusion->junctions) {
                        extrusion_polygon.points.push_back(junction.p);
                    }

                    svg.draw_outline(perimeter_generator.slices->surfaces);
                    svg.draw_outline(extrusion_polygon, "green");
                    svg.draw(r.second, "red", 0.5);
                    svg.draw_outline(r.second, "red");
                    svg.Close();
                    i++;
                }
            }
#endif

            // Split the loops into lines with different config, and fuzzy them separately
            for (const auto& r : fuzzified_regions) {
                const auto splitted = Algorithm::split_line(*extrusion, r.second, false);
                if (splitted.empty()) {
                    // No intersection, skip
                    continue;
                }

                // Fuzzy splitted extrusion
                if (std::all_of(splitted.begin(), splitted.end(), [](const Algorithm::SplitLineJunction& j) { return j.clipped; })) {
                    // The entire polygon is fuzzified
                    fuzzy_extrusion_line(extrusion->junctions, slice_z, r.first);
                    continue;
                } else {
                    const auto                              current_ext = extrusion->junctions;
                    std::vector<Arachne::ExtrusionJunction> segment;
                    segment.reserve(current_ext.size());
                    extrusion->junctions.clear();

                    const auto fuzzy_current_segment = [&segment, &extrusion, &r, slice_z]() {
                        // Orca: non fuzzy points to isolate fuzzy region
                        const auto front = segment.front();
                        const auto back  = segment.back();

                        fuzzy_extrusion_line(segment, slice_z, r.first, false);
                        // Orca: only add non fuzzy point if it's not in the extrusion closing point.
                        if (!extrusion->junctions.empty() && extrusion->junctions.front().p != front.p) {
                            extrusion->junctions.push_back(front);
                        }
                        extrusion->junctions.insert(extrusion->junctions.end(), segment.begin(), segment.end());
                        // Orca: only add non fuzzy point if it's not in the extrusion closing point.
                        if (!extrusion->junctions.empty() && extrusion->junctions.back().p != front.p) {
                            extrusion->junctions.push_back(back);
                        }
                        segment.clear();
                    };

                    const auto to_ex_junction = [&current_ext](const Algorithm::SplitLineJunction& j) -> Arachne::ExtrusionJunction {
                        Arachne::ExtrusionJunction res = current_ext[j.get_src_index()];
                        if (!j.is_src()) {
                            res.p = j.p;
                        }
                        return res;
                    };

                    for (const auto& p : splitted) {
                        if (p.clipped) {
                            segment.push_back(to_ex_junction(p));
                        } else {
                            if (segment.empty()) {
                                extrusion->junctions.push_back(to_ex_junction(p));
                            } else {
                                segment.push_back(to_ex_junction(p));
                                fuzzy_current_segment();
                            }
                        }
                    }
                    if (!segment.empty()) {
                        fuzzy_current_segment();
                    }

                    //Orca: ensure the loop is closed after fuzzy
                    if (!extrusion->junctions.empty() && extrusion->junctions.front().p != extrusion->junctions.back().p) {
                        extrusion->junctions.back().p = extrusion->junctions.front().p;
                        extrusion->junctions.back().w = extrusion->junctions.front().w;
                    }
                }
            }
        }

    }
}

} // namespace Slic3r::Feature::FuzzySkin
