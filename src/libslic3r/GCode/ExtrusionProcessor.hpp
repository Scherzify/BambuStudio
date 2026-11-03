#ifndef slic3r_ExtrusionProcessor_hpp_
#define slic3r_ExtrusionProcessor_hpp_

// Dynamic per-point overhang slowdown for Bambu Studio.
//
// The algorithm originates from PrusaSlicer (original author Pavel Mikus,
// pavel.mikus.mail@seznam.cz) and was ported from OrcaSlicer's
// src/libslic3r/GCode/ExtrusionProcessor.hpp, where it powers the
// "Slowdown for overhang" feature.
//
// Adaptations for Bambu Studio (compared to the OrcaSlicer version):
//  - 2D signed-distance variant (AABBTreeLines::LinesDistancer<Linef>) instead of Linef3.
//  - The lower-layer boundaries tree is built on demand in
//    ExtrusionQualityEstimator::prepare_for_new_layer(); the parallel
//    precompute_overhang_layers() pipeline of OrcaSlicer is not needed here.
//  - No curled-lines slowdown, no overhang-fan switching (Bambu Studio keeps
//    its degree-based overhang fan markers) and no curvature computation.
//  - estimate_extrusion_quality() consumes absolute speeds, the caller builds
//    the sections from the existing Bambu Studio overhang speed keys
//    (overhang_1_4_speed ... overhang_4_4_speed, bridge_speed).

#include "../AABBTreeLines.hpp"
#include "../libslic3r.h"
#include "../ExtrusionEntity.hpp"
#include "../Layer.hpp"
#include "../Point.hpp"
#include "../Polygon.hpp"
#include "../Flow.hpp"
#include "../Config.hpp"
#include "../ExPolygon.hpp"
#include "../Line.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Slic3r {

class PrintObject;

template<int Dim> struct ExtendedPoint
{
    Eigen::Matrix<double, Dim, 1, Eigen::DontAlign> position;
    float distance;
};

// Result of the per-point overhang evaluation of one extrusion path.
// `p`      : scaled print coordinates of the (possibly inserted) point,
// `speed`  : speed [mm/s] the segment starting at this point shall print at,
// `overlap`: remaining overlap with the layer below at this point [0..1].
struct ProcessedPoint
{
    Point p;
    float speed   = 1.0f;
    float overlap = 1.0f;
};

template<bool SCALED_INPUT, bool ADD_INTERSECTIONS, bool PREV_LAYER_BOUNDARY_OFFSET, bool SIGNED_DISTANCE, typename POINTS, typename L>
std::vector<ExtendedPoint<L::Dim>> estimate_points_properties(const POINTS&                                   input_points,
                                                              const AABBTreeLines::LinesDistancer<L>&         unscaled_prev_layer,
                                                              float                                           flow_width,
                                                              float                                           max_line_length = -1.0f,
                                                              float                                           min_distance    = -1.0f,
                                                              const std::function<float(float)>&              distance_to_speed = {})
{
    using Vec        = Eigen::Matrix<double, L::Dim, 1, Eigen::DontAlign>;
    using AABBScalar = typename AABBTreeLines::LinesDistancer<L>::Scalar;

    bool   looped = input_points.front() == input_points.back();
    std::function<size_t(size_t, size_t)> get_prev_index = [](size_t idx, size_t count) {
        if (idx > 0) {
            return idx - 1;
        } else
            return idx;
    };
    if (looped) {
        get_prev_index = [](size_t idx, size_t count) {
            if (idx == 0)
                idx = count;
            return --idx;
        };
    }
    std::function<size_t(size_t, size_t)> get_next_index = [](size_t idx, size_t size) {
        if (idx + 1 < size) {
            return idx + 1;
        } else
            return idx;
    };
    if (looped) {
        get_next_index = [](size_t idx, size_t count) {
            if (++idx == count)
                idx = 0;
            return idx;
        };
    }

    using P = typename POINTS::value_type;
    // Minimum spacing threshold for any newly generated points. 25% of the flow width keeps
    // the points spaced far enough apart to avoid micro stutters while the movement of the
    // print head is still fine-grained enough to maintain print quality.
    double min_spacing = flow_width * 0.25;

    if (input_points.empty())
        return {};
    float boundary_offset = PREV_LAYER_BOUNDARY_OFFSET ? 0.5 * flow_width : 0.0f;
    auto  maybe_unscale   = [](const P& p) -> Vec { return SCALED_INPUT ? unscaled(p).template cast<double>() : p.template cast<double>(); };

    std::vector<ExtendedPoint<L::Dim>> points;
    points.reserve(input_points.size() * (ADD_INTERSECTIONS ? 1.5 : 1));

    {
        ExtendedPoint<L::Dim> start_point{maybe_unscale(input_points.front())};
        auto [start_dist, start_idx, start_np] = unscaled_prev_layer.template distance_from_lines_extra<SIGNED_DISTANCE>(
            start_point.position.template cast<AABBScalar>());
        start_point.distance = float(start_dist + boundary_offset);
        points.push_back(start_point);
    }
    for (size_t i = 1; i < input_points.size(); i++) {
        ExtendedPoint<L::Dim> next_point{maybe_unscale(input_points[i])};
        auto [next_dist, next_idx, next_np] = unscaled_prev_layer.template distance_from_lines_extra<SIGNED_DISTANCE>(
            next_point.position.template cast<AABBScalar>());
        next_point.distance = float(next_dist + boundary_offset);

        // Intersection handling
        if (ADD_INTERSECTIONS &&
            ((points.back().distance > boundary_offset + EPSILON) != (next_point.distance > boundary_offset + EPSILON))) {
            const ExtendedPoint<L::Dim>& prev_point = points.back();
            auto intersections = unscaled_prev_layer.template intersections_with_line<true>(
                L{prev_point.position.template cast<AABBScalar>(), next_point.position.template cast<AABBScalar>()});
            for (const auto &intersection : intersections) {
                ExtendedPoint<L::Dim> p{};
                p.position = intersection.first.template cast<double>();
                p.distance = boundary_offset;
                // Filter out points introduced at intersections if their distance from the previous
                // or next point is not meaningful.
                if ((p.position - prev_point.position).norm() > min_spacing &&
                    (next_point.position - p.position).norm() > min_spacing) {
                    points.push_back(p);
                }
            }
        }
        points.push_back(next_point);
    }

    // How an overhang distance prints, which is what the passes below compare. A segment is
    // printed at the lower of the speeds at its two ends, so a point added to a path can only
    // change the G-code where it prints at a different speed from the points either side of it;
    // the passes below add points there and nowhere else.
    // Whether two overhang distances are interchangeable, i.e. have the same speed (beyond the
    // 1mm/sec threshold that GCode.cpp filters out on).
    auto same_speed = [&distance_to_speed](float a, float b) { return std::abs(distance_to_speed(a) - distance_to_speed(b)) <= 1.f; };
    // Whether the first overhang distance prints slower than the second, beyond the 1mm/sec
    // tolerance. Against a supported point (overhang distance 0) it tells whether an end is
    // affected by the overhang.
    auto slower = [&distance_to_speed](float a, float b) { return distance_to_speed(a) < distance_to_speed(b) - 1.f; };

    // Interior sampling.
    // The passes below infer the support under a span from its endpoints alone, so a part that is
    // supported differently from both ends is invisible to them. The outer perimeter of an overhang
    // whose ends are supported reads as supported along its whole length. Probe the interior, keep
    // the samples the endpoint interpolation fails to predict, and bisect either side of each one,
    // so a span that is only partly unsupported gets points where its support actually changes
    // instead of one reading being spread across all of it.
    // Skipped where there is nothing to find: min_distance <= 0 when no overhang can slow this
    // path down. It also needs distance_to_speed to tell which of the points it could add would
    // change the G-code.
    if (PREV_LAYER_BOUNDARY_OFFSET && ADD_INTERSECTIONS && distance_to_speed && min_distance > 0) {
        // Probe at least this densely before treating matching samples as evidence that a span is
        // uniform. The segmentation pass below only splits lines of 2mm or more, and every pass
        // here drops points closer together than min_spacing, so finer discovery would not produce
        // a more precise transition.
        const double max_probe_spacing = std::max(2., 4. * min_spacing);
        // A backstop for that length test, which on a non-finite length would never be met.
        constexpr int max_bisection_depth = 10;

        // Part of a segment still to bisect: its positions along the segment and bisections left.
        struct Subspan
        {
            double t0, t1;
            int    depth;
        };

        std::vector<ExtendedPoint<L::Dim>>    sampled_points; // Populated lazily, on the first insertion
        std::vector<std::pair<double, float>> interior;       // Samples of one segment, keyed by position along it
        std::vector<Subspan>                  pending;

        for (size_t point_idx = 0; point_idx + 1 < points.size(); ++point_idx) {
            const ExtendedPoint<L::Dim>& curr = points[point_idx];
            const ExtendedPoint<L::Dim>& next = points[point_idx + 1];
            const Vec                    step = next.position - curr.position;
            const double                 line_len = step.norm();

            interior.clear();
            if (line_len >= max_probe_spacing)
                pending.push_back({0., 1., max_bisection_depth});

            while (!pending.empty()) {
                const Subspan subspan = pending.back();
                pending.pop_back();
                if (subspan.depth <= 0 || (subspan.t1 - subspan.t0) * line_len < max_probe_spacing)
                    continue;

                const double t = 0.5 * (subspan.t0 + subspan.t1);
                auto [distance, nearest_line, x] = unscaled_prev_layer.template distance_from_lines_extra<SIGNED_DISTANCE>(
                    (curr.position + t * step).template cast<AABBScalar>());
                const float sampled = float(distance + boundary_offset);

                interior.emplace_back(t, sampled);
                pending.push_back({subspan.t0, t, subspan.depth - 1});
                pending.push_back({t, subspan.t1, subspan.depth - 1});
            }

            if (!interior.empty()) {
                std::sort(interior.begin(), interior.end(),
                          [](const std::pair<double, float>& l, const std::pair<double, float>& r) { return l.first < r.first; });
                // Coarse probing keeps every sample it took until this pass can see which ones
                // bracket a speed transition. Matching samples cannot be discarded during
                // discovery: one may be the last supported point before a narrow unsupported
                // pocket found by a later probe.
                size_t kept = 0;
                for (size_t i = 0; i < interior.size(); ++i) {
                    const float sample   = interior[i].second;
                    const bool  at_start = kept == 0;                // Nothing kept yet, so the segment's own start precedes it
                    const bool  at_end   = i + 1 == interior.size(); // And nothing follows the last sample but the segment's end
                    const float before   = at_start ? curr.distance : interior[kept - 1].second;
                    const float after    = at_end ? next.distance : interior[i + 1].second;
                    // A sample is worth a point in the path only where it prints differently, in
                    // speed, from the points either side of it. Differing from one of the
                    // segment's own ends is not enough on its own where the sample is not the
                    // slower of the two: the segmentation pass below already confines the
                    // slowdown at an end. Support an end cannot account for, where the interior
                    // is the slower of the two, is exactly what this pass is here to find.
                    const bool worth_before = !same_speed(sample, before) && (!at_start || slower(sample, before));
                    const bool worth_after  = !same_speed(sample, after) && (!at_end || slower(sample, after));
                    if (worth_before || worth_after)
                        interior[kept++] = interior[i];
                }
                interior.resize(kept);
            }

            if (!interior.empty() && sampled_points.empty()) {
                sampled_points.reserve(points.size() + 8);
                sampled_points.assign(points.begin(), points.begin() + point_idx + 1);
            }
            if (!sampled_points.empty()) {
                // Only a sub-span of max_probe_spacing or more is ever bisected, so these sit at
                // least 2 * min_spacing apart, and need none of the filtering the passes either
                // side of this one do.
                for (const auto& [t, distance] : interior)
                    sampled_points.push_back({curr.position + t * step, distance});
                sampled_points.push_back(next);
            }
        }
        if (!sampled_points.empty())
            points = std::move(sampled_points);
    }

    // Segmentation handling
    if (PREV_LAYER_BOUNDARY_OFFSET && ADD_INTERSECTIONS) {
        std::vector<ExtendedPoint<L::Dim>> new_points;
        new_points.reserve(points.size() * 2);
        new_points.push_back(points.front());
        for (int point_idx = 0; point_idx < int(points.size()) - 1; ++point_idx) {
            const ExtendedPoint<L::Dim>& curr = points[point_idx];
            const ExtendedPoint<L::Dim>& next = points[point_idx + 1];

            if ((curr.distance > -boundary_offset && curr.distance < boundary_offset + 2.0f) ||
                (next.distance > -boundary_offset && next.distance < boundary_offset + 2.0f)) {
                double line_len = (next.position - curr.position).norm();

                // A line prints as slow as its slower end, so an overhang at one end would
                // otherwise slow down the whole line. Split the line so that only the part beside
                // that end prints that way, if the line is reasonably long (2mm or more) and at
                // least one end prints slower compared with a supported point (overhang distance
                // 0). Without distance_to_speed, split every line over 4mm.
                const bool split_line = distance_to_speed ?
                    line_len >= 2.f && (slower(curr.distance, 0.f) || slower(next.distance, 0.f)) :
                    line_len > 4.0f;
                if (split_line) {
                    // Each end's piece is that end's overhang distance plus 1.5 line widths
                    // (3 * boundary_offset) long: a0 ends the piece beside curr, a1 starts the
                    // piece beside next.
                    double a0 = std::clamp((curr.distance + 3 * boundary_offset) / line_len, 0.0, 1.0);
                    double a1 = std::clamp(1.0f - (next.distance + 3 * boundary_offset) / line_len, 0.0, 1.0);
                    double t0 = std::min(a0, a1);
                    double t1 = std::max(a0, a1);

                    // Up to two cut points, in order along the line. Each takes its own overhang
                    // distance, so every piece prints by the overhang distances at its own two
                    // ends. t0 >= 1 or t1 <= 0 falls on the line's own end, so there is no cut. A
                    // cut closer than min_spacing to either end of the line is not meaningful and
                    // is filtered out.
                    ExtendedPoint<L::Dim> cut[2]{};
                    bool                  keep[2] = {false, false};
                    for (int k = 0; k < 2; ++k) {
                        const double t = k == 0 ? t0 : t1;
                        if (k == 0 ? t >= 1.0 : t <= 0.0)
                            continue;
                        const Vec p                  = curr.position + t * (next.position - curr.position);
                        auto [p_dist, p_near_l, p_x] = unscaled_prev_layer.template distance_from_lines_extra<SIGNED_DISTANCE>(
                            p.template cast<AABBScalar>());
                        cut[k].position = p;
                        cut[k].distance = float(p_dist + boundary_offset);
                        keep[k]         = (p - curr.position).norm() > min_spacing && (next.position - p).norm() > min_spacing;
                    }
                    if (distance_to_speed) {
                        // Only keep a cut that changes the G-code: one that prints differently
                        // from at least one of the points either side of it, which are the line's
                        // ends or the other cut where that is kept. A cut that prints like both
                        // would only split a move into two identical ones.
                        if (keep[0])
                            keep[0] = !same_speed(cut[0].distance, curr.distance) ||
                                      !same_speed(cut[0].distance, keep[1] ? cut[1].distance : next.distance);
                        if (keep[1])
                            keep[1] = !same_speed(cut[1].distance, keep[0] ? cut[0].distance : curr.distance) ||
                                      !same_speed(cut[1].distance, next.distance);
                        // Two cuts closer together than min_spacing would leave a micro segment
                        // between them, so only the first is kept.
                        if (keep[0] && keep[1] && (cut[1].position - cut[0].position).norm() <= min_spacing)
                            keep[1] = false;
                    }
                    for (int k = 0; k < 2; ++k)
                        if (keep[k])
                            new_points.push_back(cut[k]);
                }
            }
            new_points.push_back(next);
        }
        points = std::move(new_points);
    }

    // Maximum line length handling
    if (max_line_length > 0) {
        std::vector<ExtendedPoint<L::Dim>> new_points;
        new_points.reserve(points.size() * 2);
        {
            for (size_t i = 0; i + 1 < points.size(); i++) {
                const ExtendedPoint<L::Dim>& curr = points[i];
                const ExtendedPoint<L::Dim>& next = points[i + 1];
                new_points.push_back(curr);
                double len             = (next.position - curr.position).squaredNorm();
                double t               = sqrt((max_line_length * max_line_length) / len);
                size_t new_point_count = 1.0 / t;
                for (size_t j = 1; j < new_point_count + 1; j++) {
                    Vec pos = curr.position * (1.0 - j * t) + next.position * (j * t);
                    auto [p_dist, p_near_l, p_x] = unscaled_prev_layer.template distance_from_lines_extra<SIGNED_DISTANCE>(
                        pos.template cast<AABBScalar>());
                    ExtendedPoint<L::Dim> new_p{};
                    new_p.position = pos;
                    new_p.distance = float(p_dist + boundary_offset);

                    // Filter out points that are introduced if their distance from the previous
                    // or next point is not meaningful.
                    if ((pos - curr.position).norm() > min_spacing && (next.position - pos).norm() > min_spacing) {
                        new_points.push_back(new_p);
                    }
                }
            }
            new_points.push_back(points.back());
        }
        points = std::move(new_points);
    }

    return points;
}

class ExtrusionQualityEstimator
{
    using Boundaries = AABBTreeLines::LinesDistancer<Linef>;
    std::unordered_map<const PrintObject *, std::shared_ptr<const Boundaries>> prev_layer_boundaries;
    // The layers the trees above are built from.
    std::unordered_map<const PrintObject *, const Layer *>                     last_prepared_layers;
    const PrintObject                                                         *current_object = nullptr;

    template<typename T> static const T &or_empty(const std::shared_ptr<const T> &tree)
    {
        static const T empty;
        return tree ? *tree : empty;
    }

public:
    void set_current_object(const PrintObject *object) { current_object = object; }

    const PrintObject *get_current_object() const { return current_object; }

    // Whether the boundaries of the layer below the current layer of `object` are known; without
    // them no overhang estimation can run.
    bool has_prev_layer(const PrintObject *object) const
    {
        auto it = prev_layer_boundaries.find(object);
        return it != prev_layer_boundaries.end() && it->second != nullptr;
    }

    // Measures `layer` against the layer below it. Called once per object instance and layer from
    // GCode::process_layer(); safe to call repeatedly for the same layer (multiple copies).
    void prepare_for_new_layer(const PrintObject *obj, const Layer *layer)
    {
        if (layer == nullptr)
            return;
        if (last_prepared_layers[obj] == layer)
            return;
        last_prepared_layers[obj] = layer;
        if (layer->lower_layer == nullptr) {
            prev_layer_boundaries[obj] = nullptr;
            return;
        }
        prev_layer_boundaries[obj] = std::make_shared<const Boundaries>(to_unscaled_linesf(layer->lower_layer->lslices));
    }

    // Splits `path` into points and evaluates for each point the remaining overlap with the layer
    // below and the speed to print the following segment at. `overlaps` holds the supported-width
    // percentages where the speeds of `speeds` apply (same count), `ext_perimeter_speed` is the
    // reference speed the percentage entries of `speeds` relate to and `original_speed` is the
    // speed the whole path would print at without overhang slowdown (the result never exceeds it).
    std::vector<ProcessedPoint> estimate_extrusion_quality(const ExtrusionPath                &path,
                                                           const ConfigOptionPercents         &overlaps,
                                                           const ConfigOptionFloatsOrPercents &speeds,
                                                           float                               ext_perimeter_speed,
                                                           float                               original_speed)
    {
        size_t                               speed_sections_count = std::min(overlaps.values.size(), speeds.values.size());
        std::vector<std::pair<float, float>> speed_sections;

        for (size_t i = 0; i < speed_sections_count; i++) {
            float distance = path.width * (1.0 - (overlaps.get_at(i) / 100.0));
            float speed    = speeds.get_at(i).percent ? (ext_perimeter_speed * speeds.get_at(i).value / 100.0) : speeds.get_at(i).value;
            speed_sections.push_back({distance, speed});
        }
        std::sort(speed_sections.begin(), speed_sections.end(),
                  [](const std::pair<float, float> &a, const std::pair<float, float> &b) {
                      if (a.first == b.first) {
                          return a.second > b.second;
                      }
                      return a.first < b.first;
                  });

        std::pair<float, float> last_section{INFINITY, 0};
        for (auto &section : speed_sections) {
            if (section.first == last_section.first) {
                section.second = last_section.second;
            } else {
                last_section = section;
            }
        }

        // Find the smallest overhang distance where speed adjustments begin.
        float smallest_distance_with_lower_speed = std::numeric_limits<float>::infinity();
        bool  found                            = false;
        for (const auto &section : speed_sections) {
            if (section.second <= original_speed) {
                if (section.first < smallest_distance_with_lower_speed) {
                    smallest_distance_with_lower_speed = section.first;
                    found                              = true;
                }
            }
        }

        // If no overhang distance slows this path down, -1 turns interior sampling off.
        if (!found)
            smallest_distance_with_lower_speed = -1.f;

        auto calculate_speed = [&speed_sections, &original_speed](float distance) {
            float final_speed;
            if (distance <= speed_sections.front().first) {
                final_speed = original_speed;
            } else if (distance >= speed_sections.back().first) {
                final_speed = speed_sections.back().second;
            } else {
                size_t section_idx = 0;
                while (distance > speed_sections[section_idx + 1].first) {
                    section_idx++;
                }
                float t = (distance - speed_sections[section_idx].first) /
                          (speed_sections[section_idx + 1].first - speed_sections[section_idx].first);
                t           = std::clamp(t, 0.0f, 1.0f);
                final_speed = (1.0f - t) * speed_sections[section_idx].second + t * speed_sections[section_idx + 1].second;
            }
            return round(final_speed);
        };

        // The speed sections may be built from a reference speed that can be above the speed this
        // path prints at (original_speed, e.g. held down by volumetric flow limits). Every segment
        // is capped at original_speed below, so overhang distances whose speeds differ only above
        // it print the same and must not count as a speed change when the path is split.
        auto effective_speed = [&calculate_speed, original_speed](float distance) {
            return std::min(calculate_speed(distance), original_speed);
        };

        const Boundaries &prev_boundaries = or_empty(prev_layer_boundaries[current_object]);

        std::vector<ExtendedPoint<2>> extended_points =
            estimate_points_properties<true, true, true, true>(path.polyline.points, prev_boundaries, path.width, -1.f,
                                                               smallest_distance_with_lower_speed, effective_speed);

        const auto             width_inv = 1.0f / path.width;
        std::vector<ProcessedPoint> processed_points;
        processed_points.reserve(extended_points.size());
        for (size_t i = 0; i < extended_points.size(); i++) {
            const ExtendedPoint<2>& curr = extended_points[i];
            const ExtendedPoint<2>& next = extended_points[i + 1 < extended_points.size() ? i + 1 : i];

            float extrusion_speed = std::min(calculate_speed(curr.distance), calculate_speed(next.distance));
            // Clamp resulting speed to the lowest of the calculated speed based on the overhang
            // values and the current speed. Fixes a bug where the resulting overhang speed would
            // be higher than the current speed due to (for example) volumetric flow limits.
            extrusion_speed = std::min(extrusion_speed, original_speed);

            float overlap = std::min(1 - curr.distance * width_inv, 1 - next.distance * width_inv);

            processed_points.push_back({Point(scaled(curr.position.x()), scaled(curr.position.y())), extrusion_speed, overlap});
        }
        return processed_points;
    }
};

} // namespace Slic3r

#endif // slic3r_ExtrusionProcessor_hpp_
