using Real = SOLWEIG_REAL;

namespace {

template <typename T> struct __align__(16) GvfNeighborQuad {
    T shadow, lup, albedo_shadow, albedo;
};

template <typename T> struct GvfArithmetic;

template <> struct GvfArithmetic<float> {
    __device__ __forceinline__ static float add(float a, float b) {
        return __fadd_rn(a, b);
    }
    __device__ __forceinline__ static float multiply(float a, float b) {
        return __fmul_rn(a, b);
    }
    __device__ __forceinline__ static float divide(float a, float b) {
        return __fdiv_rn(a, b);
    }
};

template <> struct GvfArithmetic<double> {
    __device__ __forceinline__ static double add(double a, double b) {
        return __dadd_rn(a, b);
    }
    __device__ __forceinline__ static double multiply(double a, double b) {
        return __dmul_rn(a, b);
    }
    __device__ __forceinline__ static double divide(double a, double b) {
        return __ddiv_rn(a, b);
    }
};

template <typename T> struct GvfTriple {
    T lup, albedo, albedo_no_shadow;

    __device__ __forceinline__ void accumulate(T l, T a, T n) {
        typedef GvfArithmetic<T> A;
        lup = A::add(lup, l);
        albedo = A::add(albedo, a);
        albedo_no_shadow = A::add(albedo_no_shadow, n);
    }

    __device__ __forceinline__ void write(
        T* output, long long pixel, long long pixels, int plane,
        T divisor, T ambient_lup) const {
        typedef GvfArithmetic<T> A;
        output[static_cast<long long>(plane) * pixels + pixel] =
            A::add(A::divide(lup, divisor), ambient_lup);
        output[static_cast<long long>(plane + 1) * pixels + pixel] =
            A::divide(albedo, divisor);
        output[static_cast<long long>(plane + 2) * pixels + pixel] =
            A::divide(albedo_no_shadow, divisor);
    }
};

template <typename T>
__device__ __forceinline__ T gvf_influence(
    T wall_sum, T ground_sum, T wall_divisor, T ground_divisor,
    bool has_wall_influence) {
    typedef GvfArithmetic<T> A;
    return A::add(
        A::multiply(A::divide(A::add(wall_sum, ground_sum), wall_divisor),
                    static_cast<T>(has_wall_influence)),
        A::multiply(A::divide(ground_sum, ground_divisor),
                    static_cast<T>(!has_wall_influence)));
}

template <typename T>
__device__ __forceinline__ T gvf_distance_weight(T near_value, T far_value) {
    typedef GvfArithmetic<T> A;
    return A::divide(
        A::add(A::multiply(near_value, static_cast<T>(0.5)),
               A::multiply(far_value, static_cast<T>(0.4))),
        static_cast<T>(0.9));
}

template <typename T, bool Compact = false>
__device__ __forceinline__ void gvf_pixel(
    const T* fields, const double* aspect, const int* offsets,
    const double* shade_bounds, int rows, int cols,
    int first_steps, int second_steps, T albedo_b, T* output,
    const T* neighbors = nullptr, const unsigned char* masks = nullptr) {
    typedef GvfArithmetic<T> A;
    const long long pixels = static_cast<long long>(rows) * cols;
    const long long pixel = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const int row = static_cast<int>(pixel / cols);
    const int col = static_cast<int>(pixel % cols);
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T first = static_cast<T>(first_steps);
    const T second = static_cast<T>(second_steps);
    const T first_plus_one = A::add(first, one);
    const T second_plus_one = A::add(second, one);

    const T building = fields[pixel];
    const T local_wall_lup = fields[7 * pixels + pixel];
    const T local_lup_correction = fields[8 * pixels + pixel];
    const T local_albedo_correction = fields[9 * pixels + pixel];
    const T local_albedo_no_shadow = fields[10 * pixels + pixel];
    const T ambient_lup = fields[11 * pixels + pixel];
    const T wallbol = fields[12 * pixels + pixel];
    const double local_aspect = aspect[pixel];

    GvfTriple<T> all = {zero, zero, zero};
    GvfTriple<T> east = {zero, zero, zero};
    GvfTriple<T> south = {zero, zero, zero};
    GvfTriple<T> west = {zero, zero, zero};
    GvfTriple<T> north = {zero, zero, zero};
    T gvf_sum = zero;

    for (int direction = 0; direction < 18; ++direction) {
        T f = building;
        T temp_building = zero, temp_shadow = zero, temp_lup = zero;
        T temp_albedo_shadow = zero, temp_albedo = zero, temp_sunwall = zero;
        T wall_seen = zero, any_wall_seen = zero;

        T shadow_sum = zero, wall_sum = zero;
        T ground_lup_sum = zero, wall_lup_sum = zero;
        T ground_albedo_shadow_sum = zero, wall_albedo_sum = zero;
        T ground_albedo_sum = zero, wall_albedo_no_shadow_sum = zero;

        T first_ground_lup = zero, first_wall_lup = zero;
        T first_ground_albedo_shadow = zero, first_wall_albedo = zero;
        T first_ground_albedo = zero, first_wall_albedo_no_shadow = zero;
        bool first_sunwall_influence = false;

        const T* neighbor_lup = fields + (direction == 0 ? 3 : 4) * pixels;

        for (int step = 0; step < second_steps; ++step) {
            const long long offset_index =
                (static_cast<long long>(direction) * second_steps + step) * 2;
            const long long source_row = static_cast<long long>(row) + offsets[offset_index];
            const long long source_col = static_cast<long long>(col) + offsets[offset_index + 1];

            if (source_row >= 0 && source_row < rows && source_col >= 0 && source_col < cols) {
                const long long source = source_row * cols + source_col;
                if (Compact) {
                    const GvfNeighborQuad<T> near =
                        reinterpret_cast<const GvfNeighborQuad<T>*>(neighbors)[source];
                    const unsigned char bits = masks[source];
                    temp_building = static_cast<T>(bits & 1u);
                    temp_shadow = near.shadow;
                    temp_sunwall = static_cast<T>((bits >> 1) & 1u);
                    temp_lup = direction == 0 ? neighbor_lup[source] : near.lup;
                    temp_albedo_shadow = near.albedo_shadow;
                    temp_albedo = near.albedo;
                } else {
                    temp_building = fields[source];
                    temp_shadow = fields[pixels + source];
                    temp_sunwall = fields[2 * pixels + source];
                    temp_lup = neighbor_lup[source];
                    temp_albedo_shadow = fields[5 * pixels + source];
                    temp_albedo = fields[6 * pixels + source];
                }
            }
            if (temp_building < f) f = temp_building;

            shadow_sum = A::add(shadow_sum, A::multiply(temp_shadow, f));
            ground_lup_sum = A::add(ground_lup_sum, A::multiply(temp_lup, f));
            ground_albedo_shadow_sum = A::add(
                ground_albedo_shadow_sum, A::multiply(temp_albedo_shadow, f));
            ground_albedo_sum = A::add(ground_albedo_sum, A::multiply(temp_albedo, f));

            const T temp_wall_sun = A::multiply(temp_sunwall, f);
            const T temp_wall = A::add(A::multiply(f, static_cast<T>(-1)), one);
            wall_seen = static_cast<T>(A::add(temp_wall_sun, wall_seen) > zero);
            any_wall_seen = static_cast<T>(A::add(temp_wall, any_wall_seen) > zero);
            wall_lup_sum = A::add(wall_lup_sum, A::multiply(wall_seen, local_wall_lup));
            wall_albedo_sum = A::add(wall_albedo_sum, A::multiply(wall_seen, albedo_b));
            wall_sum = A::add(wall_sum, wall_seen);
            wall_albedo_no_shadow_sum = A::add(
                wall_albedo_no_shadow_sum, A::multiply(any_wall_seen, albedo_b));

            if (step + 1 <= first_steps) {
                first_sunwall_influence = wall_sum > zero;
                first_wall_lup = wall_lup_sum;
                first_ground_lup = ground_lup_sum;
                first_wall_albedo = wall_albedo_sum;
                first_ground_albedo_shadow = ground_albedo_shadow_sum;
                first_wall_albedo_no_shadow = wall_albedo_no_shadow_sum;
                first_ground_albedo = ground_albedo_sum;
            }
        }

        const bool second_sunwall_influence = wall_sum > zero;
        const bool first_wall_influence = first_wall_albedo_no_shadow > zero;
        const bool second_wall_influence = wall_albedo_no_shadow_sum > zero;

        const double* bounds = shade_bounds + direction * 3;
        T facesh;
        if (bounds[2] == 0.0) {
            const bool outside = local_aspect < bounds[0] || local_aspect >= bounds[1];
            facesh = A::add(A::add(static_cast<T>(outside), -wallbol), one);
        } else {
            const bool outside = local_aspect > bounds[0] || local_aspect <= bounds[1];
            facesh = A::add(one, -static_cast<T>(outside));
        }
        T keep = A::add(static_cast<T>(wall_sum == second), -facesh);
        if (keep == static_cast<T>(-1)) keep = zero;
        if (keep == one) {
            wall_sum = zero;
            wall_lup_sum = zero;
            wall_albedo_sum = zero;
        }

        T gvf2 = gvf_influence(wall_sum, shadow_sum, second_plus_one,
                              second, second_sunwall_influence);
        if (gvf2 > one) gvf2 = one;

        const T lup_near = gvf_influence(
            first_wall_lup, first_ground_lup, first_plus_one,
            first, first_sunwall_influence);
        const T lup_far = gvf_influence(
            wall_lup_sum, ground_lup_sum, second_plus_one,
            second, second_sunwall_influence);
        const T direction_lup = A::add(
            gvf_distance_weight(lup_near, lup_far), local_lup_correction);

        const T albedo_near = gvf_influence(
            first_wall_albedo, first_ground_albedo_shadow, first_plus_one,
            first, first_sunwall_influence);
        const T albedo_far = gvf_influence(
            wall_albedo_sum, ground_albedo_shadow_sum, second_plus_one,
            second, second_sunwall_influence);
        const T direction_albedo = A::add(
            gvf_distance_weight(albedo_near, albedo_far), local_albedo_correction);

        const T no_shadow_near = gvf_influence(
            first_wall_albedo_no_shadow, first_ground_albedo, first_plus_one,
            first, first_wall_influence);
        const T no_shadow_far = gvf_influence(
            wall_albedo_no_shadow_sum, ground_albedo_sum, second,
            second, second_wall_influence);
        const T direction_no_shadow = A::add(
            A::multiply(gvf_distance_weight(no_shadow_near, no_shadow_far), building),
            local_albedo_no_shadow);

        all.accumulate(direction_lup, direction_albedo, direction_no_shadow);
        gvf_sum = A::add(gvf_sum, gvf2);
        const int azimuth = 5 + 20 * direction;
        if (azimuth >= 0 && azimuth < 180)
            east.accumulate(direction_lup, direction_albedo, direction_no_shadow);
        if (azimuth >= 90 && azimuth < 270)
            south.accumulate(direction_lup, direction_albedo, direction_no_shadow);
        if (azimuth >= 180 && azimuth < 360)
            west.accumulate(direction_lup, direction_albedo, direction_no_shadow);
        if (azimuth >= 270 || azimuth < 90)
            north.accumulate(direction_lup, direction_albedo, direction_no_shadow);
    }

    all.write(output, pixel, pixels, 0, static_cast<T>(18), ambient_lup);
    east.write(output, pixel, pixels, 3, static_cast<T>(9), ambient_lup);
    south.write(output, pixel, pixels, 6, static_cast<T>(9), ambient_lup);
    west.write(output, pixel, pixels, 9, static_cast<T>(9), ambient_lup);
    north.write(output, pixel, pixels, 12, static_cast<T>(9), ambient_lup);
    output[15 * pixels + pixel] = gvf_sum;
    output[16 * pixels + pixel] =
        building == zero ? one : A::divide(gvf_sum, static_cast<T>(18));
}

}

extern "C" __global__ void gvf(
    const Real* fields, const double* aspect, const int* offsets,
    const double* shade_bounds, int rows, int cols,
    int first_steps, int second_steps, Real albedo_b, Real* output) {
    gvf_pixel(fields, aspect, offsets, shade_bounds, rows, cols,
              first_steps, second_steps, albedo_b, output);
}

extern "C" __global__ void gvf_pack(
    const Real* fields, long long pixels, Real* neighbors, unsigned char* masks) {
    const long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    const GvfNeighborQuad<Real> value = {fields[pixels + p], fields[4 * pixels + p],
                                        fields[5 * pixels + p], fields[6 * pixels + p]};
    reinterpret_cast<GvfNeighborQuad<Real>*>(neighbors)[p] = value;
    masks[p] = static_cast<unsigned char>(fields[p]) |
               (static_cast<unsigned char>(fields[2 * pixels + p]) << 1);
}

extern "C" __global__ void gvf_compact(
    const Real* fields, const double* aspect, const int* offsets,
    const double* shade_bounds, int rows, int cols,
    int first_steps, int second_steps, Real albedo_b, Real* output,
    const Real* neighbors, const unsigned char* masks) {
    gvf_pixel<Real, true>(fields, aspect, offsets, shade_bounds, rows, cols,
                         first_steps, second_steps, albedo_b, output, neighbors, masks);
}
