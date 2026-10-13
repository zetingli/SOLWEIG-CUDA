using Real = SOLWEIG_REAL;

constexpr double LW_PI = 3.141592653589793238462643383279502884;
constexpr double LW_SBC = 5.67051e-8;

extern "C" __global__ void longwave_coefficients(
    const double* coefficients, int patches, Real* radiation,
    unsigned char* conditions) {
    const int patch = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch >= patches) return;
    const double* c = coefficients + static_cast<long long>(patch) * 17;
    Real* r = radiation + static_cast<long long>(patch) * 12;
    #pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = static_cast<Real>(c[j]);
    #pragma unroll
    for (int j = 0; j < 4; ++j) r[8 + j] = static_cast<Real>(c[11 + j]);

    unsigned char mask = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        if (c[4 + j] != 0.0) mask |= (1u << j);
    if (c[10] != 0.0) mask |= 16u;
    conditions[patch] = mask;
}

__device__ __forceinline__ int longwave_sunshade(double x, double altitude,
                                         double lower, double upper) {
    if (x < lower) return -1;
    if (x > upper) return 1;
    const double limit = __dmul_rn(atan(x), 180.0 / LW_PI);
    return (int)(limit > altitude) - (int)(limit < altitude);
}

template <typename T> struct LongwaveArithmetic;

template <> struct LongwaveArithmetic<float> {
    __device__ __forceinline__ static float add(float a, float b) {
        return __fadd_rn(a, b);
    }
    __device__ __forceinline__ static float multiply(float a, float b) {
        return __fmul_rn(a, b);
    }
    __device__ __forceinline__ static float divide(float a, float b) {
        return __fdiv_rn(a, b);
    }
    __device__ __forceinline__ static float fourth_power(float a) {
        return powf(a, 4.0f);
    }
};

template <> struct LongwaveArithmetic<double> {
    __device__ __forceinline__ static double add(double a, double b) {
        return __dadd_rn(a, b);
    }
    __device__ __forceinline__ static double multiply(double a, double b) {
        return __dmul_rn(a, b);
    }
    __device__ __forceinline__ static double divide(double a, double b) {
        return __ddiv_rn(a, b);
    }
    __device__ __forceinline__ static double fourth_power(double a) {
        return pow(a, 4.0);
    }
};

template <typename T>
__device__ __forceinline__ T longwave_surface(T temperature_kelvin, T emissivity) {
    typedef LongwaveArithmetic<T> A;
    return A::divide(
        A::multiply(A::multiply(emissivity, static_cast<T>(LW_SBC)),
                    A::fourth_power(temperature_kelvin)),
        static_cast<T>(LW_PI));
}

template <typename T>
__device__ __forceinline__ void add_cardinal(
    T contribution, const T* radiation, unsigned char conditions,
    T& east, T& west, T& north, T& south) {
    typedef LongwaveArithmetic<T> A;
    if (conditions & 1u)
        east = A::add(east, A::multiply(contribution, radiation[4]));
    if (conditions & 2u)
        west = A::add(west, A::multiply(contribution, radiation[5]));
    if (conditions & 4u)
        north = A::add(north, A::multiply(contribution, radiation[6]));
    if (conditions & 8u)
        south = A::add(south, A::multiply(contribution, radiation[7]));
}

extern "C" __global__ void longwave(
    const unsigned char* visibility, const double* coefficients,
    const Real* radiation, const unsigned char* conditions,
    const Real* lup, const Real* tgwall, const double* asvf,
    Real air_temperature, Real wall_emissivity,
    long long pixels, int patches, Real* output) {
    typedef LongwaveArithmetic<Real> A;
    const long long pixel = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;

    const Real zero = static_cast<Real>(0);
    Real down_sky = zero, down_vegetation = zero, down_sun = zero, down_shade = zero;
    Real side_sky = zero, side_vegetation = zero, side_sun = zero, side_shade = zero;
    Real down_reflected = zero, side_reflected = zero;
    Real east = zero, west = zero, north = zero, south = zero;

    const Real ambient_kelvin = A::add(air_temperature, static_cast<Real>(273.15));
    const Real sunlit_kelvin = A::add(A::add(air_temperature, tgwall[pixel]),
                                  static_cast<Real>(273.15));
    const Real ambient_surface = longwave_surface(ambient_kelvin, wall_emissivity);
    const Real sunlit_surface = longwave_surface(sunlit_kelvin, wall_emissivity);
    const double horizon_tangent = tan(asvf[pixel]);

    for (int patch = 0; patch < patches; ++patch) {
        const double* c = coefficients + static_cast<long long>(patch) * 17;
        const Real* r = radiation + static_cast<long long>(patch) * 12;
        const unsigned char condition = conditions[patch];
        const unsigned char flags = visibility[static_cast<long long>(patch) * pixels + pixel];
        const Real down_factor = r[2];
        const Real side_factor = r[3];

        if (flags & 1u) {
            const Real sky_side = r[1];
            down_sky = A::add(down_sky, r[0]);
            side_sky = A::add(side_sky, sky_side);
            east = A::add(east, r[8]);
            west = A::add(west, r[9]);
            north = A::add(north, r[10]);
            south = A::add(south, r[11]);
        }
        if (flags & 2u) {
            const Real vegetation_side = A::multiply(ambient_surface, side_factor);
            side_vegetation = A::add(side_vegetation, vegetation_side);
            down_vegetation = A::add(down_vegetation,
                                     A::multiply(ambient_surface, down_factor));
            add_cardinal(vegetation_side, r, condition, east, west, north, south);
        }

        if (flags & 4u) {
            bool sunlit = false;
            bool shaded = true;
            if (condition & 16u) {
                const int classification = longwave_sunshade(
                    __dadd_rn(horizon_tangent, c[8]), c[9], c[15], c[16]);
                sunlit = classification < 0;
                shaded = classification > 0;
            }
            if (sunlit) {
                const Real wall_side = A::multiply(sunlit_surface, side_factor);
                side_sun = A::add(side_sun, wall_side);
                down_sun = A::add(down_sun, A::multiply(sunlit_surface, down_factor));
                add_cardinal(wall_side, r, condition, east, west, north, south);
            }
            if (shaded) {
                const Real wall_side = A::multiply(ambient_surface, side_factor);
                side_shade = A::add(side_shade, wall_side);
                down_shade = A::add(down_shade, A::multiply(ambient_surface, down_factor));
                add_cardinal(wall_side, r, condition, east, west, north, south);
            }
        }
    }

    const Real reflected_surface = A::divide(
        A::multiply(
            A::multiply(A::add(down_sky, lup[pixel]),
                        A::add(static_cast<Real>(1), -wall_emissivity)),
            static_cast<Real>(0.5)),
        static_cast<Real>(LW_PI));

    for (int patch = 0; patch < patches; ++patch) {
        const unsigned char flags = visibility[static_cast<long long>(patch) * pixels + pixel];
        if (!(flags & 8u)) continue;
        const Real* r = radiation + static_cast<long long>(patch) * 12;
        const Real reflected_side = A::multiply(reflected_surface, r[3]);
        side_reflected = A::add(side_reflected, reflected_side);
        down_reflected = A::add(down_reflected,
                               A::multiply(reflected_surface, r[2]));
        add_cardinal(reflected_side, r, conditions[patch], east, west, north, south);
    }

    output[pixel] = A::add(A::add(A::add(A::add(down_sky, down_vegetation),
                                        down_shade), down_sun), down_reflected);
    output[pixels + pixel] = A::add(A::add(A::add(A::add(side_sky, side_vegetation),
                                                 side_shade), side_sun), side_reflected);
    output[2 * pixels + pixel] = east;
    output[3 * pixels + pixel] = west;
    output[4 * pixels + pixel] = north;
    output[5 * pixels + pixel] = south;
}
