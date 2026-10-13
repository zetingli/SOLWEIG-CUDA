using Real = SOLWEIG_REAL;
constexpr double SW_PI = 3.141592653589793238462643383279502884;

extern "C" __global__ void shortwave_coefficients(
    const double* coefficients, int patches, Real* radiation, unsigned short* branch_flags) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= patches) return;
    const double* c = coefficients + static_cast<long long>(k) * 24;
    Real* r = radiation + static_cast<long long>(k) * 11;
    for (int j = 0; j < 7; ++j) r[j] = static_cast<Real>(c[j]);
    for (int j = 0; j < 4; ++j) r[7+j] = static_cast<Real>(c[18+j]);
    unsigned short bits = 0;
    for (int j = 0; j < 8; ++j) if (c[7+j] != 0) bits |= (1u << j);
    if (c[17] != 0) bits |= (1u << 8);
    branch_flags[k] = bits;
}

__device__ __forceinline__ int shortwave_sunshade(double x, double altitude,
                                         double lower, double upper) {
    if (x < lower) return -1;
    if (x > upper) return 1;
    const double limit = __dmul_rn(atan(x), 180.0 / SW_PI);
    return (int)(limit > altitude) - (int)(limit < altitude);
}

template <typename T> struct ShortwavePower;
template <> struct ShortwavePower<double> {
    __device__ __forceinline__ static double value(double a, int n) {
        return pow(a, static_cast<double>(n));
    }
};
template <> struct ShortwavePower<float> {
    __device__ __forceinline__ static float value(float a, int n) {
        return powf(a, static_cast<float>(n));
    }
};

template <typename T>
__device__ __forceinline__ T weighting_polynomial(T x) {
    return static_cast<T>(63.227) * ShortwavePower<T>::value(x, 6)
         - static_cast<T>(161.51) * ShortwavePower<T>::value(x, 5)
         + static_cast<T>(156.91) * ShortwavePower<T>::value(x, 4)
         - static_cast<T>(70.424) * ShortwavePower<T>::value(x, 3)
         + static_cast<T>(16.773) * ShortwavePower<T>::value(x, 2)
         - static_cast<T>(0.4863) * x;
}

template <typename T>
__device__ __forceinline__ T surface_weight(T svf, T vegetation_svf, T psi) {
    const T total = static_cast<T>(4.4897);
    const T wall = (total - weighting_polynomial(svf)) / total;
    const T combined = vegetation_svf + svf - static_cast<T>(1);
    const T vegetation = (total - weighting_polynomial(combined)) / total - wall;
    return wall + vegetation * (static_cast<T>(1) - psi);
}

template <typename T> struct ShortwaveDirections {
    T east, south, west, north;
    __device__ __forceinline__ void reflection(T value, const T* r, unsigned short bits) {
        if (bits & (1u << 4)) east += value * r[3];
        if (bits & (1u << 5)) south += value * r[4];
        if (bits & (1u << 6)) west += value * r[5];
        if (bits & (1u << 7)) north += value * r[6];
    }
    __device__ __forceinline__ void diffuse(T visibility, const T* r, unsigned short bits) {
        if (bits & 1u)
            east += visibility * r[2] * r[7] * r[0];
        if (bits & 2u)
            south += visibility * r[2] * r[8] * r[0];
        if (bits & 4u)
            west += visibility * r[2] * r[9] * r[0];
        if (bits & 8u)
            north += visibility * r[2] * r[10] * r[0];
    }
};

template <typename T>
__device__ __forceinline__ T isotropic_direction(
    T direct, T svf, T svfveg, T kup, T f_sh, const double* parameters) {
    const T radD = static_cast<T>(parameters[1]);
    const T radG = static_cast<T>(parameters[2]);
    const T psi = static_cast<T>(parameters[4]);
    const T albedo = static_cast<T>(parameters[5]);
    const T weight = surface_weight(svf, svfveg, psi);
    const T diffuse_ground = (
        radD * (static_cast<T>(1) - weight) +
        albedo * (weight * (radG * (static_cast<T>(1) - f_sh) + radD * f_sh)) +
        kup) * static_cast<T>(0.5);
    return direct + diffuse_ground;
}

template <typename T>
__device__ __forceinline__ T box_total(
    T direct, T diffuse, T sun, T shade, T vegetation, T kup) {
    return direct + diffuse + sun + shade + vegetation + kup * static_cast<T>(0.5);
}

extern "C" __global__ void diffuse(
    const Real* vis, const double* lum, long long n, int patches, double radD, Real* out
) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    Real sum = 0;
    for (int k = 0; k < patches; ++k)
        sum = sum + (Real)(vis[(long long)k * n + p] * lum[k]);
    out[p] = (Real)(sum * radD);
}

extern "C" __global__ void shortwave(
    const unsigned char* visibility, const Real* diffsh, const Real* fields,
    const double* asvf, const double* coefficients, const double* parameters,
    const Real* radiation, const unsigned short* branch_flags,
    long long pixels, int patches, int cyl, int anisotropic, Real* output) {
    const long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    const Real zero = static_cast<Real>(0);
    const Real half = static_cast<Real>(0.5);
    const Real shadow = fields[p];
    const Real radI = static_cast<Real>(parameters[0]);
    const Real radD = static_cast<Real>(parameters[1]);
    const Real cos_sun_altitude = static_cast<Real>(parameters[3]);
    const Real albedo = static_cast<Real>(parameters[5]);
    const int kup = anisotropic ? 1 : 2;
    const Real kupE = fields[kup * pixels + p], kupS = fields[(kup+1) * pixels + p];
    const Real kupW = fields[(kup+2) * pixels + p], kupN = fields[(kup+3) * pixels + p];
    const Real side_direct = cyl ? shadow * radI * cos_sun_altitude : shadow * zero;
    ShortwaveDirections<Real> direct = {zero, zero, zero, zero};
    if (!cyl) {
        direct.east = radI * shadow * cos_sun_altitude * static_cast<Real>(parameters[6]);
        direct.south = radI * shadow * cos_sun_altitude * static_cast<Real>(parameters[7]);
        direct.west = radI * shadow * cos_sun_altitude * static_cast<Real>(parameters[8]);
        direct.north = radI * shadow * cos_sun_altitude * static_cast<Real>(parameters[9]);
    }

    Real east, south, west, north;
    Real side_diffuse = zero, side_total = zero;
    if (anisotropic) {
        const Real sunlit_surface =
            (albedo * (radI * cos_sun_altitude) + radD * half) / static_cast<Real>(SW_PI);
        const Real shaded_surface = (albedo * radD * half) / static_cast<Real>(SW_PI);
        const double tangent = tan(asvf[p]);
        Real ref_sun = zero, ref_shade = zero, ref_vegetation = zero;
        ShortwaveDirections<Real> diffuse = {zero, zero, zero, zero};
        ShortwaveDirections<Real> sun = {zero, zero, zero, zero};
        ShortwaveDirections<Real> shade = {zero, zero, zero, zero};
        ShortwaveDirections<Real> vegetation = {zero, zero, zero, zero};

        for (int k = 0; k < patches; ++k) {
            const double* c = coefficients + static_cast<long long>(k) * 24;
            const Real* r = radiation + static_cast<long long>(k) * 11;
            const unsigned short bits = branch_flags[k];
            const long long index = static_cast<long long>(k) * pixels + p;
            const unsigned char flags = visibility[index];
            const Real omega = r[0];
            const Real cos_altitude = r[1];
            if (cyl)
                side_diffuse += diffsh[index] * r[2] * cos_altitude * omega;
            else
                diffuse.diffuse(diffsh[index], r, bits);

            if (flags & 2u) {
                const Real reflected = shaded_surface * omega * cos_altitude;
                if (cyl) ref_vegetation += reflected;
                else vegetation.reflection(reflected, r, bits);
            }
            if (flags & 4u) {
                bool sunlit = false, shaded = true;
                if (cyl || (bits & (1u << 8))) {
                    const int classification = shortwave_sunshade(
                        __dadd_rn(tangent, c[15]), c[16], c[22], c[23]);
                    sunlit = classification < 0;
                    shaded = classification > 0;
                }
                if (sunlit) {
                    const Real reflected = sunlit_surface * omega * cos_altitude;
                    if (cyl) ref_sun += reflected;
                    else sun.reflection(reflected, r, bits);
                }
                if (shaded) {
                    const Real reflected = shaded_surface * omega * cos_altitude;
                    if (cyl) ref_shade += reflected;
                    else shade.reflection(reflected, r, bits);
                }
            }
        }
        if (cyl) {
            side_total = side_direct + side_diffuse + ref_sun + ref_shade + ref_vegetation;
            east = kupE * half; south = kupS * half;
            west = kupW * half; north = kupN * half;
        } else {
            east = box_total(direct.east, diffuse.east, sun.east, shade.east, vegetation.east, kupE);
            south = box_total(direct.south, diffuse.south, sun.south, shade.south, vegetation.south, kupS);
            west = box_total(direct.west, diffuse.west, sun.west, shade.west, vegetation.west, kupW);
            north = box_total(direct.north, diffuse.north, sun.north, shade.north, vegetation.north, kupN);
        }
    } else {
        const Real f_sh = fields[pixels + p];
        east = isotropic_direction(direct.east, fields[6 * pixels + p], fields[10 * pixels + p], kupE, f_sh, parameters);
        south = isotropic_direction(direct.south, fields[7 * pixels + p], fields[11 * pixels + p], kupS, f_sh, parameters);
        west = isotropic_direction(direct.west, fields[8 * pixels + p], fields[12 * pixels + p], kupW, f_sh, parameters);
        north = isotropic_direction(direct.north, fields[9 * pixels + p], fields[13 * pixels + p], kupN, f_sh, parameters);
    }
    output[p] = east;
    output[pixels + p] = south;
    output[2 * pixels + p] = west;
    output[3 * pixels + p] = north;
    output[4 * pixels + p] = side_direct;
    output[5 * pixels + p] = side_diffuse;
    output[6 * pixels + p] = side_total;
}
