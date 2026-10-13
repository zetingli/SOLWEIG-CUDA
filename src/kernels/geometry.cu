#if defined(SOLWEIG_PACK_VISIBILITY)
long long p = i % pixels;
long long k = i / pixels;
long long j = p * patches + k;
packed = ((sh[j] == 1 && veg[j] == 1) ? 1 : 0) |
         ((veg[j] == 0 || vb[j] == 0) ? 2 : 0) |
         ((sh[j] == 0 && vb[j] == 1) ? 4 : 0) |
         ((sh[j] == 0 || veg[j] == 0 || vb[j] == 0) ? 8 : 0);

#elif defined(SOLWEIG_TRANSPOSE_DIFFUSE)
target = source[(i % pixels) * patches + i / pixels];

#else

extern "C" __global__ void wall_heights(
    const double* dsm, double* walls, int rows, int cols, double threshold)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)rows * cols;
    if (p >= size) return;
    int row = p / cols;
    int col = p % cols;
    double height = 0.0;
    if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
        double neighbour = fmax(fmax(dsm[p - cols], dsm[p + cols]),
                                fmax(dsm[p - 1], dsm[p + 1]));
        double difference = neighbour - dsm[p];
        if (difference >= threshold) height = difference;
    } else if (row == 0 && rows > 1 && col >= rows && col < cols - 1) {
        double difference = -dsm[p];
        if (difference >= threshold) height = difference;
    }
    walls[p] = height;
}

extern "C" __global__ void wall_directions(
    const unsigned char* walls, const long long* pixels, long long count,
    const long long* offsets, const int* starts,
    int rows, int cols, int radius, short* angles)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long p = pixels[i];
    int row = p / cols;
    int col = p % cols;
    if (row < radius || row >= rows - radius - 2 ||
        col < radius || col >= cols - radius - 2) {
        angles[i] = -1;
        return;
    }
    int best_score = 0;
    short best_angle = -1;
    for (int angle = 0; angle < 180; ++angle) {
        int score = 0;
        for (int j = starts[angle]; j < starts[angle + 1]; ++j)
            score += walls[p + offsets[j]];
        if (score > best_score) {
            best_score = score;
            best_angle = (short)angle;
        }
    }
    angles[i] = best_angle;
}


__device__ __forceinline__ bool above(
    float sample, float local, float dz, const double* original,
    const double* terrain, long long q, long long pixel,
    const double* heights, long long step) {
    const float delta = __fsub_rn(__fsub_rn(sample, local), dz);
    const float bound = (fabsf(sample) + fabsf(local) + fabsf(dz)) * 4.76837158203125e-7f;
    if (delta > bound) return true;
    if (delta < -bound) return false;
    return __dsub_rn(original[q], heights[step]) > terrain[pixel];
}

extern "C" __global__ void svf_visibility(
    const float* a, const float* top, const float* bottom,
    const double* a64, const double* top64, const double* bottom64,
    const unsigned char* bushplant, int rows, int cols, int patches,
    int capacity, const int* offsets, const float* heights,
    const double* heights64, const int* lengths, int patch_start, int patch_end,
    float* shmat, float* vegmat, float* vbmat) {
    const long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int patch = patch_start + blockIdx.y;
    if (pixel >= (long long)rows * cols || patch >= patch_end) return;
    const int row = pixel / cols, col = pixel % cols;
    const float local = a[pixel];
    bool sh = false, veg = bushplant[pixel] != 0, ever = false;
    const long long base = (long long)patch * capacity;
    for (int n = 0; n < lengths[patch]; n++) {
        const long long slot = base + n, prev = base + (n > 0 ? n - 1 : 0);
        const int r = row + offsets[2 * slot], c = col + offsets[2 * slot + 1];
        const bool inside = r >= 0 && r < rows && c >= 0 && c < cols;
        if (inside) {
            const long long q = (long long)r * cols + c;
            const float dz = heights[slot], dzprev = heights[prev];
            sh = above(a[q], local, dz, a64, a64, q, pixel, heights64, slot);
            if (sh) { veg = false; break; }
            const float vt = top[q], vb = bottom[q];
            const int count =
                (int)above(vt, local, dz, top64, a64, q, pixel, heights64, slot) +
                (int)above(vb, local, dz, bottom64, a64, q, pixel, heights64, slot) +
                (int)above(vt, local, dzprev, top64, a64, q, pixel, heights64, prev) +
                (int)above(vb, local, dzprev, bottom64, a64, q, pixel, heights64, prev);
            veg = veg || (count > 0 && count < 4);
            ever = ever || veg;
        } else {
            sh = a64[pixel] < 0.0;
            if (sh) veg = false;
            ever = ever || veg;
            break;
        }
    }
    const long long out = pixel * patches + patch;
    shmat[out] = 1.0f - (float)sh;
    vegmat[out] = 1.0f - (float)veg;
    vbmat[out] = 1.0f - ((float)ever - (float)veg);
}

extern "C" __global__ void svf_reduce(
    const float* shmat, const float* vegmat, const float* vbmat,
    const double* bottom, const unsigned char* sectors, const int* band_offsets,
    const double* weights, long long pixels, int patches, int bands,
    double trans, double* output) {
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    double v[15] = {};
    for (int b = 0; b < bands; b++) {
        int counts[15] = {};
        for (int j = band_offsets[b]; j < band_offsets[b + 1]; j++) {
            const long long pos = p * patches + j;
            const int sh = (int)shmat[pos], veg = (int)vegmat[pos], vb = (int)vbmat[pos];
            const unsigned char mask = sectors[j];
            counts[0] += sh; counts[5] += veg; counts[10] += vb;
            #pragma unroll
            for (int d = 0; d < 4; d++) if (mask & (1 << d)) {
                counts[1 + d] += sh; counts[6 + d] += veg; counts[11 + d] += vb;
            }
        }
        const double wf = weights[2 * b], wa = weights[2 * b + 1];
        #pragma unroll
        for (int group = 0; group < 3; group++) {
            v[group * 5] += wf * counts[group * 5];
            #pragma unroll
            for (int d = 1; d < 5; d++) v[group * 5 + d] += wa * counts[group * 5 + d];
        }
    }
    const double eps = 3.0459e-4;
    v[2] += eps; v[3] += eps;
    const double last = bottom[p] == 0.0 ? eps : 0.0;
    v[7] += last; v[8] += last; v[12] += last; v[13] += last;
    #pragma unroll
    for (int i = 0; i < 15; i++) {
        if (v[i] > 1.0) v[i] = 1.0;
        output[i * pixels + p] = v[i];
    }
    output[15 * pixels + p] = v[0] - (1.0 - v[5]) * (1.0 - trans);
}

extern "C" __global__
void compute_diffsh_kernel(
    const float* __restrict__ shmat,
    const float* __restrict__ vegshmat,
    float* __restrict__ diffsh,
    int n,
    float transmissivity
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    diffsh[i] = shmat[i] - (1.0f - vegshmat[i]) * (1.0f - transmissivity);
}


__device__ __forceinline__ bool above_receiver(
    float value, float receiver, float tolerance,
    const double* fields, long long q, const double* heights, int h,
    double exact_receiver) {
    float margin = value - receiver;
    if (margin > tolerance) return true;
    if (margin < -tolerance) return false;
    return fields[q] - heights[h] > exact_receiver;
}

extern "C" __global__ void shadow_pixels(
    const double* fields, const float* scan, const int* offsets,
    const double* heights, const float* fast_heights, int steps,
    int rows, int cols, double low, double high, int middle, int vegetation,
    double* out) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * cols;
    if (p >= n) return;
    int row = p/cols, col = p%cols;
    double a=fields[p], wall=fields[n+p], aspect=fields[2*n+p];
    double f=a, topmax=fields[3*n+p];
    float receiver=scan[p], fast_f=receiver, fast_top=scan[n+p];
    bool veg=fields[5*n+p]>1., ever=false, building_shadow=false;
    for (int k=0;k<steps;++k) {
        int r=row+offsets[2*k], c=col+offsets[2*k+1];
        bool inside=r>=0 && r<rows && c>=0 && c<cols;
        long long q=(long long)r*cols+c;
        float dz=fast_heights[3*k], previous=fast_heights[3*k+1];
        float tolerance=fast_heights[3*k+2];
        float building=inside?scan[q]-dz:0.f;
        if (!(building < fast_f-tolerance)) {
            double exact=inside?fields[q]-heights[2*k]:0.;
            f=fmax(f,exact);
            fast_f=(float)f;
            building_shadow=f>a;
        }
        if (vegetation) {
            if (building_shadow) {veg=false; continue;}
            float top=inside?scan[n+q]-dz:0.f;
            if (!(top < fast_top-tolerance)) {
                double exact=inside?fields[3*n+q]-heights[2*k]:0.;
                topmax=fmax(topmax,exact);
                fast_top=(float)topmax;
            }
            int above;
            if (inside) {
                float bottom=scan[2*n+q]-dz;
                float prevtop=scan[n+q]-previous;
                float prevbottom=scan[2*n+q]-previous;
                above=above_receiver(top,receiver,tolerance,fields,3*n+q,heights,2*k,a)
                     +above_receiver(bottom,receiver,tolerance,fields,4*n+q,heights,2*k,a)
                     +above_receiver(prevtop,receiver,tolerance,fields,3*n+q,heights,2*k+1,a)
                     +above_receiver(prevbottom,receiver,tolerance,fields,4*n+q,heights,2*k+1,a);
            } else {
                above=(0.>a)?4:0;
            }
            veg=veg || (above>0 && above<4);
            ever=ever || veg;
        }
    }
    double wallbol=wall>0;
    double facesh=middle?double(aspect<low || aspect>=high)-wallbol+1.:
                          1.-double(aspect>low || aspect<=high);
    double volume=f-a;
    double wallsun=fmax(wall-volume,0.);
    if (facesh==1.) wallsun=0.;
    double wallsh=wall-wallsun, wallshve=0.;
    if (vegetation) {
        wallshve=(topmax-a)*double(veg)*wallbol-wallsh;
        wallshve=fmin(fmax(wallshve,0.),wall);
        wallsun=wallsun-wallshve;
        if (wallsun<0.) {wallshve=0.; wallsun=0.;}
    }
    out[p]=1.-double(veg);
    out[n+p]=vegetation?1.-double(building_shadow):1.-double(volume!=0.);
    out[2*n+p]=1.-(double(ever)-double(veg));
    out[3*n+p]=wallsh; out[4*n+p]=wallsun; out[5*n+p]=wallshve;
    out[6*n+p]=facesh; out[7*n+p]=(facesh+wallbol==1.) && wall>0.;
}

#endif
