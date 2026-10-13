extern "C" __global__ void utci_fused_kernel(
    const float* ta, const float* rh, const float* tmrt, const float* wind,
    float* output, const long long n)
{
    #pragma unroll 1
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float Ta = ta[i], RH = rh[i], Tmrt = tmrt[i], va10m = wind[i];
        float out;
        if (Ta <= -999.0f || RH <= -999.0f || va10m <= -999.0f || Tmrt <= -999.0f) {
            out = -999.0f;
        } else {
            float tk = Ta + 273.15f;
            float es = 2.7150305f * logf(tk);
            es += -2.8365744E3f * powf(tk, -2.0f);
            es += -6.028076559E3f * powf(tk, -1.0f);
            es += 1.954263612E1f;
            es += -2.737830188E-2f * tk;
            es += 1.6261698E-5f * powf(tk, 2.0f);
            es += 7.0229056E-10f * powf(tk, 3.0f);
            es += -1.8680009E-13f * powf(tk, 4.0f);
            es = expf(es) * 0.01f;
            float ehPa = es * RH / 100.0f;
            float D_Tmrt = Tmrt - Ta;
            float Pa = ehPa / 10.0f;
            float va = va10m;
            out = Ta +
                (6.07562052E-01) +
                (-2.27712343E-02) * Ta +
                (8.06470249E-04) * powf(Ta, 2.0f) +
                (-1.54271372E-04) * powf(Ta, 3.0f) +
                (-3.24651735E-06) * powf(Ta, 4.0f) +
                (7.32602852E-08) * powf(Ta, 5.0f) +
                (1.35959073E-09) * powf(Ta, 6.0f) +
                (-2.25836520E+00) * va +
                (8.80326035E-02) * Ta * va +
                (2.16844454E-03) * powf(Ta, 2.0f) * va +
                (-1.53347087E-05) * powf(Ta, 3.0f) * va +
                (-5.72983704E-07) * powf(Ta, 4.0f) * va +
                (-2.55090145E-09) * powf(Ta, 5.0f) * va +
                (-7.51269505E-01) * powf(va, 2.0f) +
                (-4.08350271E-03) * Ta * powf(va, 2.0f) +
                (-5.21670675E-05) * powf(Ta, 2.0f) * powf(va, 2.0f) +
                (1.94544667E-06) * powf(Ta, 3.0f) * powf(va, 2.0f) +
                (1.14099531E-08) * powf(Ta, 4.0f) * powf(va, 2.0f) +
                (1.58137256E-01) * powf(va, 3.0f) +
                (-6.57263143E-05) * Ta * powf(va, 3.0f) +
                (2.22697524E-07) * powf(Ta, 2.0f) * powf(va, 3.0f) +
                (-4.16117031E-08) * powf(Ta, 3.0f) * powf(va, 3.0f) +
                (-1.27762753E-02) * powf(va, 4.0f) +
                (9.66891875E-06) * Ta * powf(va, 4.0f) +
                (2.52785852E-09) * powf(Ta, 2.0f) * powf(va, 4.0f) +
                (4.56306672E-04) * powf(va, 5.0f) +
                (-1.74202546E-07) * Ta * powf(va, 5.0f) +
                (-5.91491269E-06) * powf(va, 6.0f) +
                (3.98374029E-01) * D_Tmrt +
                (1.83945314E-04) * Ta * D_Tmrt +
                (-1.73754510E-04) * powf(Ta, 2.0f) * D_Tmrt +
                (-7.60781159E-07) * powf(Ta, 3.0f) * D_Tmrt +
                (3.77830287E-08) * powf(Ta, 4.0f) * D_Tmrt +
                (5.43079673E-10) * powf(Ta, 5.0f) * D_Tmrt +
                (-2.00518269E-02) * va * D_Tmrt +
                (8.92859837E-04) * Ta * va * D_Tmrt +
                (3.45433048E-06) * powf(Ta, 2.0f) * va * D_Tmrt +
                (-3.77925774E-07) * powf(Ta, 3.0f) * va * D_Tmrt +
                (-1.69699377E-09) * powf(Ta, 4.0f) * va * D_Tmrt +
                (1.69992415E-04) * powf(va, 2.0f) * D_Tmrt +
                (-4.99204314E-05) * Ta * powf(va, 2.0f) * D_Tmrt +
                (2.47417178E-07) * powf(Ta, 2.0f) * powf(va, 2.0f) * D_Tmrt +
                (1.07596466E-08) * powf(Ta, 3.0f) * powf(va, 2.0f) * D_Tmrt +
                (8.49242932E-05) * powf(va, 3.0f) * D_Tmrt +
                (1.35191328E-06) * Ta * powf(va, 3.0f) * D_Tmrt +
                (-6.21531254E-09) * powf(Ta, 2.0f) * powf(va, 3.0f) * D_Tmrt +
                (-4.99410301E-06) * powf(va, 4.0f) * D_Tmrt +
                (-1.89489258E-08) * Ta * powf(va, 4.0f) * D_Tmrt +
                (8.15300114E-08) * powf(va, 5.0f) * D_Tmrt +
                (7.55043090E-04) * powf(D_Tmrt, 2.0f) +
                (-5.65095215E-05) * Ta * powf(D_Tmrt, 2.0f) +
                (-4.52166564E-07) * powf(Ta, 2.0f) * powf(D_Tmrt, 2.0f) +
                (2.46688878E-08) * powf(Ta, 3.0f) * powf(D_Tmrt, 2.0f) +
                (2.42674348E-10) * powf(Ta, 4.0f) * powf(D_Tmrt, 2.0f) +
                (1.54547250E-04) * va * powf(D_Tmrt, 2.0f) +
                (5.24110970E-06) * Ta * va * powf(D_Tmrt, 2.0f) +
                (-8.75874982E-08) * powf(Ta, 2.0f) * va * powf(D_Tmrt, 2.0f) +
                (-1.50743064E-09) * powf(Ta, 3.0f) * va * powf(D_Tmrt, 2.0f) +
                (-1.56236307E-05) * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) +
                (-1.33895614E-07) * Ta * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) +
                (2.49709824E-09) * powf(Ta, 2.0f) * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) +
                (6.51711721E-07) * powf(va, 3.0f) * powf(D_Tmrt, 2.0f) +
                (1.94960053E-09) * Ta * powf(va, 3.0f) * powf(D_Tmrt, 2.0f) +
                (-1.00361113E-08) * powf(va, 4.0f) * powf(D_Tmrt, 2.0f) +
                (-1.21206673E-05) * powf(D_Tmrt, 3.0f) +
                (-2.18203660E-07) * Ta * powf(D_Tmrt, 3.0f) +
                (7.51269482E-09) * powf(Ta, 2.0f) * powf(D_Tmrt, 3.0f) +
                (9.79063848E-11) * powf(Ta, 3.0f) * powf(D_Tmrt, 3.0f) +
                (1.25006734E-06) * va * powf(D_Tmrt, 3.0f) +
                (-1.81584736E-09) * Ta * va * powf(D_Tmrt, 3.0f) +
                (-3.52197671E-10) * powf(Ta, 2.0f) * va * powf(D_Tmrt, 3.0f) +
                (-3.36514630E-08) * powf(va, 2.0f) * powf(D_Tmrt, 3.0f) +
                (1.35908359E-10) * Ta * powf(va, 2.0f) * powf(D_Tmrt, 3.0f) +
                (4.17032620E-10) * powf(va, 3.0f) * powf(D_Tmrt, 3.0f) +
                (-1.30369025E-09) * powf(D_Tmrt, 4.0f) +
                (4.13908461E-10) * Ta * powf(D_Tmrt, 4.0f) +
                (9.22652254E-12) * powf(Ta, 2.0f) * powf(D_Tmrt, 4.0f) +
                (-5.08220384E-09) * va * powf(D_Tmrt, 4.0f) +
                (-2.24730961E-11) * Ta * va * powf(D_Tmrt, 4.0f) +
                (1.17139133E-10) * powf(va, 2.0f) * powf(D_Tmrt, 4.0f) +
                (6.62154879E-10) * powf(D_Tmrt, 5.0f) +
                (4.03863260E-13) * Ta * powf(D_Tmrt, 5.0f) +
                (1.95087203E-12) * va * powf(D_Tmrt, 5.0f) +
                (-4.73602469E-12) * powf(D_Tmrt, 6.0f) +
                (5.12733497E+00) * Pa +
                (-3.12788561E-01) * Ta * Pa +
                (-1.96701861E-02) * powf(Ta, 2.0f) * Pa +
                (9.99690870E-04) * powf(Ta, 3.0f) * Pa +
                (9.51738512E-06) * powf(Ta, 4.0f) * Pa +
                (-4.66426341E-07) * powf(Ta, 5.0f) * Pa +
                (5.48050612E-01) * va * Pa +
                (-3.30552823E-03) * Ta * va * Pa +
                (-1.64119440E-03) * powf(Ta, 2.0f) * va * Pa +
                (-5.16670694E-06) * powf(Ta, 3.0f) * va * Pa +
                (9.52692432E-07) * powf(Ta, 4.0f) * va * Pa +
                (-4.29223622E-02) * powf(va, 2.0f) * Pa +
                (5.00845667E-03) * Ta * powf(va, 2.0f) * Pa +
                (1.00601257E-06) * powf(Ta, 2.0f) * powf(va, 2.0f) * Pa +
                (-1.81748644E-06) * powf(Ta, 3.0f) * powf(va, 2.0f) * Pa +
                (-1.25813502E-03) * powf(va, 3.0f) * Pa +
                (-1.79330391E-04) * Ta * powf(va, 3.0f) * Pa +
                (2.34994441E-06) * powf(Ta, 2.0f) * powf(va, 3.0f) * Pa +
                (1.29735808E-04) * powf(va, 4.0f) * Pa +
                (1.29064870E-06) * Ta * powf(va, 4.0f) * Pa +
                (-2.28558686E-06) * powf(va, 5.0f) * Pa +
                (-3.69476348E-02) * D_Tmrt * Pa +
                (1.62325322E-03) * Ta * D_Tmrt * Pa +
                (-3.14279680E-05) * powf(Ta, 2.0f) * D_Tmrt * Pa +
                (2.59835559E-06) * powf(Ta, 3.0f) * D_Tmrt * Pa +
                (-4.77136523E-08) * powf(Ta, 4.0f) * D_Tmrt * Pa +
                (8.64203390E-03) * va * D_Tmrt * Pa +
                (-6.87405181E-04) * Ta * va * D_Tmrt * Pa +
                (-9.13863872E-06) * powf(Ta, 2.0f) * va * D_Tmrt * Pa +
                (5.15916806E-07) * powf(Ta, 3.0f) * va * D_Tmrt * Pa +
                (-3.59217476E-05) * powf(va, 2.0f) * D_Tmrt * Pa +
                (3.28696511E-05) * Ta * powf(va, 2.0f) * D_Tmrt * Pa +
                (-7.10542454E-07) * powf(Ta, 2.0f) * powf(va, 2.0f) * D_Tmrt * Pa +
                (-1.24382300E-05) * powf(va, 3.0f) * D_Tmrt * Pa +
                (-7.38584400E-09) * Ta * powf(va, 3.0f) * D_Tmrt * Pa +
                (2.20609296E-07) * powf(va, 4.0f) * D_Tmrt * Pa +
                (-7.32469180E-04) * powf(D_Tmrt, 2.0f) * Pa +
                (-1.87381964E-05) * Ta * powf(D_Tmrt, 2.0f) * Pa +
                (4.80925239E-06) * powf(Ta, 2.0f) * powf(D_Tmrt, 2.0f) * Pa +
                (-8.75492040E-08) * powf(Ta, 3.0f) * powf(D_Tmrt, 2.0f) * Pa +
                (2.77862930E-05) * va * powf(D_Tmrt, 2.0f) * Pa +
                (-5.06004592E-06) * Ta * va * powf(D_Tmrt, 2.0f) * Pa +
                (1.14325367E-07) * powf(Ta, 2.0f) * va * powf(D_Tmrt, 2.0f) * Pa +
                (2.53016723E-06) * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) * Pa +
                (-1.72857035E-08) * Ta * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) * Pa +
                (-3.95079398E-08) * powf(va, 3.0f) * powf(D_Tmrt, 2.0f) * Pa +
                (-3.59413173E-07) * powf(D_Tmrt, 3.0f) * Pa +
                (7.04388046E-07) * Ta * powf(D_Tmrt, 3.0f) * Pa +
                (-1.89309167E-08) * powf(Ta, 2.0f) * powf(D_Tmrt, 3.0f) * Pa +
                (-4.79768731E-07) * va * powf(D_Tmrt, 3.0f) * Pa +
                (7.96079978E-09) * Ta * va * powf(D_Tmrt, 3.0f) * Pa +
                (1.62897058E-09) * powf(va, 2.0f) * powf(D_Tmrt, 3.0f) * Pa +
                (3.94367674E-08) * powf(D_Tmrt, 4.0f) * Pa +
                (-1.18566247E-09) * Ta * powf(D_Tmrt, 4.0f) * Pa +
                (3.34678041E-10) * va * powf(D_Tmrt, 4.0f) * Pa +
                (-1.15606447E-10) * powf(D_Tmrt, 5.0f) * Pa +
                (-2.80626406E+00) * powf(Pa, 2.0f) +
                (5.48712484E-01) * Ta * powf(Pa, 2.0f) +
                (-3.99428410E-03) * powf(Ta, 2.0f) * powf(Pa, 2.0f) +
                (-9.54009191E-04) * powf(Ta, 3.0f) * powf(Pa, 2.0f) +
                (1.93090978E-05) * powf(Ta, 4.0f) * powf(Pa, 2.0f) +
                (-3.08806365E-01) * va * powf(Pa, 2.0f) +
                (1.16952364E-02) * Ta * va * powf(Pa, 2.0f) +
                (4.95271903E-04) * powf(Ta, 2.0f) * va * powf(Pa, 2.0f) +
                (-1.90710882E-05) * powf(Ta, 3.0f) * va * powf(Pa, 2.0f) +
                (2.10787756E-03) * powf(va, 2.0f) * powf(Pa, 2.0f) +
                (-6.98445738E-04) * Ta * powf(va, 2.0f) * powf(Pa, 2.0f) +
                (2.30109073E-05) * powf(Ta, 2.0f) * powf(va, 2.0f) * powf(Pa, 2.0f) +
                (4.17856590E-04) * powf(va, 3.0f) * powf(Pa, 2.0f) +
                (-1.27043871E-05) * Ta * powf(va, 3.0f) * powf(Pa, 2.0f) +
                (-3.04620472E-06) * powf(va, 4.0f) * powf(Pa, 2.0f) +
                (5.14507424E-02) * D_Tmrt * powf(Pa, 2.0f) +
                (-4.32510997E-03) * Ta * D_Tmrt * powf(Pa, 2.0f) +
                (8.99281156E-05) * powf(Ta, 2.0f) * D_Tmrt * powf(Pa, 2.0f) +
                (-7.14663943E-07) * powf(Ta, 3.0f) * D_Tmrt * powf(Pa, 2.0f) +
                (-2.66016305E-04) * va * D_Tmrt * powf(Pa, 2.0f) +
                (2.63789586E-04) * Ta * va * D_Tmrt * powf(Pa, 2.0f) +
                (-7.01199003E-06) * powf(Ta, 2.0f) * va * D_Tmrt * powf(Pa, 2.0f) +
                (-1.06823306E-04) * powf(va, 2.0f) * D_Tmrt * powf(Pa, 2.0f) +
                (3.61341136E-06) * Ta * powf(va, 2.0f) * D_Tmrt * powf(Pa, 2.0f) +
                (2.29748967E-07) * powf(va, 3.0f) * D_Tmrt * powf(Pa, 2.0f) +
                (3.04788893E-04) * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (-6.42070836E-05) * Ta * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (1.16257971E-06) * powf(Ta, 2.0f) * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (7.68023384E-06) * va * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (-5.47446896E-07) * Ta * va * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (-3.59937910E-08) * powf(va, 2.0f) * powf(D_Tmrt, 2.0f) * powf(Pa, 2.0f) +
                (-4.36497725E-06) * powf(D_Tmrt, 3.0f) * powf(Pa, 2.0f) +
                (1.68737969E-07) * Ta * powf(D_Tmrt, 3.0f) * powf(Pa, 2.0f) +
                (2.67489271E-08) * va * powf(D_Tmrt, 3.0f) * powf(Pa, 2.0f) +
                (3.23926897E-09) * powf(D_Tmrt, 4.0f) * powf(Pa, 2.0f) +
                (-3.53874123E-02) * powf(Pa, 3.0f) +
                (-2.21201190E-01) * Ta * powf(Pa, 3.0f) +
                (1.55126038E-02) * powf(Ta, 2.0f) * powf(Pa, 3.0f) +
                (-2.63917279E-04) * powf(Ta, 3.0f) * powf(Pa, 3.0f) +
                (4.53433455E-02) * va * powf(Pa, 3.0f) +
                (-4.32943862E-03) * Ta * va * powf(Pa, 3.0f) +
                (1.45389826E-04) * powf(Ta, 2.0f) * va * powf(Pa, 3.0f) +
                (2.17508610E-04) * powf(va, 2.0f) * powf(Pa, 3.0f) +
                (-6.66724702E-05) * Ta * powf(va, 2.0f) * powf(Pa, 3.0f) +
                (3.33217140E-05) * powf(va, 3.0f) * powf(Pa, 3.0f) +
                (-2.26921615E-03) * D_Tmrt * powf(Pa, 3.0f) +
                (3.80261982E-04) * Ta * D_Tmrt * powf(Pa, 3.0f) +
                (-5.45314314E-09) * powf(Ta, 2.0f) * D_Tmrt * powf(Pa, 3.0f) +
                (-7.96355448E-04) * va * D_Tmrt * powf(Pa, 3.0f) +
                (2.53458034E-05) * Ta * va * D_Tmrt * powf(Pa, 3.0f) +
                (-6.31223658E-06) * powf(va, 2.0f) * D_Tmrt * powf(Pa, 3.0f) +
                (3.02122035E-04) * powf(D_Tmrt, 2.0f) * powf(Pa, 3.0f) +
                (-4.77403547E-06) * Ta * powf(D_Tmrt, 2.0f) * powf(Pa, 3.0f) +
                (1.73825715E-06) * va * powf(D_Tmrt, 2.0f) * powf(Pa, 3.0f) +
                (-4.09087898E-07) * powf(D_Tmrt, 3.0f) * powf(Pa, 3.0f) +
                (6.14155345E-01) * powf(Pa, 4.0f) +
                (-6.16755931E-02) * Ta * powf(Pa, 4.0f) +
                (1.33374846E-03) * powf(Ta, 2.0f) * powf(Pa, 4.0f) +
                (3.55375387E-03) * va * powf(Pa, 4.0f) +
                (-5.13027851E-04) * Ta * va * powf(Pa, 4.0f) +
                (1.02449757E-04) * powf(va, 2.0f) * powf(Pa, 4.0f) +
                (-1.48526421E-03) * D_Tmrt * powf(Pa, 4.0f) +
                (-4.11469183E-05) * Ta * D_Tmrt * powf(Pa, 4.0f) +
                (-6.80434415E-06) * va * D_Tmrt * powf(Pa, 4.0f) +
                (-9.77675906E-06) * powf(D_Tmrt, 2.0f) * powf(Pa, 4.0f) +
                (8.82773108E-02) * powf(Pa, 5.0f) +
                (-3.01859306E-03) * Ta * powf(Pa, 5.0f) +
                (1.04452989E-03) * va * powf(Pa, 5.0f) +
                (2.47090539E-04) * D_Tmrt * powf(Pa, 5.0f) +
                (1.48348065E-03) * powf(Pa, 6.0f) ;
        }
        output[i] = out;
    }
}

extern "C" __global__ void utci_scalar_weather_prepare(
    const float* ta, const float* rh, const float* wind, double* coefficients)
{
    const float Ta = ta[0], RH = rh[0], va10m = wind[0];
    if (!(Ta <= -999.0f || RH <= -999.0f || va10m <= -999.0f)) {

        float tk = Ta + 273.15f;
        float es = 2.7150305f * logf(tk);
        es += -2.8365744E3f * powf(tk, -2.0f);
        es += -6.028076559E3f * powf(tk, -1.0f);
        es += 1.954263612E1f;
        es += -2.737830188E-2f * tk;
        es += 1.6261698E-5f * powf(tk, 2.0f);
        es += 7.0229056E-10f * powf(tk, 3.0f);
        es += -1.8680009E-13f * powf(tk, 4.0f);
        es = expf(es) * 0.01f;
        float ehPa = es * RH / 100.0f;
        float Pa = ehPa / 10.0f;
        float va = va10m;

        const float Ta2 = powf(Ta, 2.0f);
        const float Ta3 = powf(Ta, 3.0f);
        const float Ta4 = powf(Ta, 4.0f);
        const float Ta5 = powf(Ta, 5.0f);
        const float Ta6 = powf(Ta, 6.0f);
        const float va2 = powf(va, 2.0f);
        const float va3 = powf(va, 3.0f);
        const float va4 = powf(va, 4.0f);
        const float va5 = powf(va, 5.0f);
        const float va6 = powf(va, 6.0f);
        const float Pa2 = powf(Pa, 2.0f);
        const float Pa3 = powf(Pa, 3.0f);
        const float Pa4 = powf(Pa, 4.0f);
        const float Pa5 = powf(Pa, 5.0f);
        const float Pa6 = powf(Pa, 6.0f);

        coefficients[0] =
            Ta +
            (6.07562052E-01) +
            (-2.27712343E-02) * Ta +
            (8.06470249E-04) * Ta2 +
            (-1.54271372E-04) * Ta3 +
            (-3.24651735E-06) * Ta4 +
            (7.32602852E-08) * Ta5 +
            (1.35959073E-09) * Ta6 +
            (-2.25836520E+00) * va +
            (8.80326035E-02) * Ta * va +
            (2.16844454E-03) * Ta2 * va +
            (-1.53347087E-05) * Ta3 * va +
            (-5.72983704E-07) * Ta4 * va +
            (-2.55090145E-09) * Ta5 * va +
            (-7.51269505E-01) * va2 +
            (-4.08350271E-03) * Ta * va2 +
            (-5.21670675E-05) * Ta2 * va2 +
            (1.94544667E-06) * Ta3 * va2 +
            (1.14099531E-08) * Ta4 * va2 +
            (1.58137256E-01) * va3 +
            (-6.57263143E-05) * Ta * va3 +
            (2.22697524E-07) * Ta2 * va3 +
            (-4.16117031E-08) * Ta3 * va3 +
            (-1.27762753E-02) * va4 +
            (9.66891875E-06) * Ta * va4 +
            (2.52785852E-09) * Ta2 * va4 +
            (4.56306672E-04) * va5 +
            (-1.74202546E-07) * Ta * va5 +
            (-5.91491269E-06) * va6 +
            (5.12733497E+00) * Pa +
            (-3.12788561E-01) * Ta * Pa +
            (-1.96701861E-02) * Ta2 * Pa +
            (9.99690870E-04) * Ta3 * Pa +
            (9.51738512E-06) * Ta4 * Pa +
            (-4.66426341E-07) * Ta5 * Pa +
            (5.48050612E-01) * va * Pa +
            (-3.30552823E-03) * Ta * va * Pa +
            (-1.64119440E-03) * Ta2 * va * Pa +
            (-5.16670694E-06) * Ta3 * va * Pa +
            (9.52692432E-07) * Ta4 * va * Pa +
            (-4.29223622E-02) * va2 * Pa +
            (5.00845667E-03) * Ta * va2 * Pa +
            (1.00601257E-06) * Ta2 * va2 * Pa +
            (-1.81748644E-06) * Ta3 * va2 * Pa +
            (-1.25813502E-03) * va3 * Pa +
            (-1.79330391E-04) * Ta * va3 * Pa +
            (2.34994441E-06) * Ta2 * va3 * Pa +
            (1.29735808E-04) * va4 * Pa +
            (1.29064870E-06) * Ta * va4 * Pa +
            (-2.28558686E-06) * va5 * Pa +
            (-2.80626406E+00) * Pa2 +
            (5.48712484E-01) * Ta * Pa2 +
            (-3.99428410E-03) * Ta2 * Pa2 +
            (-9.54009191E-04) * Ta3 * Pa2 +
            (1.93090978E-05) * Ta4 * Pa2 +
            (-3.08806365E-01) * va * Pa2 +
            (1.16952364E-02) * Ta * va * Pa2 +
            (4.95271903E-04) * Ta2 * va * Pa2 +
            (-1.90710882E-05) * Ta3 * va * Pa2 +
            (2.10787756E-03) * va2 * Pa2 +
            (-6.98445738E-04) * Ta * va2 * Pa2 +
            (2.30109073E-05) * Ta2 * va2 * Pa2 +
            (4.17856590E-04) * va3 * Pa2 +
            (-1.27043871E-05) * Ta * va3 * Pa2 +
            (-3.04620472E-06) * va4 * Pa2 +
            (-3.53874123E-02) * Pa3 +
            (-2.21201190E-01) * Ta * Pa3 +
            (1.55126038E-02) * Ta2 * Pa3 +
            (-2.63917279E-04) * Ta3 * Pa3 +
            (4.53433455E-02) * va * Pa3 +
            (-4.32943862E-03) * Ta * va * Pa3 +
            (1.45389826E-04) * Ta2 * va * Pa3 +
            (2.17508610E-04) * va2 * Pa3 +
            (-6.66724702E-05) * Ta * va2 * Pa3 +
            (3.33217140E-05) * va3 * Pa3 +
            (6.14155345E-01) * Pa4 +
            (-6.16755931E-02) * Ta * Pa4 +
            (1.33374846E-03) * Ta2 * Pa4 +
            (3.55375387E-03) * va * Pa4 +
            (-5.13027851E-04) * Ta * va * Pa4 +
            (1.02449757E-04) * va2 * Pa4 +
            (8.82773108E-02) * Pa5 +
            (-3.01859306E-03) * Ta * Pa5 +
            (1.04452989E-03) * va * Pa5 +
            (1.48348065E-03) * Pa6;
        coefficients[1] =
            (3.98374029E-01) +
            (1.83945314E-04) * Ta +
            (-1.73754510E-04) * Ta2 +
            (-7.60781159E-07) * Ta3 +
            (3.77830287E-08) * Ta4 +
            (5.43079673E-10) * Ta5 +
            (-2.00518269E-02) * va +
            (8.92859837E-04) * Ta * va +
            (3.45433048E-06) * Ta2 * va +
            (-3.77925774E-07) * Ta3 * va +
            (-1.69699377E-09) * Ta4 * va +
            (1.69992415E-04) * va2 +
            (-4.99204314E-05) * Ta * va2 +
            (2.47417178E-07) * Ta2 * va2 +
            (1.07596466E-08) * Ta3 * va2 +
            (8.49242932E-05) * va3 +
            (1.35191328E-06) * Ta * va3 +
            (-6.21531254E-09) * Ta2 * va3 +
            (-4.99410301E-06) * va4 +
            (-1.89489258E-08) * Ta * va4 +
            (8.15300114E-08) * va5 +
            (-3.69476348E-02) * Pa +
            (1.62325322E-03) * Ta * Pa +
            (-3.14279680E-05) * Ta2 * Pa +
            (2.59835559E-06) * Ta3 * Pa +
            (-4.77136523E-08) * Ta4 * Pa +
            (8.64203390E-03) * va * Pa +
            (-6.87405181E-04) * Ta * va * Pa +
            (-9.13863872E-06) * Ta2 * va * Pa +
            (5.15916806E-07) * Ta3 * va * Pa +
            (-3.59217476E-05) * va2 * Pa +
            (3.28696511E-05) * Ta * va2 * Pa +
            (-7.10542454E-07) * Ta2 * va2 * Pa +
            (-1.24382300E-05) * va3 * Pa +
            (-7.38584400E-09) * Ta * va3 * Pa +
            (2.20609296E-07) * va4 * Pa +
            (5.14507424E-02) * Pa2 +
            (-4.32510997E-03) * Ta * Pa2 +
            (8.99281156E-05) * Ta2 * Pa2 +
            (-7.14663943E-07) * Ta3 * Pa2 +
            (-2.66016305E-04) * va * Pa2 +
            (2.63789586E-04) * Ta * va * Pa2 +
            (-7.01199003E-06) * Ta2 * va * Pa2 +
            (-1.06823306E-04) * va2 * Pa2 +
            (3.61341136E-06) * Ta * va2 * Pa2 +
            (2.29748967E-07) * va3 * Pa2 +
            (-2.26921615E-03) * Pa3 +
            (3.80261982E-04) * Ta * Pa3 +
            (-5.45314314E-09) * Ta2 * Pa3 +
            (-7.96355448E-04) * va * Pa3 +
            (2.53458034E-05) * Ta * va * Pa3 +
            (-6.31223658E-06) * va2 * Pa3 +
            (-1.48526421E-03) * Pa4 +
            (-4.11469183E-05) * Ta * Pa4 +
            (-6.80434415E-06) * va * Pa4 +
            (2.47090539E-04) * Pa5;
        coefficients[2] =
            (7.55043090E-04) +
            (-5.65095215E-05) * Ta +
            (-4.52166564E-07) * Ta2 +
            (2.46688878E-08) * Ta3 +
            (2.42674348E-10) * Ta4 +
            (1.54547250E-04) * va +
            (5.24110970E-06) * Ta * va +
            (-8.75874982E-08) * Ta2 * va +
            (-1.50743064E-09) * Ta3 * va +
            (-1.56236307E-05) * va2 +
            (-1.33895614E-07) * Ta * va2 +
            (2.49709824E-09) * Ta2 * va2 +
            (6.51711721E-07) * va3 +
            (1.94960053E-09) * Ta * va3 +
            (-1.00361113E-08) * va4 +
            (-7.32469180E-04) * Pa +
            (-1.87381964E-05) * Ta * Pa +
            (4.80925239E-06) * Ta2 * Pa +
            (-8.75492040E-08) * Ta3 * Pa +
            (2.77862930E-05) * va * Pa +
            (-5.06004592E-06) * Ta * va * Pa +
            (1.14325367E-07) * Ta2 * va * Pa +
            (2.53016723E-06) * va2 * Pa +
            (-1.72857035E-08) * Ta * va2 * Pa +
            (-3.95079398E-08) * va3 * Pa +
            (3.04788893E-04) * Pa2 +
            (-6.42070836E-05) * Ta * Pa2 +
            (1.16257971E-06) * Ta2 * Pa2 +
            (7.68023384E-06) * va * Pa2 +
            (-5.47446896E-07) * Ta * va * Pa2 +
            (-3.59937910E-08) * va2 * Pa2 +
            (3.02122035E-04) * Pa3 +
            (-4.77403547E-06) * Ta * Pa3 +
            (1.73825715E-06) * va * Pa3 +
            (-9.77675906E-06) * Pa4;
        coefficients[3] =
            (-1.21206673E-05) +
            (-2.18203660E-07) * Ta +
            (7.51269482E-09) * Ta2 +
            (9.79063848E-11) * Ta3 +
            (1.25006734E-06) * va +
            (-1.81584736E-09) * Ta * va +
            (-3.52197671E-10) * Ta2 * va +
            (-3.36514630E-08) * va2 +
            (1.35908359E-10) * Ta * va2 +
            (4.17032620E-10) * va3 +
            (-3.59413173E-07) * Pa +
            (7.04388046E-07) * Ta * Pa +
            (-1.89309167E-08) * Ta2 * Pa +
            (-4.79768731E-07) * va * Pa +
            (7.96079978E-09) * Ta * va * Pa +
            (1.62897058E-09) * va2 * Pa +
            (-4.36497725E-06) * Pa2 +
            (1.68737969E-07) * Ta * Pa2 +
            (2.67489271E-08) * va * Pa2 +
            (-4.09087898E-07) * Pa3;
        coefficients[4] =
            (-1.30369025E-09) +
            (4.13908461E-10) * Ta +
            (9.22652254E-12) * Ta2 +
            (-5.08220384E-09) * va +
            (-2.24730961E-11) * Ta * va +
            (1.17139133E-10) * va2 +
            (3.94367674E-08) * Pa +
            (-1.18566247E-09) * Ta * Pa +
            (3.34678041E-10) * va * Pa +
            (3.23926897E-09) * Pa2;
        coefficients[5] =
            (6.62154879E-10) +
            (4.03863260E-13) * Ta +
            (1.95087203E-12) * va +
            (-1.15606447E-10) * Pa;
        coefficients[6] =
            (-4.73602469E-12);
    }
}

extern "C" __global__ void utci_scalar_weather_integrate(
    const float* ta, const float* rh, const float* tmrt, const float* wind,
    const double* coefficients, float* output, const long long n)
{
    #pragma unroll 1
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (long long)blockDim.x * gridDim.x) {
        const float Ta = ta[0], RH = rh[0], Tmrt = tmrt[i], va10m = wind[0];
        float out;
        if (Ta <= -999.0f || RH <= -999.0f || va10m <= -999.0f || Tmrt <= -999.0f) {
            out = -999.0f;
        } else {
            float D_Tmrt = Tmrt - Ta;
            out = coefficients[0] +
                coefficients[1] * D_Tmrt +
                coefficients[2] * powf(D_Tmrt, 2.0f) +
                coefficients[3] * powf(D_Tmrt, 3.0f) +
                coefficients[4] * powf(D_Tmrt, 4.0f) +
                coefficients[5] * powf(D_Tmrt, 5.0f) +
                coefficients[6] * powf(D_Tmrt, 6.0f);
        }
        output[i] = out;
    }
}
