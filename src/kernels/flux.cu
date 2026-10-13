using Real = SOLWEIG_REAL;
constexpr double PI = 3.141592653589793238462643383279502884;

template<typename T> __device__ __forceinline__ T at(
    const unsigned long long* ptr, const long long* size, int k, long long i) {
    return reinterpret_cast<const T*>(ptr[k])[size[k] == 1 ? 0 : i];
}
template<typename T> __device__ __forceinline__ T fourth(T x) { return pow(x, T(4)); }
template<typename T> __device__ __forceinline__ T poly(T x) {
    return T(63.227)*pow(x,T(6)) - T(161.51)*pow(x,T(5)) + T(156.91)*pow(x,T(4))
        - T(70.424)*pow(x,T(3)) + T(16.773)*pow(x,T(2)) - T(0.4863)*x;
}
template<typename T> __device__ __forceinline__ void weights(
    T svf, T veg, T aveg, T total, T& wveg, T& wwall, T& wsky, T& wrefl) {
    T onlywall = (total-poly(svf))/total;
    T aboveveg = (total-poly(aveg))/total;
    wwall = onlywall-aboveveg;
    T combined = veg+svf-T(1);
    wsky = poly(combined)/total;
    wrefl = (total-poly(combined))/total;
    wveg = (total-poly(combined))/total-wwall;
}

extern "C" __global__ void wedge(
    const unsigned long long* p, const long long* sizes, Real* out, long long n, double zen) {
    long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double alfa=at<double>(p,sizes,0,i), tb=tan(zen), ta=tan(alfa);
    double xa=1.0-2.0/(ta*tb), ha=2.0/(ta*tb), ba=1.0/ta;
    double hkil=2.0*ba*ha, ukil=0.0;
    if(xa<0.0) {
        double qa=tb/2.0;
        double za=sqrt(pow(ba,2.0)-pow(qa,2.0)/4.0);
        double phi=atan(za/qa);
        double area=(sin(phi)-phi*cos(phi))/(1.0-cos(phi));
        ukil=2.0*ba*xa*area;
    }
    double surface=hkil+ukil;
    out[i]=Real((2.0*PI*ba-surface)/(2.0*PI*ba));
}

extern "C" __global__ void delay(
    const unsigned long long* p, const long long* sizes, Real* out, long long n,
    int first, int update, Real weight) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Real current=at<Real>(p,sizes,0,i), previous=first?current:at<Real>(p,sizes,1,i);
    Real delayed=current*(Real(1)-weight)+previous*weight;
    out[i]=delayed;
    out[n+i]=update?delayed:previous;
}

extern "C" __global__ void kup(
    const unsigned long long* p, const long long* sizes, Real* out, long long n,
    Real radI,Real radD,Real radG,double altitude,Real albedo) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Real svf=at<Real>(p,sizes,0,i), f=at<Real>(p,sizes,1,i);
    Real solar=Real(sin(altitude*(PI/180.0)));
    Real reflected=radD*svf+albedo*(Real(1)-svf)*(radG*(Real(1)-f)+radD*f);
    for(int d=0;d<5;++d) {
        Real alb=at<Real>(p,sizes,2+d,i), nosh=at<Real>(p,sizes,7+d,i);
        out[(long long)d*n+i]=alb*radI*solar+reflected*nosh;
    }
}

extern "C" __global__ void lside(
    const unsigned long long* p, const long long* sizes, Real* out, long long n,
    double azimuth,double altitude,Real Ta,Real SBC,Real ewall,Real esky,double offset,Real CI,int anisotropic) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    if(anisotropic==1) {
        for(int d=0;d<4;++d) out[(long long)d*n+i]=at<Real>(p,sizes,15+d,i)*Real(0.5);
        return;
    }
    Real walltemp=at<Real>(p,sizes,12,i), down=at<Real>(p,sizes,13,i);
    Real f=Real(2)*at<Real>(p,sizes,14,i)-Real(1), c=Real(1)-CI;
    Real base=fourth(Ta+Real(273.15));
    Real allsky=esky*SBC*base*(Real(1)-c)+c*SBC*base;
    for(int d=0;d<4;++d) {
        Real svf=at<Real>(p,sizes,d,i), veg=at<Real>(p,sizes,4+d,i), aveg=at<Real>(p,sizes,8+d,i);
        Real wveg,wwall,wsky,wrefl;
        weights(svf,veg,aveg,Real(4.4897),wveg,wwall,wsky,wrefl);
        Real wallsun=Real(0), wallshade=SBC*ewall*base*wwall*Real(0.5);
        bool illuminated=false; double azi=0.0;
        if(d==0){illuminated=azimuth>(180.0-offset)&&azimuth<=(360.0-offset);azi=azimuth-180.0+offset;}
        if(d==1){illuminated=azimuth<=(90.0-offset)||azimuth>(270.0-offset);azi=azimuth-270.0+offset;}
        if(d==2){illuminated=azimuth>(360.0-offset)||azimuth<=(180.0-offset);azi=azimuth+offset;}
        if(d==3){illuminated=azimuth>(90.0-offset)&&azimuth<=(270.0-offset);azi=azimuth-90.0+offset;}
        if(altitude>0.0 && illuminated) {
            Real svfa=asin(exp(log(Real(1)-svf)/Real(2)));
            Real alfa=atan(svfa), beta=atan(tan(svfa*f));
            Real betasun=(alfa-beta)/Real(2)+beta;
            Real warmed=Ta+Real(273.15)+walltemp*Real(sin(azi*(PI/180.0)));
            wallsun=SBC*ewall*fourth(warmed)*wwall*(Real(1)-f)*cos(betasun)*Real(0.5);
            wallshade=SBC*ewall*base*wwall*f*Real(0.5);
        }
        Real up=at<Real>(p,sizes,15+d,i);
        Real sky=((svf+veg-Real(1))*allsky)*wsky*Real(0.5);
        Real vegetation=SBC*ewall*base*wveg*Real(0.5);
        Real ground=up*Real(0.5), reflection=(down+up)*wrefl*(Real(1)-ewall)*Real(0.5);
        out[(long long)d*n+i]=sky+wallsun+wallshade+vegetation+ground+reflection;
    }
}

extern "C" __global__ void surface(
    const unsigned long long* p, const long long* sizes,Real* out,long long n,
    double altmax,double fraction,double sunrise,Real clearness,int landcover) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Real amplitude=at<Real>(p,sizes,0,i)*Real(altmax)+at<Real>(p,sizes,1,i);
    Real wallamplitude=at<Real>(p,sizes,2,i)*Real(altmax)+at<Real>(p,sizes,3,i);
    Real phase=((Real(fraction)-Real(sunrise)/Real(24))/(at<Real>(p,sizes,4,i)/Real(24)-Real(sunrise)/Real(24)))*Real(PI)/Real(2);
    Real wallphase=((Real(fraction)-Real(sunrise)/Real(24))/(at<Real>(p,sizes,5,i)/Real(24)-Real(sunrise)/Real(24)))*Real(PI)/Real(2);
    Real ground=amplitude*sin(phase), wall=wallamplitude*sin(wallphase);
    if(wall<Real(0)) wall=Real(0);
    ground=ground*clearness;wall=wall*clearness;
    if(landcover==1 && ground<Real(0)) ground=Real(0);
    out[i]=ground;out[n+i]=wall;
}

extern "C" __global__ void assemble(
    const unsigned long long* p,const long long* sizes,Real* out,long long n,
    Real absK,Real absL,Real Fside,Real Fup,Real Fcyl,int cyl,int anisotropic,Real SBC) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    Real kd=at<Real>(p,sizes,0,i),ku=at<Real>(p,sizes,1,i);
    Real ksum=at<Real>(p,sizes,2,i)+at<Real>(p,sizes,3,i)+at<Real>(p,sizes,4,i)+at<Real>(p,sizes,5,i);
    Real ld=at<Real>(p,sizes,6,i),lu=at<Real>(p,sizes,7,i);
    Real lsum=at<Real>(p,sizes,8,i)+at<Real>(p,sizes,9,i)+at<Real>(p,sizes,10,i)+at<Real>(p,sizes,11,i);
    Real absorbed;
    if(cyl==1 && anisotropic==0) {
        absorbed=absK*(at<Real>(p,sizes,12,i)*Fcyl+(kd+ku)*Fup+ksum*Fside)
            +absL*((ld+lu)*Fup+lsum*Fside);
    } else if(cyl==1 && anisotropic==1) {
        absorbed=absK*(at<Real>(p,sizes,13,i)*Fcyl+(kd+ku)*Fup+ksum*Fside)
            +absL*((ld+lu)*Fup+at<Real>(p,sizes,14,i)*Fcyl+lsum*Fside);
    } else {
        absorbed=absK*((kd+ku)*Fup+ksum*Fside)+absL*((ld+lu)*Fup+lsum*Fside);
    }
    out[i]=sqrt(sqrt(absorbed/(absL*SBC)))-Real(273.2);
    out[n+i]=absorbed;
}
