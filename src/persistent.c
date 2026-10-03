#include <math.h>
#include <stdlib.h>

/* Exact forward recursion for OFF plus persistent emission-size states.
   Observations may be separated by arbitrary positive integer gaps.
   K=1 reduces to an ordinary two-state HMM. */
double forward_ll(double pon, double poff, const double *w,
                  const double *on, const double *off, const double *gaps,
                  int n, int k) {
    if (n < 1 || k < 1 || pon <= 0 || poff <= 0 || pon >= 1 || poff >= 1)
        return -INFINITY;
    double pi1 = pon / (pon + poff), pi0 = 1 - pi1;
    double r = 1 - pon - poff;
    double *a = calloc(k, sizeof(double));
    double a0 = pi0 * off[0], s = a0;
    for (int j=0; j<k; j++) { a[j] = pi1 * w[j] * on[j]; s += a[j]; }
    if (s <= 0 || !isfinite(s)) { free(a); return -INFINITY; }
    double ll = log(s);
    a0 /= s;
    for (int j=0; j<k; j++) a[j] /= s;
    for (int t=1; t<n; t++) {
        double total = 0;
        for (int j=0; j<k; j++) total += a[j];
        double gap = gaps[t-1];
        double rn = gap == 1 ? r : pow(r, gap);
        double stay = gap == 1 ? 1-poff : pow(1-poff, gap);
        double t00 = pi0 + pi1*rn, t01 = pi1*(1-rn);
        double t10 = pi0*(1-rn), t11 = pi1 + pi0*rn;
        double reset = t11 - stay;
        if (reset < 0 && reset > -1e-14) reset = 0;
        double b0 = off[t] * (a0*t00 + total*t10);
        double incoming = a0*t01 + total*reset;
        s = b0;
        for (int j=0; j<k; j++) {
            a[j] = on[t*k+j] * (a[j]*stay + w[j]*incoming);
            s += a[j];
        }
        if (s <= 0 || !isfinite(s)) { free(a); return -INFINITY; }
        a0 = b0/s;
        for (int j=0; j<k; j++) a[j] /= s;
        ll += log(s);
    }
    free(a);
    return ll;
}

/* Original event-level IPW backward recursion, no statistical change. */
void event_detection(double poff, const double *miss, int n, int k, double *out) {
    double *alpha = malloc(k*sizeof(double));
    double *total = calloc(k,sizeof(double));
    for (int j=0;j<k;j++) alpha[j]=1;
    for (int t=n-1;t>=0;t--) {
        for (int j=0;j<k;j++) {
            alpha[j]=miss[t*k+j]*(poff+(1-poff)*alpha[j]);
            total[j]+=alpha[j];
        }
    }
    for (int j=0;j<k;j++) out[j]=1-total[j]/n;
    free(alpha); free(total);
}
