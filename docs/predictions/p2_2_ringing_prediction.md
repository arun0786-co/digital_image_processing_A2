# 2.2 prediction: ringing of the step-edge image

written 2026-10-23 16:29 IST, before any filtering code for problem 2 was
written or run. this file is not edited after the experiment.

## setup

s(x, y) is 512 x 512, a square of side 256 and intensity B = 200 centred on a
background A = 50. the square covers rows and columns 128 to 383, so its left
edge lies between columns 127 and 128, at x_e = 127.5. it is zero-padded to
P = 1024 and filtered with a centred, real, zero-phase H. the profile is a row
through the middle of the square (row 255), far from the corners.
u is the frequency index measured from the centre of the padded spectrum, so
index u means u / P cycles per pixel.

## the 1-d reduction

near the middle row and away from the corners the image looks like a function
of x only. an image that varies only in x has a fourier transform confined to
the line v = 0: if f(x, y) = g(x), then F(u, v) = G(u) * P * delta(v). the
filter then acts only through its slice H(u, 0), and the filtered row is the
1-d convolution of g with h_1(x) = ift{ H(u, 0) }.

## ideal low-pass

the slice of the ideal filter is H(u, 0) = 1 for |u| <= D0, else 0, i.e. a
cutoff of f_c = D0 / P cycles per pixel. in the continuous model

    h_1(x) = 2 f_c sinc(2 f_c x) = sin(2 pi f_c x) / (pi x).

a step of height (B - A) at x_e has response

    g(x) = A + (B - A) [ 1/2 + (1/pi) Si(2 pi f_c (x - x_e)) ],
    Si(z) = integral_0^z sin(t) / t dt.

Si has its global maximum at z = pi, where Si(pi) = 1.851937. so

**(i) overshoot**

    OS = 100 (max g - B) / (B - A) = 100 (Si(pi)/pi - 1/2) = 8.95 %,

independent of D0 and P (gibbs). with B - A = 150 the peak is at about
200 + 13.4 = 213.4.

**(ii) distance from the edge to the first overshoot peak**

    2 pi f_c d = pi  ->  d = 1 / (2 f_c) = P / (2 D0) pixels.

| D0 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| predicted OS (%) | 8.95 | 8.95 | 8.95 | 8.95 |
| predicted d = P / (2 D0) (px) | 51.2 | 25.6 | 12.8 | 6.4 |

## expected sources of discrepancy (also predicted before running)

1. **discrete passband.** on the P grid the ideal slice keeps the 2 D0 + 1
   integer frequencies |u| <= D0. the kernel is then a dirichlet kernel,
   h[x] = sin(pi (2 D0 + 1) x / P) / (P sin(pi x / P)), whose first zero is at
   x = P / (2 D0 + 1). the overshoot peak sits at that zero, so the discrete
   prediction is d = P / (2 D0 + 1) = 48.8, 25.0, 12.6, 6.4 px. the overshoot
   of a dirichlet partial sum tends to the same 8.95 % as the number of terms
   grows.
2. **sampling of the profile.** the edge is at x_e = 127.5 and the profile is
   only known at integer columns, so a measured d is a half-integer and can be
   off by up to 0.5 px. the sampled maximum can also miss the true peak, which
   lowers the measured OS slightly, most for D0 = 80 where the peak is narrow.
3. **other edges.** the square is finite, and the zero padding adds a jump of
   -50 at the image border, 128 px from each square edge. for large z,
   Si(z) ~ pi/2 - cos(z)/z, so the ripple from an edge at distance r has
   amplitude about P / (2 pi^2 D0 r) of that edge's jump. for D0 = 10 this is
   about 2 % of the jump for the opposite edge (r = 256) and 1.4 % for the
   border jump (r = 128, jump 50). these ripples add or subtract depending on
   phase, so at D0 = 10 the measured OS can differ from 8.95 % by a few
   percentage points, and the difference should shrink roughly like 1 / D0.
4. **the image is not really 1-d.** the square is only 256 px tall, so its
   spectrum is spread over a band of v around 0, of width about P / 256 = 4
   bins. on the slice at height v the ideal cutoff is sqrt(D0^2 - v^2) < D0,
   which matters most at D0 = 10. the top and bottom edges also send their own
   ripple down to the middle row. both effects again fade as D0 grows.
5. **near a corner** the 1-d reduction fails completely: the local image
   varies in both x and y, its spectrum is not on the v = 0 line, and the 2-d
   ideal kernel is a jinc whose rings are circular. ripples from the vertical
   and horizontal edges overlap there, so the profile near a corner should show
   a different overshoot and a different peak position.

## gaussian low-pass

H(u, 0) = exp(-u^2 / (2 D0^2)). its inverse transform is again a gaussian,

    h_1(x) = exp(-x^2 / (2 sigma_s^2)) / (sqrt(2 pi) sigma_s),
    sigma_s = P / (2 pi D0),

which is positive everywhere and integrates to one. the edge response is

    g(x) = A + (B - A) Phi((x - x_e) / sigma_s),

with Phi the standard normal cdf. this is monotone, so

**predicted gaussian overshoot: OS = 0 % for every D0.**

the transition just gets wider as D0 drops (10 % to 90 % width
2.56 sigma_s = 2.56 P / (2 pi D0)). on the discrete grid the slice is cut off
at |u| = P/2 = 512, where it is exp(-512^2 / (2 D0^2)), which is far below
float64 precision for D0 <= 160, so the discrete kernel stays non-negative to
round-off and OS should be 0 up to about 1e-10 %.

## butterworth (not required, written for 2.2b)

for n = 1 the slice 1 / (1 + (u/D0)^2) is a lorentzian whose inverse transform
is a two-sided exponential, positive everywhere, so OS = 0. for n >= 2 the
kernel has negative lobes and OS > 0, growing with n toward the ideal-filter
limit of 8.95 % as n -> infinity, since the butterworth slice tends to the ideal
box.
