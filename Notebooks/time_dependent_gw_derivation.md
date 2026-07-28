# The time-dependent GW spectrum of a turbulent / magnetic source

A self-contained analytic derivation of how the gravitational-wave spectrum
$\Omega_{\rm GW}(k,t)$ **builds up in time**, with the code that implements it
(`Notebooks/time_dependent_gw_spectrum.py`). Notation: conformal time $t$,
$c=1$, one Fourier mode $\mathbf{k}$ with $k=|\mathbf{k}|$; the source is the
transverse–traceless (TT) anisotropic stress $S_k(t)$ (for us $\sim(B_iB_j)_k$).

---

## Chapter 1 — The sourced wave equation

In a flat FRW background the TT metric perturbation obeys
$$
h_{ij}'' + 2\frac{a'}{a}h_{ij}' - \nabla^2 h_{ij} = 16\pi G\, a^2\,\Pi^{\rm TT}_{ij}.
$$
For sub-horizon modes ($k\gg aH$) the Hubble friction and the $a''/a$ term are
negligible over an emission time, and after Fourier transforming each mode
decouples into a **driven harmonic oscillator**:
$$
\boxed{\,h_k'' + k^2 h_k = S_k(t)\,},\qquad S_k \equiv 16\pi G\,a\,\Pi^{\rm TT}_k .
$$
Two facts that matter later are already visible here: (i) the equation is
**diagonal in $k$** — source mode $k$ drives GW mode $k$, with no spatial
propagation; (ii) the natural frequency is $\omega=k$ (the GW dispersion), so a
mode "wants" to oscillate at $\omega=k$.

---

## Chapter 2 — Green's function solution

The retarded Green's function of the oscillator is $G(t,t')=\sin[k(t-t')]/k$, so
for a source switched on at $t_0$,
$$
h_k(t)=\int_{t_0}^{t}\frac{\sin[k(t-t')]}{k}\,S_k(t')\,dt'
=\frac{1}{k}\Big[\sin(kt)\,C(t)-\cos(kt)\,D(t)\Big],
$$
$$
C(t)=\int_{t_0}^{t}\!\cos(kt')\,S_k(t')\,dt',\qquad
D(t)=\int_{t_0}^{t}\!\sin(kt')\,S_k(t')\,dt'.
$$
$C,D$ are the **accumulated source integrals**: they change only while $S_k\neq0$.

---

## Chapter 3 — The exact GW energy spectrum

The GW energy density is $\rho_{\rm GW}\propto\langle\dot h_{ij}\dot h_{ij}\rangle$;
per mode the oscillator energy is $\dot h_k^2+k^2h_k^2$. Differentiating $h_k$
(the boundary term vanishes because $\sin[k(t-t)]=0$),
$$
\dot h_k=\cos(kt)\,C(t)+\sin(kt)\,D(t),
$$
and the fast oscillations **cancel** in the energy:
$$
\boxed{\;\dot h_k^2+k^2h_k^2=C(t)^2+D(t)^2=\Big|\int_{t_0}^{t}e^{ikt'}S_k(t')\,dt'\Big|^2\;}.
$$
This is the crux. The GW energy at mode $k$ is the **modulus-squared of the
source's temporal Fourier transform at $\omega=k$, accumulated to time $t$.**
Note $h_k$ keeps oscillating forever, but its *energy* changes only while the
source acts — it **freezes** the instant $S_k\to0$.

The spectral energy density (energy per $\ln k$), with the $k^3$ from the mode
measure $d^3k=4\pi k^2\,dk$ and $d/d\ln k$, is
$$
\Omega_{\rm GW}(k,t)\ \propto\ k^3\,\big[C^2+D^2\big].
$$

---

## Chapter 4 — Ensemble average: the master formula

A turbulent source is stochastic, so we ensemble-average $C^2+D^2$. Writing the
**unequal-time correlator (UETC)** of the stress as a spatial power spectrum
$\Pi(k)$ times a normalised temporal correlation $f$ (with $f(t,t)=1$),
$$
\langle S_k(t_1)S_k^*(t_2)\rangle=\Pi(k)\,f(t_1,t_2),
$$
we get the **master formula**
$$
\boxed{\;\Omega_{\rm GW}(k,t)\ \propto\ k^3\,\Pi(k)\,I(k,t)\;},\qquad
I(k,t)=\int_{t_0}^{t}\!\!\int_{t_0}^{t}\!dt_1\,dt_2\,\cos[k(t_1-t_2)]\,f(t_1,t_2).
$$
The spectrum factorises into a **spatial part** $\Pi(k)$ (the stress shape — for
$\sim B^2$ it is the convolution of two field spectra, white at large scales,
peaked at $2k_0$) and a **temporal build-up** $I(k,t)$ carrying all the time
dependence. Everything that follows is about $I(k,t)$.

---

## Chapter 5 — The temporal build-up $I(k,t)$ for three source models

**(a) Coherent, constant source** ($f\equiv1$, source on over $[t_0,t]$).
$$
I(k,t)=\Big|\int_{t_0}^{t}e^{ikt'}dt'\Big|^2=\frac{2-2\cos(k\Delta t)}{k^2}
=\frac{4\sin^2(k\Delta t/2)}{k^2},\qquad \Delta t=t-t_0.
$$
Its running time-average (the smooth envelope, dropping the ringing) is
$$
\langle I\rangle(k,\Delta t)=\frac{2}{k^2}\Big(1-\frac{\sin k\Delta t}{k\Delta t}\Big)
=\frac{2}{k^2}\,\big[1-\mathrm{sinc}(k\Delta t)\big].
$$
Normalised to its saturated value $2/k^2$ this defines the **build-up fraction**
$$
\boxed{\,B(k,\Delta t)=1-\mathrm{sinc}(k\Delta t)\,}\ \xrightarrow[k\Delta t\ll1]{}\ \tfrac16(k\Delta t)^2,
\qquad \xrightarrow[k\Delta t\gg1]{}\ 1 .
$$

**(b) Decorrelating stationary source** ($f=g(t_1-t_2)$, on for a lifetime
$\tau$). For $\tau$ much longer than the decorrelation time,
$$
I(k,\tau)\ \approx\ \tau\!\int g(s)\cos(ks)\,ds\ =\ \tau\,\hat g(k),
$$
so the GW grows **linearly in the source lifetime**, at a rate set by the
source's temporal power at $\omega=k$. This is the "many eddy-times" / sound-wave
regime (the source keeps adding incoherently).

**(c) Impulsive source** ($S\propto\delta(t-t_0)$). Then $I(k,t)=$ const for
$t>t_0$: the spectrum is set **instantly** to $k^3\Pi(k)\times$const and frozen.

The physical source lies between these: it turns on ~impulsively (broadband,
which is why all $k$ light up at once), stays coherent for a while, and then
either decorrelates or turns off.

---

## Chapter 6 — The time-dependent spectrum and how it evolves

Take the coherent build-up (a). The spectrum is
$$
\boxed{\;\Omega_{\rm GW}(k,t)=\Omega_{\rm sat}(k)\,\big[1-\mathrm{sinc}(k(t-t_0))\big]\;},
\qquad \Omega_{\rm sat}(k)\propto k\,\Pi(k).
$$

**The knee sweeps to low $k$.** At fixed $t$ there is a transition at
$k_\star\sim1/\Delta t$:
- $k>k_\star$ ($k\Delta t\gg1$): $B\to1$, the mode is **saturated** — the
  spectrum equals its final shape $\Omega_{\rm sat}(k)$ (flat/plateau at low $k$
  if $\Pi$ is white, peak at $2k_0$, Kolmogorov $k^{-11/3}$ tail).
- $k<k_\star$ ($k\Delta t\ll1$): $B\to\tfrac16(k\Delta t)^2$, so
  $\Omega_{\rm GW}\to\tfrac16 k^3\Pi(k)\,\Delta t^2$ — the **causal $k^3$**,
  growing as $\Delta t^2$.

As time advances, $k_\star=1/\Delta t$ marches to lower $k$: the flat band
extends, the causal-$k^3$ knee recedes, and the whole spectrum climbs toward
$\Omega_{\rm sat}$. This is the "spectrum fills from high $k$ to low $k$" picture
— high frequencies (small scales) saturate first (fastest, in $\sim1/k$), the
infrared last. If the source **turns off** at $\Delta t=\tau$, the build-up
stops and the spectrum **freezes** with a permanent causal-$k^3$ knee at
$k\sim1/\tau$; for an impulsive $\tau\to0$ the knee runs off to $k\to0$ and the
whole observable band is the saturated shape (the frozen, source-scale spectrum
seen in the decaying simulations).

The figure `images/time_dependent_gw_spectrum.pdf` shows exactly this: five
times, the knee sweeping left, the curves stacking up onto the saturated shape.

---

## Chapter 7 — The code

`Notebooks/time_dependent_gw_spectrum.py` implements both routes and checks they
agree.

- `exact_energy(k, t, S)` — the **exact solution** of Chapter 3 for an arbitrary
  source $S_k(t')$: it forms $C,D$ by cumulative trapezoid and returns
  $E_k(t)=C^2+D^2$ on the whole time grid. `main()` verifies it reproduces the
  closed form $4\sin^2(k\Delta t/2)/k^2$ for a constant source to 6 digits
  (`ratio=1.000000` for $k=0.5,2,8$).
- `build_up(k, dt) = 1 - np.sinc(k dt/pi)` — the analytic window $B$ of Chapter 5.
- `omega_sat(k, kp)` — a physically-motivated saturated shape (white-stress
  plateau, peak at $k_p=2k_0$, $k^{-11/3}$ ultraviolet), and
  `omega_gw(k, t, kp) = omega_sat * build_up` — the time-dependent spectrum of
  Chapter 6, plotted at several times.

Run: `python Notebooks/time_dependent_gw_spectrum.py`. It prints the verification
table and writes the evolution figure.

---

## Chapter 8 — Connection to the simulations

Against the Pencil-Code data (`Notebooks/roperpol_analytic_timeseries.py`) this
model reproduces the *shape* and the qualitative build-up, but the simulation
saturates **faster** than the coherent window: it reaches saturation within
$\Delta t\sim0.02$–$0.06$ at every scale, whereas $1-\mathrm{sinc}(k\Delta t)$
needs $\Delta t\sim0.1$–$0.4$. The sudden magnetic-field onset is therefore
effectively **impulsive** (case (c)): broadband in time, so it lights up all
$k$ at once and sets the frozen, source-scale spectrum — which is why the peak
sits at $2k_0$ (not the sweeping scale $Mk_0$) and does not follow the later
inverse cascade. The continuous, decorrelating emission of case (b) is present
but $M$-suppressed and subdominant.
