# Part 2 — which controller, and why

**Picked: a two-stage lead compensator with a single rolloff pole. No
integrator and no term aimed at any particular frequency. Crossed over at the
delay-limited frequency, 2.81 Hz.**

```
C(s) = Kp * [(1 + sT)/(1 + sT/alpha)]^2 * 1/(1 + s/w_p)

  Kp = 0.1724,  T = 110.883 ms,  alpha = 8,  w_p = 20 Hz
  1 kHz, bilinear-discretised as an SOS cascade, output saturated at 0.5 N*m
```

Everything below is produced by four scripts in `scripts/part2/`:
`design_lead_lag.py`, `analyze_sensitivity.py`, `analyze_tonals.py`,
`evaluate_closed_loop.py` (or all four via `python scripts/run_part2.py`).

---

## Result, stated honestly

| | LOS RMS | vs open loop |
| --- | ---: | ---: |
| open loop (delivered plant) | 236.7 µrad | — |
| **closed loop (delivered plant)** | **236.7 µrad** | +0.0 |
| closed loop (frictionless plant) | 289.7 µrad | +53.0 |

**Falls short of the ~50 µrad target by 4.7×, and does not improve on open
loop.** The spec anticipates ~330 µrad open loop; this seed and record length
measure 236.7 µrad, which is the number the closed-loop results should be
compared against rather than the nominal 330.

The two closed-loop rows say different things, and only the second is about
control design.

**The delivered loop does essentially nothing, because it never commands past
the Coulomb friction level.** Peak commanded torque is 4.389 mN·m against a
5.000 mN·m Coulomb torque, and the gimbal moves 0.99 µrad RMS in total. A
2.81 Hz loop never asks for enough torque to move the axis. So the exact
equality with open loop is a *friction deadband*, not evidence that the designed
sensitivity was achieved — and it must not be reported as a controller that
wisely chose not to act.

Precisely what that deadband is, since it matters for anything built on top of
it: friction here is modelled as `Tc·sign(θ̇) + b·θ̇`, a velocity-dependent
torque rather than a static-friction constraint, so nothing literally sticks. At
rest the friction term is zero and a sub-threshold command does accelerate the
axis; the instant velocity goes positive the term snaps to −Tc and reverses the
net torque. The velocity chatters about zero and net displacement stays
negligible. Sustained motion requires T > Tc + b·θ̇, so 5 mN·m is the right
threshold and the measured behaviour is right — but the mechanism is sign
chatter near zero velocity, not sticking.

**With friction removed, the designed loop does act — and makes things worse.**
That is the real verdict, and it follows directly from the validated sensitivity
function rather than being a surprise.

## Why it gets worse, not better

The sensitivity function was measured, not just computed
(`analyze_sensitivity.py`, `sensitivity.png`): the analytical |S| matches the
simulator to −0.4% at 47 Hz and 75 Hz with coherence 1.000, and −11.1% at 22 Hz
with coherence 0.998. Below ~8 Hz coherence collapses and no transfer function
exists there, so those points are excluded rather than quoted.

| quantity | value |
| --- | --- |
| crossover | 2.81 Hz |
| rejection bandwidth (−3 dB) | 1.27 Hz |
| peak sensitivity Ms | 3.81 (+11.6 dB) at 5.09 Hz |
| \|S(22 Hz)\| | 1.156 (+1.26 dB) |
| \|S(47 Hz)\| | 1.003 (+0.03 dB) |
| \|S(13 Hz)\| — the 47 Hz alias | 0.755 (−2.44 dB) |

The band table from `evaluate_closed_loop.py` shows the trade directly
(frictionless closed loop ÷ open loop):

| band | ratio |
| --- | ---: |
| 0.0 – 1.5 Hz | 3.38 |
| 1.5 – 5.0 Hz | 2.46 |
| 5.0 – 20 Hz | 1.72 |
| 20 – 60 Hz | 0.99 |
| 60 – 200 Hz | 1.00 |

Three structural reasons, none of them a tuning failure:

1. **Crossover is delay-limited to a few Hz.** Camera ZOH plus the
   90th-percentile latency is 43.96 ms. A pure delay costs ωT of phase, so a 45°
   margin caps crossover at (π/4)/T ≈ 2.81 Hz no matter how the compensator is
   shaped.
2. **The disturbance lives above that crossover.** The broadband platform
   vibration is shaped only by an 80 Hz second-order low-pass, so most of its
   ~200 µrad sits where |S| ≈ 1, and both tones are far above crossover.
3. **The Bode trade is lopsided here.** The attenuation won below 1.27 Hz spans
   a narrow band holding little disturbance energy; the payback lands across
   1.5–20 Hz, which is wide and well populated. The 22 Hz and 47 Hz bands come
   out at ratio ≈ 1.0 — untouched — while everything below 20 Hz is amplified.

## The tonals

**The controller has no term aimed at either tone.** Nothing in it is tuned to a
tone frequency, so nothing has to track their drift. That is a deliberate
choice, and the cost of it is measured rather than hidden.

**They drift, and faster than "slow wander" suggests** (`analyze_tonals.py`,
`tonal_drift.png`), measured over 120 s from the disturbance generator:

| tone | spec | measured range | std | drift rate |
| --- | --- | --- | ---: | --- |
| 22 Hz | ±2 Hz | 20.00 – 24.00 Hz | 0.96 | 0.63 Hz per root-second |
| 47 Hz | ±3 Hz | 44.35 – 50.00 Hz | 1.35 | 0.95 Hz per root-second |

The correlation time is 20 s, which sounds slow, but the frequency still moves
most of a hertz within a single second. Anything that targets a tone *by
frequency* would have to follow it continuously rather than be placed once at
nominal.

**But the drift is not what stops this loop.** |S| was evaluated across the
entire band each tone actually visits, not only at nominal:

| tone | band | \|S\| min | \|S\| max | \|S\| at nominal |
| --- | --- | ---: | ---: | ---: |
| 22 Hz | 20.0 – 24.0 Hz | 1.104 | 1.158 | 1.156 |
| 47 Hz | 44.0 – 50.0 Hz | 0.979 | 1.031 | 1.003 |

|S| is flat and within a few percent of 1 across both bands. Wherever the tone
happens to sit, the loop is doing nothing about it — so the obstacle is the
2.81 Hz crossover, an octave and two octaves below the tones, not the wander.

**And for the 47 Hz tone the sensor makes it worse than "not rejected".** It sits
above the 30 Hz camera Nyquist, so the camera does not attenuate it — it
*relocates* it (`tonal_aliasing.png`). Measured against the true LOS, the
camera's own output reports:

- at 47 Hz: **−15 dB** of the true power — the tone is nearly gone;
- at 13 Hz, where the true LOS carries only broadband: **+7 dB more** than the
  truth, with the camera's peak at 12.82 Hz.

So the 47 Hz tone arrives at the controller as a 13 Hz signal that is
indistinguishable from a genuine 13 Hz disturbance. Acting on that peak would
mean commanding torque at the wrong frequency entirely. This is an information
limit of the sensor; no control law recovers from it.

**Summary of the answer:** the tones pass through untouched, |S| ≈ 1 at both,
and this is reported rather than disguised. Rejecting them needs a higher
crossover than the camera's latency permits and — for 47 Hz — a sensor that does
not fold it. Both are sensor changes, not compensator changes.

## Saturation and wind-up

**Saturation** is applied at the controller output, clamped at ±0.5 N·m. At the
nominal disturbance the limit is inactive 100% of the time — peak demand is
0.88% of the limit — so the path is exercised deliberately by scaling the
disturbance ×200, which makes the limit active on 0.2% of ticks and clips
cleanly (`saturation.png`). The script prints a warning if that stress case ever
stops reaching the limit, so the test cannot silently become vacuous.

**Wind-up cannot occur, by construction rather than by an anti-windup scheme.**
Wind-up requires a state that integrates error while the actuator is clipped,
i.e. a compensator pole at s = 0. This compensator's poles are:

```
s = -72.148  (11.48 Hz)     lead stage 1
s = -72.148  (11.48 Hz)     lead stage 2
s = -125.664 (20.00 Hz)     rolloff
```

All strictly stable, none at the origin. The output therefore returns to the
unsaturated control law on the first tick after the demand falls back inside the
limit.

This is also why there is **no integrator**: there is no DC pointing requirement
here — the error is a zero-mean vibration, not a bias — and an integrator would
spend phase at crossover, which is the scarce resource, while adding the wind-up
problem it would then need protecting against.

## The rest of the design, briefly

**Why lead and not PID.** The plant is a double integrator, so it contributes a
fixed −180° before any delay or mode is counted. Stabilising it needs phase lead
— the one thing an integrator cannot supply and a derivative term supplies only
with unbounded high-frequency gain. Each lead stage gets a matching pole at α
times its zero, bounding the high-frequency boost to α².

**Why two stages.** One stage at α = 8 supplies at most ~51°, and the plant plus
delay needs more. Raising α to get it from a single stage reintroduces the
high-frequency gain problem the pole exists to solve.

**Why the rolloff pole at 20 Hz.** It gain-stabilises both structural modes:
|L(75 Hz)| = 0.094 (+20.5 dB margin) and |L(220 Hz)| = 0.00033 (+69.6 dB). Gain
stabilisation is preferred to phase stabilisation because it does not depend on
the modes sitting at their nominal frequencies — with ζ = 0.03 the 75 Hz mode
has Q ≈ 16.7. Measured: Ms varies only between 3.79 and 3.82 across ±10% of
mode-frequency error, so the design does not care where the modes actually are.

**Design verification.** |L(jω_c)| = 1.000000 and phase margin 45.00° against
targets of 1.000000 and 45.00° (`design_lead_lag.py`, `loop_bode.png`).

## What would actually help

Nothing in the compensator. The limit is the camera: 44 ms of latency caps
crossover below the disturbance, and 60 Hz sampling folds the 47 Hz tone to
13 Hz so it cannot be distinguished from a real 13 Hz disturbance. Both are
sensor limits, and the second is an information limit.

Two things follow for a next iteration, and they are separable:

- **A faster, lower-latency sensor** raises the achievable crossover and unfolds
  the 47 Hz tone. That is the only route to meaningful rejection of either tone.
  This is what Part 3 tests.
- **Friction must be dealt with independently.** Even with a perfect
  compensator, a loop whose commanded torque stays under the 5 mN·m Coulomb
  level is mechanically decoupled from the load. Any design for this axis needs
  enough authority at the frequencies where the disturbance actually lives to
  command past that, or |S| on paper means nothing.
