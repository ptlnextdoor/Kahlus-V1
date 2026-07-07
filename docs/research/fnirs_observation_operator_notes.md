# fNIRS Observation-Operator Notes

fNIRS is not a NeuroTwin pivot. It is useful because it validates the observation-operator worldview: measured signals are lossy observations of latent neural and physiological fields, not the field itself.

## Observation Story

Latent cortical activity drives hemodynamic activity. Hemodynamics interact with scalp/skull physiology, systemic signals, optode geometry, and motion artifacts. The fNIRS measurement is therefore:

```text
latent neural field
-> hemodynamic field
-> optical propagation and systemic physiology
-> artifact processes
-> observed optical density
```

## Rytov Approximation

```{math}
\log\left(\frac{\phi_1(t;\theta)}{\phi_0}\right)
\approx
J\delta\mu_a(t;\theta)
```

```{math}
\phi_1(t;\theta)\approx \phi_0\exp(J\delta\mu_a(t;\theta))
```

```{math}
\Delta OD(t;\theta)=\log\bar\phi_{\mathrm{base}}-\log\phi_1(t;\theta)
```

```{math}
Y_{\mathrm{fNIRS}}(t)=\Delta OD(\phi_0\exp(J\delta\mu_a(t)))+\epsilon
```

Here `J` is a sensitivity matrix and `delta mu_a` is absorption change.

## HRF and Physiology

Future synthetic realism should separate:

- neural field
- hemodynamic field
- physiology nuisance field
- artifact field
- modality observation operator

The hemodynamic bridge can be written:

```{math}
F_{\mathrm{hemo}}(t)=(h_{\mathrm{HRF}}*g(F_{\mathrm{neural}}))(t)
```

## Why This Helps NFC

fNIRS makes it obvious that a modality is not the primitive. The primitive is the latent field plus a measurement operator. That supports the NFC framing for fMRI, EEG, behavior, and future modalities.

## Claim Boundaries

- No fNIRS implementation claim is made.
- No clinical, MDD, diagnostic, or treatment claim is made.
- No SIMR/private work or personal contact information belongs in this repo unless public, licensed, and explicitly approved.
- fNIRS notes must not block the A100 synthetic diagnostic.
