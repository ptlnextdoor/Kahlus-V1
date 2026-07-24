# Kahlus IEEE figure source

**Figures that show Kahlus working:** Neural-CASP gates, residual forecastability (RFS), and the synthetic Passive PIC instrument.

> Kahlus is a leakage-controlled residual forecastability engine. The courtroom works when a known signal yields positive RFS and a null yields ~0.

## Why this exists

The IEEE draft needs figures of **Kahlus success**, not only the overlap autopsy. Success here means: M1 recovers synthetic residual information, controls collapse, M5 Passive PIC fires on integrated worlds, and M3/M4 fail honestly when evidence is weak.

The recovered GRU sidecar MSE 3.116 / r 0.972 is **not** a success figure. It is blocked as forecasting skill (overlap-dominated; 2026-07-21).

## Headline number

From committed M4 synthetic horizon sweep (`m4_horizon_sweep.csv`):

| Result | Horizon | Number |
| --- | ---: | --- |
| **KnownSignal RFS** | 1 | **0.220 bits** (cluster p = 0.0005) |
| KnownSignal RFS | 3 | 0.080 bits (still detected) |
| SyntheticNull RFS | 1–3 | ~0 bits |
| **False-certify on null** | 1–3 | **0** |

### Reproduce

```bash
python3 docs/paper/ieee_figure_source/src/Figure1_core_task.py
# or full packet:
python3 docs/paper/ieee_figure_source/render_all.py
```

## Figures

| File | Shows |
| --- | --- |
| `fig1_core_task` | **Hero sweep** (kahlus-sweep-figure): RFS / detection / false-certify vs horizon |
| `fig2_nfc_schematic` | Neural-CASP gate ladder |
| `fig3_gate_protocol` | M5 Passive PIC worlds |
| `fig4_mse_bar` | Product thesis + claim scope |

Filenames kept for IEEE `\includegraphics` compatibility.

## Honest scope

- Demonstrated: synthetic residual forecastability and Passive PIC instrument validity under Neural-CASP gates.
- Not demonstrated in this packet: public-data Passive PCI win, strictly-future GRU superiority over ridge, clinical utility.
- Overlap-audit CSVs (`bnci_horizon_sweep.csv`, `copy_trap_seeds.csv`) remain available for a supplementary leakage figure if needed; they are not the hero set.

## Style

kahlus-bench house style: `constrained_layout`, dpi 130, CI error bars, black dashed zero reference, sentence-length takeaway titles, regenerate from committed CSVs.
