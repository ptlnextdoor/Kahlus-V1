# Concepts for non-specialists

```{admonition} Audience
:class: note
Start here if you are a high school student, a curious builder, or a technical reader who wants the plain-English version before the math.
```

## What is Kahlus trying to measure?

The brain is always changing. Electroencephalography, or EEG, records tiny voltage changes from sensors on the scalp. Kahlus treats those signals as partial observations of a hidden nervous-system state.

A useful first question is simple:

> Given a short EEG window right now, can a model predict the next EEG window without cheating?

That question is not the same as understanding the mind. It is a controlled test of whether the code can prepare data, prevent leakage, run baselines, and explain when simple models work.

## Why can a simple ridge model look strong?

EEG is smooth over very short time spans. If the target window is only one sample later, much of the future can look like a shifted copy of the present. A linear model may score well because the benchmark is easy, not because the model found a deep neural state.

That is why Kahlus docs separate:

- **benchmark evidence**: figures generated from real run artifacts;
- **diagnostic refits**: sanity checks that explain a result;
- **schematics**: conceptual diagrams only;
- **clinical claims**: claims that require much stronger evidence.

## The safety rule

If a figure does not say where the data came from, what split was used, and what code generated it, it does not count as evidence.
