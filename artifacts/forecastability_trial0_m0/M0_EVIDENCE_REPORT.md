# Kahlus Forecastability Trial 0 - M0 Evidence Report

Gate passed: `True`
Bit-stable baseline table: `True`
Clean worktree required: `True`
Clean worktree observed: `True`
Missing required rows: `none`

## Baseline Ranking

| rank | model | MSE | MAE | Pearson r | R2 |
|---:|---|---:|---:|---:|---:|
| 1 | ridge | 0.400167477825 | 0.491623961193 | 0.889839403979 | 0.791596240765 |
| 2 | persistence | 0.453430674093 | 0.527753859726 | 0.883110009389 | 0.76385722911 |
| 3 | gbm | 0.487715925363 | 0.543412061194 | 0.865786528198 | 0.746001766967 |
| 4 | transformer | 0.616897149298 | 0.621403143155 | 0.826538901871 | 0.67872530353 |
| 5 | model | 0.956837794261 | 0.781133523775 | 0.756067058238 | 0.501687157621 |
| 6 | tcn | 1.00218884779 | 0.76428077666 | 0.754491121906 | 0.478068721431 |
| 7 | mlp | 1.47584746633 | 0.933128017516 | 0.585243398257 | 0.231391412136 |

## Gate Discipline

M0 stops here. M1 should not start until the worktree is clean and this harness is accepted as the ground-truth evaluator.
