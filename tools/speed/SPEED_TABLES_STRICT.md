# Thinking off: the speed table with the stricter rule for repetition

Output of `tools/speed/tables_strict.py` from the records of `bench/20261108`. The rule is in the docstring of the script: in
addition to the first tally, a story or translation run is left out when lines that occur more than once make up 20 % or more of
its lines (lines of 20 characters or more). The rule is the same for every engine. tok/s; conditions and limits as in the README of
`bench/`.

| Engine | Code | Story | Translation | Structured | Code edit | JSONL edit | Long JSONL |
|---|---|---|---|---|---|---|---|
| YASAKANI fastest model | 529 | 201 | 494 | 909 | 1,222 | 1,156 | 1,260 |
| YASAKANI quality model | 486 | 179 | 483 | 930 | 1,136 | 1,151 | 1,160 |
| MegaCapybara 1.39 Tiny | 465 | 164 | 268 | 599 | 441 | 658 | 666 |
| mkyt v0.3.2 | 395 | 181 | 374 | 555 | 423 | 634 | 639 |
| ignis v0.9.6 | 330 | 167 | 298 | 439 | 331 | 510 | 514 |
| NInfer (November 2 build) | 264 | 123 | 257 | 386 | 295 | 443 | 438 |
| vLLM 0.31.0 | 290 | 148 | 261 | 354 | 241 | 426 | 431 |
| SGLang 0.5.22 | 252 | 129 | 246 | 371 | 430 | 466 | 451 |
| llama.cpp b11700 | 242 | 133 | 232 | 337 | 244 | 360 | 362 |
| TensorFold 0.6.6 | 181 | 104 | 172 | 405 | 383 | 408 | 261 |
| (largest value of the other engines) | 465 | 181 | 374 | 599 | 441 | 658 | 666 |
| YASAKANI fastest model / largest value of the others | 1.14 | 1.11 | 1.32 | 1.52 | 2.77 | 1.76 | 1.89 |
| YASAKANI quality model / largest value of the others | 1.04 | 0.99 | 1.29 | 1.55 | 2.58 | 1.75 | 1.74 |

## What changed against the first tally (`bench/20261108/tables/SPEED_TABLES.md`)

* YASAKANI fastest model, Story: 285 -> 201
* mkyt v0.3.2, Story: 204 -> 181
* vLLM 0.31.0, Story: 150 -> 148
* llama.cpp b11700, Story: 139 -> 133
* largest value of the other engines, Story: 204 -> 181

## Runs left out by the added rule

* YASAKANI fastest model: scenario_story_en_mystery r0, scenario_story_en_mystery r1
* mkyt v0.3.2 `mkyt_v032`: scenario_story_en_mystery r1
* vLLM 0.31.0 `n12_vl_mtp_lm4`: scenario_story_en_mystery r0, scenario_story_en_mystery r1
* SGLang 0.5.22 `n12_sg_dspark_lm4`: scenario_story_en_mystery r0, scenario_story_en_mystery r1
* llama.cpp b11700 `n12_llama_dflq4`: scenario_story_en_mystery r0, scenario_story_en_mystery r1
* llama.cpp b11700 `n12_llama_mtp2`: scenario_story_en_mystery r0

## The 12 published prompts one by one (thinking off, average of the runs that are kept)

`L` = left out as a repetition loop, `T` = left out because the answer is a tool call nobody asked for; two letters = the two runs.

| Prompt | YASAKANI fastest model | YASAKANI quality model | Fastest other engine | Fastest model / fastest other |
|---|---|---|---|---|
| code_cuda | left out (LL) | 475 | 446 (MegaCapybara 1.39 Tiny) | - |
| code_python | 529 | 514 | 450 (MegaCapybara 1.39 Tiny) | 1.18 |
| code_typescript | left out (TT) | 468 | 533 (MegaCapybara 1.39 Tiny) | - |
| story_en_mystery | left out (LL) | left out (LL) | 262 (mkyt v0.3.2) | - |
| story_zh_dialogue | 192 | 176 | 147 (ignis v0.9.6) | 1.30 |
| story_zh_scifi | 210 | 181 | 177 (mkyt v0.3.2) | 1.19 |
| structured_csv | 813 | 923 | 647 (MegaCapybara 1.39 Tiny) | 1.26 |
| structured_jsonl | 1,220 | 1,132 | 635 (mkyt v0.3.2) | 1.92 |
| structured_sql | 694 | 736 | 533 (mkyt v0.3.2) | 1.30 |
| translation_en_zh | 502 | 477 | 359 (mkyt v0.3.2) | 1.40 |
| translation_markdown | 372 | 376 | 295 (mkyt v0.3.2) | 1.26 |
| translation_zh_en | 608 | 597 | 467 (mkyt v0.3.2) | 1.30 |

The thinking-on tables leave out no run, so they are the ones of `bench/20261108/tables/SPEED_TABLES.md`.
