# The converter: from the original weights to a `.yskn` file

日本語：元のモデル（Qwen3.8-27B の BF16 の重み）から `.yskn` ファイルを作るプログラムです。公開している2つのモデルの本体はこの方法で作りました。
手元で変換し直して、公開ファイルと突き合わせられます。私たちが同じPCで変換し直した結果は下の「What we got」にあります。

This folder holds the program that makes a YASAKANI model file (`.yskn`, defined in [docs/FORMAT.md](../../docs/FORMAT.md)) from
the original checkpoint of Qwen3.8-27B, the data it needs, and what we got when we ran it again.

| | |
|---|---|
| [`yskn_convert.py`](yskn_convert.py) | the conversion: `convert` (GPU: statistics and codes), `assemble` (CPU: the file), `compare` (two files, tensor by tensor), `selftest` |
| [`trellis_search.py`](trellis_search.py) | the search for the codes: a call into the CUDA kernel, and a PyTorch version of the same kernel that gives the same symbols |
| [`native/`](native/) | the CUDA kernel of the search (one file) and its CMake file |
| [`make_calibration.py`](make_calibration.py) | the 192 calibration windows as token ids: WikiText-2 made again from the public dataset, and the transcripts below |
| [`data/`](data/) | the two value tables, the number of bits of every matrix of the quality model, the 96 transcripts of the calibration |
| [`records/`](records/) | what we got: the log of the run, and the comparison of the files with the released ones |

## What the conversion does

[docs/TECHNOLOGY.md](../../docs/TECHNOLOGY.md) (section 3.2) explains it in words. In detail, for every matrix W of the 64 layers
(rows = outputs, columns = inputs):

1. **Statistics**, on the original model (bfloat16 weights, the model classes of Hugging Face transformers), over 192 windows of
   2,048 tokens. For every position one token is drawn from the original model's own next-token distribution (seed 2026); the
   loss is the cross-entropy to these tokens. With x the input of the matrix and g the gradient of the loss with respect to its
   output:
   * `H_in0 = mean of x xᵀ`
   * `H_in = mean of |g|² x xᵀ` (input side: every token weighted by how much its output matters)
   * `H_out = mean of w g gᵀ`, with `w = x H_in0 xᵀ` divided by its median over the window, at most 8 (output side)

   The gradient at the output of a layer comes from the layer above, so the layers are done from the top down. The inputs of
   every 8th layer are kept on disk; the 8 layers in between are run again from there.
2. **Coordinates.** W and the statistics are rotated the way the file stores the matrix: a Hadamard transform over blocks of 512
   with fixed signs, on both sides of W (docs/FORMAT.md, section 4).
3. **Rounding.** Every row is divided by its scale (the root mean square of the row) and cut into sequences of 256 weights. The
   code of a sequence is chosen by the Viterbi algorithm (`trellis_search.py`): the symbols whose decoded values are closest to the
   targets in squared error.
   * Input side (columns): the columns are rounded 256 at a time from the left, and the error of a block is passed on to the
     columns not rounded yet, weighted by `H_in` (the update of GPTQ, also known as LDLQ). Inside a block the same feedback runs
     inside the search, over the 8 steps (16 weights) before every step.
   * Output side (rows): the rows are rounded in blocks of 256 (128 for the two matrices with 1,024 rows), from the last block
     upwards, and the error of a block moves the targets of all rows above it, weighted by `H_out`.
   * Damping: 0.03 times the mean of the diagonal is added to `H_in`, 1.0 times to `H_out`.
4. **The file.** The codes, the value table of the bit width and the row scales of every matrix; the 449 small tensors (norms and
   the like) as they are, in bfloat16.

The output matrix (`lm_head`) is rounded with the input side only; its statistic is `mean of x xᵀ` of the output of the final norm
on the same windows. The embedding table is rounded for plain squared error: a lookup has no input to weigh.

Bit widths: a step of two weights stores K = 6 bits (3 bits per weight) or K = 8 bits (4 bits per weight). The fastest model has
K = 6 everywhere in the body. The quality model has K = 8 for 288 matrices and K = 6 for 112 ([`data/bits_quality.json`](data/bits_quality.json)).
Both have K = 8 for the output matrix and K = 6 for the embedding table. All numbers are constants at the top of `yskn_convert.py`.

## Steps

You need Python 3.10 or later, PyTorch with CUDA, transformers 5.5 (it has the model classes of Qwen3.8), numpy and pyarrow, the
original checkpoint `Qwen/Qwen3.8-27B` (about 54 GB), a GPU with 32 GB, at least 64 GB of main memory (our run kept about 53 GB in
use), about 60 GB of disk for the work folder and 10 to 12 GB for every file you assemble.

1. **Build the search kernel** (CMake and the CUDA toolkit; on Windows also the Visual Studio Build Tools, from a
   "x64 Native Tools" prompt):

       cmake -S tools/convert/native -B tools/convert/native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
       cmake --build tools/convert/native/build

   Without it, `--reference-search` uses the PyTorch version of the search: the same symbols, but 70 to 140 times slower in our
   test (days instead of hours for the whole model).
2. **Check** the kernel against the PyTorch version, and a small matrix through the file format and back:

       python tools/convert/yskn_convert.py selftest

3. **The calibration windows** (downloads the WikiText-2 file of 6 MB from the Hugging Face Hub):

       python tools/convert/make_calibration.py --tokenizer <folder of the original checkpoint>

   It checks every step against the SHA-256 of what we got and says whether the windows are the same as ours.
4. **Convert** (GPU; resumable: run the same command again after a stop):

       python tools/convert/yskn_convert.py convert --checkpoint <folder of the original checkpoint> --work <work folder>

   By default every body matrix is rounded at both bit widths, so that both models can be assembled from one run.
5. **Assemble** (CPU):

       python tools/convert/yskn_convert.py assemble --checkpoint <folder> --work <work folder> --bits tools/convert/data/bits_quality.json --out quality.yskn
       python tools/convert/yskn_convert.py assemble --checkpoint <folder> --work <work folder> --bits 6 --out fastest.yskn

   `--head-from <released .yskn>` takes the output matrix from a released file instead (see "How the released files were made").
6. **Compare** with a released file, symbol by symbol:

       python tools/convert/yskn_convert.py compare quality.yskn qwen38_27b_quality.yskn

   The quality can be measured with [`tools/quality/`](../quality/), the weights against the original with `tools/yskn_decode.py compare`.

## What we got

On November 14, 2026 we ran the steps above on the PC that made the released files (the software is listed under "Limits"),
with the CUDA kernel. The records are in [`records/`](records/): the log of the run (`convert_log.txt`), the settings, versions and
the rounding error of every matrix at both bit widths (`convert.json`), the four assembled files (`assemble_*.json`), their
comparison with the released files tensor by tensor (`compare_*.json`, `head_compare.json`) and the quality of the two models that
have the output matrix of this program (`score_*.json`).

| Step | Result |
|---|---|
| `make_calibration.py` | the same 192 windows as in the conversion of the released files (every check "yes") |
| `convert`, both bit widths | 1 hour 55 minutes: the embedding table 12 s, the 64 layers 114 minutes (their statistics, then every matrix at 3 and at 4 bits per weight), the output matrix 27 s. About 53 GB of main memory in use, 58 GB in the work folder |
| the 400 matrices of the body | the same symbols and row scales as the released files, at both bit widths: 0 of 12,163,481,600 symbols different in either model |
| the embedding table | the same as in the released files: 0 of 635,699,200 symbols different |
| `assemble --head-from` the released file | **both files identical to the released ones, byte for byte**: SHA-256 `6493983c…d18b` (quality model) and `3ba17a76…d1ca` (fastest model) |
| the output matrix of this program | 48.5% of its symbols differ from the released one, because it is made from another statistic (see the next section) |
| the whole model from this program, output matrix included | quality model: KL divergence 0.011872, top-1 agreement 95.577%, perplexity 2.0744 (the released file: 0.012591, 95.347%, 2.0686). Fastest model: 0.030040, 93.270%, 2.0817 (released: 0.030890, 92.981%, 2.0759). Measured with [`tools/quality/`](../quality/): KV5 cache, the reference of the quality table, 20 windows |

So the program makes the released files again, except for the output matrix, which has to be taken from a released file to get
the same bytes. The output matrix it makes itself gives a slightly lower KL divergence and a slightly higher perplexity on the
evaluation windows; we have not measured more than that.

## How the released files were made, and where this program differs

The released files were not made by this one program but by research scripts, between October 6 and November 7, 2026. This
program is those steps, cleaned up and put together. Part by part:

| Part | How the released files have it | This program |
|---|---|---|
| the 400 matrices of the body | made on November 7, 2026 with the method above (the same windows, the same statistics, the same rounding) | the same |
| the embedding table | made on October 11, 2026 by an earlier encoder of the same kind (plain squared error) | the same (identical, see above) |
| the output matrix | made on October 8, 2026 by the earlier encoder, with the input statistic of an **earlier conversion**: all matrices at 3 bits, 128 WikiText windows, the statistic taken behind that compressed body | the input statistic of the **original** model on the 192 windows. Given the statistic of the released one (`--head-stat`; that file is not published), this program differs from the released output matrix in 0.011% of the symbols, from the arithmetic of the two encoders. `--head-from` copies the released one |
| which matrices get 4 bits | decided on October 21 and 29, 2026, in two steps: first the matrices that a published quantization of this model (GSQ-RCO, IQ3_S) stores at 4 bits, then the order of its published bit allocations, filled up to a size limit, one bit width for the matrices that share an input. The allocation is not a measurement of ours | `data/bits_quality.json`, the result |
| the two value tables | learned on October 6, 2026 from Gaussian data (the same for every matrix of a bit width; nothing is fitted to the model) | `data/value_table_k6.bin`, `data/value_table_k8.bin`, the result |
| the small tensors | the original bfloat16 values | the same |

## Limits

* **The same file only with the same arithmetic.** The codes follow from the statistics, and the statistics from the order of
  the floating-point operations of the forward and backward passes. Another GPU, driver, PyTorch or transformers version, or another
  batch size changes that order. In one test, batching the gradient pass 4 windows at a time instead of 1 changed 12% of the
  symbols of a matrix in the top layer. On another system, expect codes of the same kind and a quality close to ours, not the same
  file. We have run it on one system only (below).
* **Tested on one system**: Windows 11, one RTX 5090 (32 GB), 93 GB of main memory, Python 3.14.4, numpy 2.4.4, PyTorch 2.10.0+cu128,
  transformers 5.5.0, CUDA 13.2 for the kernel. Not tested on Linux or on other GPUs; the kernel needs 75,152 bytes of shared memory
  per block.
* **Qwen3.8-27B only.** The program knows the layers and matrices of this model. Another model needs changes in the lists at the top of
  `yskn_convert.py` and an engine that runs it.
* **The PyTorch search is slow.** It exists to read and to check the kernel.

## Data and licences

* The program and the kernel: Apache License 2.0, as the rest of this repository.
* `data/calibration_transcripts.json`: 96 windows of a prompt and the beginning of what the model wrote for it (the thinking comes
  first), 50 in Japanese and 46 in English. The prompts were written for this purpose; the answers were generated on October 15, 2026
  by an earlier compressed version of Qwen3.8-27B (all matrices at 3 bits per weight), temperature 1.0, top-p 0.95, top-k 20. None of
  the prompts is among the 20 evaluation windows of `tools/quality`.
* WikiText-2 (Merity et al., 2016; Creative Commons Attribution-ShareAlike 3.0) is **not** included. `make_calibration.py` downloads it
  from the dataset `Salesforce/wikitext` at a fixed revision and checks it by SHA-256.
* The value tables and `bits_quality.json` are data of this project (Apache License 2.0).
