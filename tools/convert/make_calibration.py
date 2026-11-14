#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""make_calibration.py - the 192 calibration windows of the conversion, as token ids (data/calibration_ids.npy).

  python make_calibration.py --tokenizer DIR [--wikitext FILE.parquet] [--out data/calibration_ids.npy]

Rows 0 to 95: WikiText-2 (raw, train split), made again here from the public dataset, because its text is not ours to
redistribute (Creative Commons BY-SA 3.0; Merity et al., 2016).  Rows 96 to 191: the transcripts of data/calibration_transcripts.json.

The WikiText rows are made as the conversion made them in September 2026:
  * the file wikitext-2-raw-v1/train-00000-of-00001.parquet of the dataset Salesforce/wikitext at revision
    b08601e04326c79dfdd32d625aee71d232d685c3 (SHA-256 below; downloaded with huggingface_hub when --wikitext is not given);
  * all rows of its "text" column joined with nothing in between;
  * the tokenizer of Qwen/Qwen3.8-27B (the folder of the original checkpoint), without special tokens;
  * 128 start positions numpy.random.default_rng(20260922).integers(0, number of tokens - 2048, size=128); a window is the 2,048
    tokens from a start.  The conversion used the first 96.
Every step is checked against the SHA-256 of what we got.  Another version of the tokenizer library could give other ids; then
the check of the WikiText rows fails and says so.

Needs numpy, pyarrow and transformers (for the tokenizer).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET, DATA_REVISION = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"
PARQUET = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
PARQUET_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
SEED, WINDOW, STARTS, USED = 20260922, 2048, 128, 96
EXPECT = dict(
    tokens=2504609,
    wikitext_rows_int32="8d7ce39e81f7c9fe0c93919685e38ec5cf26b663795f515967305440bd86a256",
    transcripts_int32="381a9e53904c45d9d40570ad636bb3f46cef4aa619d4f47cd180301fc1479f5f",
    all_int32="ed82db6b45451ab1f09ddba7af4d209f35cdf2a29c6e8d09c87431c7518e07b7",
)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", required=True, help="the folder of the original checkpoint (tokenizer.json etc.)")
    ap.add_argument("--wikitext", help="the train parquet file; downloaded from the Hugging Face Hub when not given")
    ap.add_argument("--transcripts", default=str(HERE / "data" / "calibration_transcripts.json"))
    ap.add_argument("--out", default=str(HERE / "data" / "calibration_ids.npy"))
    a = ap.parse_args()
    path = a.wikitext
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=DATA_REVISION)
    if sha_file(path) != PARQUET_SHA256:
        raise SystemExit("%s: not the WikiText file of the conversion (SHA-256 differs)" % path)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    text = "".join(rows)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False, use_fast=True)
    ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype="<i4")
    print("WikiText-2 train: %d rows, %d tokens" % (len(rows), ids.size))
    starts = np.random.default_rng(SEED).integers(0, ids.size - WINDOW, size=STARTS)[:USED]
    wiki = np.stack([ids[s:s + WINDOW] for s in starts]).astype("<i4")
    ok_tokens = ids.size == EXPECT["tokens"]
    ok_wiki = hashlib.sha256(wiki.tobytes()).hexdigest() == EXPECT["wikitext_rows_int32"]
    doc = json.loads(Path(a.transcripts).read_text(encoding="utf-8"))
    own = np.asarray([w["ids"] for w in doc["windows"]], dtype="<i4")
    ok_own = own.shape == (USED, WINDOW) and hashlib.sha256(own.tobytes()).hexdigest() == EXPECT["transcripts_int32"]
    allw = np.concatenate([wiki, own])
    ok_all = hashlib.sha256(allw.tobytes()).hexdigest() == EXPECT["all_int32"]
    print("number of tokens as in the conversion:     %s" % ("yes" if ok_tokens else "NO (%d)" % ids.size))
    print("WikiText rows as in the conversion:        %s" % ("yes" if ok_wiki else "NO"))
    print("transcript rows as in the conversion:      %s" % ("yes" if ok_own else "NO"))
    print("all 192 rows as in the conversion:         %s" % ("yes" if ok_all else "NO"))
    np.save(a.out, allw)
    print("written: %s (%s)" % (a.out, "the same windows as the released models" if ok_all else
                                "OTHER windows than the released models: a conversion with them will give other codes"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
