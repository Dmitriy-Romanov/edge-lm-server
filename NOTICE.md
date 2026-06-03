# Notices

This project is a Rust launcher and local OpenAI-compatible server wrapper for
using TheStageAI Edge-LM models with Pi Agent.

## TheStageAI Edge-LM

The launcher installs and uses the Python package and models from
[TheStageAI/edge-lm](https://github.com/TheStageAI/edge-lm).

The `edge-lm` source repository is released under the MIT License:

> Copyright (c) 2026 thestage.ai labs

The MIT permission notice is preserved in this repository's `LICENSE` file.

## Gemma model terms

The Edge-LM model weights are derivatives of Google's Gemma models. The
upstream Edge-LM README notes that the compressed model weights are additionally
subject to Google's Gemma Terms of Use:

https://ai.google.dev/gemma/terms

This launcher does not redistribute model weights. It downloads them into the
local runtime cache when the server is first started.
