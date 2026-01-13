# Task Plan: VoxCPM 1.5B ONNX CPU Inference (Root-Level Project)

## Goal
Deliver a root-level VoxCPM 1.5B ONNX CPU inference project (simple CLI, export/quantize scripts, voice presets + cloning) based on `Text-to-Speech-TTS-ONNX/VoxCPM`.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Research/gather information
- [x] Phase 3: Execute/build
- [x] Phase 4: Review and deliver

## Key Questions
1. How should we package files at the repo root while keeping a simple CLI and export/quantize flow?
2. How should voice presets be defined while still supporting prompt-based cloning?
3. What defaults should be used for CPU inference (threads, paths, config)?

## Decisions Made
- Use the `planning-with-files` workflow with persistent files.

## Errors Encountered
- ModelScope API search endpoint returned HTTP 404 during attempts to locate VoxCPM1.5 ONNX assets.
- GitHub code search API returned HTTP 401 (unauthorized) when searching for 1.5B ONNX filenames.
- Export dependencies failed on x86_64 venv (torch wheel unavailable); resolved by recreating venv with arm64 Python.
- `voxcpm` pip dependency pulled `llvmlite` incompatible with Python 3.11; resolved by using local VoxCPM code and trimming export requirements.

## Status
**Currently complete** - Pipeline runs end-to-end; `scripts/run_service.sh` generates `outputs/demo.wav`.
