#!/usr/bin/env python3
"""Test script for 0.5B engine directly.

Usage:
    uv run python test_05b.py
"""

import json
import os
import sys

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines import load_engine

def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    model_size = config.get("model_size", "1.5b").lower()
    
    if model_size in ("0.5b", "05b"):
        models_dir = config.get("models_dir_05b", "models/onnx_models_05b")
        voxcpm_dir = config.get("voxcpm_dir_05b", "models/VoxCPM0.5")
        # Use 1.5B tokenizer since they're identical
        if not os.path.exists(os.path.join(voxcpm_dir, "tokenizer.json")):
            voxcpm_dir = config.get("voxcpm_dir", "models/VoxCPM1.5")
    else:
        models_dir = config.get("models_dir", "models/onnx_models")
        voxcpm_dir = config.get("voxcpm_dir", "models/VoxCPM1.5")
    
    voices_file = config.get("voices_file", "voices.json")
    if not os.path.isabs(voices_file):
        voices_file = os.path.join(os.path.dirname(__file__), voices_file)
    
    print(f"Loading {model_size.upper()} model from {models_dir}...")
    print(f"Tokenizer from: {voxcpm_dir}")
    
    try:
        engine = load_engine(
            model_size=model_size,
            models_dir=models_dir,
            voxcpm_dir=voxcpm_dir,
            voices_file=voices_file,
            max_threads=config.get("max_threads", 0),
            text_normalizer=config.get("text_normalizer", True),
            audio_normalizer=config.get("audio_normalizer", False),
        )
        print(f"✓ Engine loaded successfully!")
        print(f"  Sample rate: {engine.sample_rate}Hz")
    except Exception as e:
        print(f"✗ Failed to load engine: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    texts = config.get("text", ["你好，这是 0.5B 测试。"])
    if isinstance(texts, str):
        texts = [texts]
    
    voice = config.get("voice")
    output = config.get("output", "outputs/demo_05b.wav")
    
    print(f"\nSynthesizing with voice: {voice}")
    print(f"Texts: {texts}")
    
    import time
    import soundfile as sf
    
    start = time.time()
    try:
        audio_out, sample_rate = engine.synthesize(
            texts=texts,
            voice=voice,
            cfg_value=config.get("cfg_value"),
            fixed_timesteps=config.get("fixed_timesteps"),
            seed=config.get("seed"),
        )
        
        elapsed = time.time() - start
        duration = len(audio_out) / sample_rate
        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        sf.write(output, audio_out, sample_rate, format="WAV", subtype="PCM_16")
        
        print(f"\n✓ Synthesis complete!")
        print(f"  Output: {output}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {elapsed/duration:.3f}")
        
    except Exception as e:
        print(f"✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
