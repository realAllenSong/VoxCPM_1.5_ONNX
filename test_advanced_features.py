#!/usr/bin/env python3
"""
Verification tests for VoxCPM 1.5B advanced streaming features.

Tests:
1. TTFB improvement - streaming first byte latency
2. Multi-sentence consistency - PromptCache functionality  
3. Badcase auto-retry - synthesize_with_retry
"""

import sys
import time
import json

# Add project to path
sys.path.insert(0, "/Users/songallen/Desktop/ONNX_Lab")

def test_ttfb():
    """Test 1: TTFB (Time To First Byte) for streaming endpoint"""
    import requests
    
    print("=" * 60)
    print("TEST 1: TTFB (Time To First Byte) Streaming Test")
    print("=" * 60)
    
    url = "http://127.0.0.1:8000/synthesize-stream"
    payload = {"text": "你好，这是一个流式输出测试。", "voice": "trump_promptvn"}
    
    start = time.time()
    first_chunk_time = None
    total_bytes = 0
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                    print(f"  First chunk received at: {first_chunk_time:.3f}s")
                total_bytes += len(chunk)
        
        total_time = time.time() - start
        print(f"  Total bytes: {total_bytes}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  TTFB: {first_chunk_time:.3f}s")
        
        if first_chunk_time < 30:
            print(f"✓ TTFB Test PASSED ({first_chunk_time:.2f}s < 30s threshold)")
            return True
        else:
            print(f"✗ TTFB Test FAILED ({first_chunk_time:.2f}s >= 30s)")
            return False
            
    except Exception as e:
        print(f"✗ TTFB Test ERROR: {e}")
        return False


def test_prompt_cache():
    """Test 2: PromptCache for multi-sentence consistency"""
    print()
    print("=" * 60)
    print("TEST 2: PromptCache Multi-sentence Consistency Test")
    print("=" * 60)
    
    try:
        from engines.voxcpm_15b import VoxCPM15BEngine, PromptCache
        
        # Load config
        with open("/Users/songallen/Desktop/ONNX_Lab/config.json") as f:
            config = json.load(f)
        
        print("  Loading engine...")
        engine = VoxCPM15BEngine(
            models_dir=config.get("models_dir", "models/onnx_models"),
            voxcpm_dir=config.get("voxcpm_dir", "models/VoxCPM1.5"),
            voices_file=config.get("voices_file", "voices.json"),
        )
        
        # Test build_prompt_cache
        print("  Building prompt cache...")
        prompt_audio = "/Users/songallen/Desktop/ONNX_Lab/reference/trump_promptvn.wav"
        prompt_text = "In short, we embarked on a mission to make America great again for all Americans."
        
        cache = engine.build_prompt_cache(prompt_audio, prompt_text)
        
        # Verify cache structure
        assert isinstance(cache, PromptCache), "Cache should be PromptCache instance"
        assert cache.audio_feat is not None, "audio_feat should not be None"
        assert cache.feat_cond is not None, "feat_cond should not be None"
        assert cache.prompt_embed is not None, "prompt_embed should not be None"
        assert cache.prompt_text_len > 0, "prompt_text_len should be positive"
        assert cache.use_prompt_audio == True, "use_prompt_audio should be True"
        
        print(f"    ✓ cache.audio_feat: OK")
        print(f"    ✓ cache.feat_cond: OK")
        print(f"    ✓ cache.prompt_embed: OK")
        print(f"    ✓ cache.prompt_text_len: {cache.prompt_text_len}")
        print(f"    ✓ cache.use_prompt_audio: {cache.use_prompt_audio}")
        
        # Test merge_prompt_cache
        print("  Testing cache merge...")
        new_cache = engine.merge_prompt_cache(cache, [1, 2, 3])
        assert len(new_cache.generated_audio_feats) == 3, "Should have 3 features"
        
        new_cache = engine.merge_prompt_cache(new_cache, [4, 5])
        assert len(new_cache.generated_audio_feats) == 5, "Should have 5 features"
        print(f"    ✓ merge_prompt_cache: 3 + 2 = {len(new_cache.generated_audio_feats)} features")
        
        print("✓ PromptCache Test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ PromptCache Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retry_mechanism():
    """Test 3: synthesize_with_retry badcase handling"""
    print()
    print("=" * 60)
    print("TEST 3: synthesize_with_retry Badcase Retry Test")
    print("=" * 60)
    
    try:
        from engines.voxcpm_15b import VoxCPM15BEngine
        
        with open("/Users/songallen/Desktop/ONNX_Lab/config.json") as f:
            config = json.load(f)
        
        print("  Loading engine...")
        engine = VoxCPM15BEngine(
            models_dir=config.get("models_dir", "models/onnx_models"),
            voxcpm_dir=config.get("voxcpm_dir", "models/VoxCPM1.5"),
            voices_file=config.get("voices_file", "voices.json"),
        )
        
        # Test _estimate_text_duration
        print("  Testing duration estimation...")
        duration1 = engine._estimate_text_duration("你好世界")  # 4 Chinese chars
        duration2 = engine._estimate_text_duration("Hello world test")  # 3 English words
        print(f"    Chinese '你好世界': {duration1:.2f}s estimated")
        print(f"    English 'Hello world test': {duration2:.2f}s estimated")
        
        assert duration1 > 0, "Duration should be positive"
        assert duration2 > 0, "Duration should be positive"
        
        # Test synthesize_with_retry
        print("  Testing synthesize_with_retry...")
        start = time.time()
        audio, sr = engine.synthesize_with_retry(
            texts=["测试"],
            voice="trump_promptvn",
            max_retries=2,
            length_ratio_threshold=10.0,  # High threshold - should pass first try
        )
        elapsed = time.time() - start
        
        duration = len(audio) / sr
        print(f"    Audio duration: {duration:.2f}s")
        print(f"    Sample rate: {sr}")
        print(f"    Synthesis time: {elapsed:.2f}s")
        
        assert len(audio) > 0, "Audio should not be empty"
        assert sr == 44100, "Sample rate should be 44100"
        
        print("✓ Retry Mechanism Test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Retry Mechanism Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("=" * 60)
    print("VoxCPM 1.5B Advanced Features Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: TTFB
    results.append(("TTFB Streaming", test_ttfb()))
    
    # Test 2: PromptCache
    results.append(("PromptCache", test_prompt_cache()))
    
    # Test 3: Retry Mechanism
    results.append(("Retry Mechanism", test_retry_mechanism()))
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print()
    print(f"Total: {passed}/{len(results)} tests passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
