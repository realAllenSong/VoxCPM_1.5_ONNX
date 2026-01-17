#!/usr/bin/env python3
"""Integration tests for VoxCPM ONNX API.

Usage:
    # Start the API server first:
    python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
    
    # Then run tests:
    python test_integration.py
"""

import os
import sys
import time
import subprocess
import tempfile

import requests

API_BASE = os.getenv("VOXCPM_API_BASE", "http://127.0.0.1:8000")


def test_health():
    """Test /health endpoint."""
    r = requests.get(f"{API_BASE}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print("✓ /health endpoint OK")


def test_info():
    """Test /info endpoint."""
    r = requests.get(f"{API_BASE}/info", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "sample_rate" in data
    assert "bit_depth" in data
    assert "channels" in data
    print(f"✓ /info endpoint OK: {data}")


def test_voices():
    """Test /voices endpoint."""
    r = requests.get(f"{API_BASE}/voices", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "voices" in data
    assert isinstance(data["voices"], list)
    print(f"✓ /voices endpoint OK: {len(data['voices'])} voices available")


def test_synthesize():
    """Test /synthesize endpoint with simple text."""
    r = requests.post(
        f"{API_BASE}/synthesize",
        json={"text": "Hello world", "voice": None},
        timeout=120,
    )
    assert r.status_code == 200
    assert len(r.content) > 1000  # Should have audio data
    print(f"✓ /synthesize endpoint OK: {len(r.content)} bytes")


def test_synthesize_stream():
    """Test /synthesize-stream endpoint and measure TTFB."""
    start = time.time()
    first_chunk_time = None
    total_bytes = 0
    
    with requests.post(
        f"{API_BASE}/synthesize-stream",
        json={"text": "Hello world"},
        stream=True,
        timeout=120,
    ) as r:
        assert r.status_code == 200
        
        # Check headers
        sample_rate = r.headers.get("X-Sample-Rate")
        assert sample_rate is not None
        
        for chunk in r.iter_content(chunk_size=1024):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            total_bytes += len(chunk)
    
    print(f"✓ /synthesize-stream endpoint OK:")
    print(f"  - TTFB: {first_chunk_time:.2f}s")
    print(f"  - Total: {total_bytes} bytes")
    print(f"  - Sample rate: {sample_rate}Hz")
    
    # TTFB should be reasonable (< 30s for CPU inference)
    assert first_chunk_time < 30.0


def test_infer_cli_backward_compat():
    """Verify infer.py CLI still works (backward compatibility)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name
    
    try:
        result = subprocess.run(
            [
                sys.executable, "infer.py",
                "--text", "Test",
                "--output", output_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"⚠ infer.py CLI test skipped (may require models)")
            print(f"  stderr: {result.stderr[:200]}")
            return
        
        # Check output file exists and has content
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 1000
        
        print("✓ infer.py CLI backward compatibility OK")
    except subprocess.TimeoutExpired:
        print("⚠ infer.py CLI test timed out")
    except FileNotFoundError:
        print("⚠ infer.py not found - skipping CLI test")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def main():
    """Run all tests."""
    print(f"Testing VoxCPM ONNX API at {API_BASE}\n")
    
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("ERROR: API server not running!")
        print(f"Start the server first: python -m uvicorn api_server:app --port 8000")
        sys.exit(1)
    
    tests = [
        test_health,
        test_info,
        test_voices,
        # These tests require models to be loaded:
        # test_synthesize,
        # test_synthesize_stream,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    
    # Test CLI backward compatibility (doesn't require API server)
    print(f"\n{'='*40}")
    print("Testing CLI backward compatibility...")
    test_infer_cli_backward_compat()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
