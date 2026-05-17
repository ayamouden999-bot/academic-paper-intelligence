"""
test_edge_cases.py — Test guardrails and edge cases.
Run: python test_edge_cases.py
"""

from tools.guardrails import validate_pdf, sanitize_text, PipelineError
from tools.classifier_tool import classify_paper

print("Running edge case tests...\n")

# Test 1: Non-existent file
try:
    validate_pdf("fake_file.pdf")
    print("❌ Test 1 FAILED — should have raised error")
except PipelineError as e:
    print(f"✅ Test 1 PASSED — caught missing file: {e}")

# Test 2: Wrong file type
import tempfile, os
tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
tmp.write(b"not a pdf")
tmp.close()
try:
    validate_pdf(tmp.name)
    print("❌ Test 2 FAILED — should have raised error")
except PipelineError as e:
    print(f"✅ Test 2 PASSED — caught wrong type: {e}")
os.unlink(tmp.name)

# Test 3: Empty text sanitization
result = sanitize_text("  hello   \n\n\n\n  world  ")
assert "hello" in result and "world" in result
print("✅ Test 3 PASSED — text sanitization works")

# Test 4: Long text truncation
long_text = "word " * 5000
result = sanitize_text(long_text, max_chars=100)
assert len(result) <= 120
print("✅ Test 4 PASSED — text truncation works")

# Test 5: Classifier with empty input
result = classify_paper("")
assert "field" in result
print("✅ Test 5 PASSED — classifier handles empty input")

# Test 6: Classifier with gibberish
result = classify_paper("asdfghjkl qwerty zxcvbnm 12345")
assert "field" in result
print("✅ Test 6 PASSED — classifier handles gibberish")

print("\n✅ All edge case tests passed!")
print("These results go in your Week 4 report under 'Evaluation & Robustness'")
