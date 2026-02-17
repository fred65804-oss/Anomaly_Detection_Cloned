"""
Test script for Hybrid IDS API
Runs all test cases from test_samples.json
"""

import json
import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def test_single_prediction(test_case):
    """Test single prediction endpoint"""
    response = requests.post(
        f"{API_URL}/predict",
        json=test_case["data"]
    )
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return {"error": response.status_code, "detail": response.text}

def test_batch_prediction(test_cases):
    """Test batch prediction endpoint"""
    samples = [tc["data"] for tc in test_cases]
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"samples": samples}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Hybrid IDS API")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    args = parser.parse_args()
    
    # Update global API URL
    global API_URL
    API_URL = f"http://localhost:{args.port}"
    
    # Load test cases
    test_file = Path(__file__).parent / "test_samples.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    test_cases = data["test_cases"] # Not passing the metadata(which contains whether it is an attack or not)
    
    print("="*70)
    print(f"HYBRID IDS API TESTING (Port: {args.port})")
    print("="*70)
    
    # Test health endpoint
    print("\n[1] Testing Health Endpoint...")
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"  Status: {health['status']}")
        print(f"  Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Make sure API is running: python -m app.main --port {args.port}")
        return
    
    # Test model info
    print("\n[2] Testing Model Info Endpoint...")
    try:
        info = requests.get(f"{API_URL}/model/info").json()
        print(f"  Threshold: {info['threshold']:.3f}")
        print(f"  Supervised Weight: {info['supervised_weight']:.3f}")
        print(f"  Ensemble Method: {info['ensemble_method']}")
        print(f"  Use Autoencoder: {info['use_autoencoder']}")
        print(f"  Anomaly Detectors: {', '.join(info['anomaly_detectors'])}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test single predictions
    print("\n[3] Testing Single Predictions...")
    print("-"*70)
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  Expected: {test_case['expected']}")
        
        result = test_single_prediction(test_case)
        
        if "error" in result:
            print(f"  ERROR: {result}")
        else:
            is_intrusion = result['is_intrusion']
            confidence = result['confidence']
            
            # Check if prediction matches expectation
            expected_intrusion = (test_case['expected'] == 'intrusion')
            match = "✓" if is_intrusion == expected_intrusion else "✗"
            
            alert_level = result.get('alert_level', 'N/A')
            alert_message = result.get('alert_message', '')

            print(f"  Prediction: {'INTRUSION' if is_intrusion else 'NORMAL'} {match}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Alert Level: {alert_level} {alert_message}")

            results.append({
                "name": test_case['name'],
                "expected": test_case['expected'],
                "predicted": "intrusion" if is_intrusion else "normal",
                "confidence": confidence,
                "alert_level" : alert_level,
                "alert_message" : alert_message,
                "correct": is_intrusion == expected_intrusion
            })
    
    # Test batch prediction
    print("\n" + "="*70)
    print("[4] Testing Batch Prediction...")
    print("-"*70)
    
    batch_result = test_batch_prediction(test_cases)
    
    if "error" in batch_result:
        print(f"  ERROR: {batch_result}")
    else:
        print(f"  Total Samples: {batch_result['count']}")
        print(f"  Intrusions Detected: {batch_result['intrusions_detected']}")
        print(f"  Normal Traffic: {batch_result['count'] - batch_result['intrusions_detected']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = (correct / total) * 100

        # Print the distribution of alert levels
        alert_counts = {}
        for r in results:
            level = r.get('alert_level', 'N/A')
            alert_counts[level] = alert_counts.get('level',0) + 1

        print("\n Alert Level Distribution:")
        alert_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL', 'N/A']

        for level in alert_order:
            if level in alert_counts: # If the current level of anomaly is present in the dictionary already 
                print(f"{level} {alert_counts[level]}") # Level and its corresponding count

        print(f"\nTest Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        print("\nDetailed Results:")
        for r in results:
            status = "✓ PASS" if r['correct'] else "✗ FAIL"
            print(f"  {status} - {r['name']}")
            print(f"      Expected: {r['expected']}, Got: {r['predicted']} (conf: {r['confidence']:.4f})")
        
        # Save results
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
