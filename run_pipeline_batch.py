#!/usr/bin/env python3
"""
Automate running pipeline_v2.py for multiple file pairs
"""
import subprocess
import sys
from pathlib import Path

# Define file pairs
FILE_PAIRS = [
    (97, 81),
    (97, 52),
    (89, 68),
    (79, 80),
    (61, 71),
    (90, 98),
    (88, 59)
]

# Base paths
CSV_DIR = Path("datasets/rawdata_csv_samples")
#SEMANTIC_DIR = Path("datasets/semantic_models")
RESULTS_DIR = Path("results")

def run_pipeline(src_id, tgt_id):
    """Run pipeline for a specific file pair"""
    # Format IDs with leading zeros
    src = f"{src_id:04d}"
    tgt = f"{tgt_id:04d}"
    
    # Build command
    cmd = [
        "python3", "pipeline_v2.py",
        "--sources", str(CSV_DIR / f"{src}.csv"),
        "--targets", str(CSV_DIR / f"{tgt}.csv"),
        #"--semantic", 
        #str(SEMANTIC_DIR / f"{src}.ttl"),
        #str(SEMANTIC_DIR / f"{tgt}.ttl"),
        "--out", str(RESULTS_DIR / f"results_{src_id}_{tgt_id}.json")
    ]
    
    print(f"\n{'='*60}")
    print(f"Processing pair: {src_id} -> {tgt_id}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Successfully completed pair ({src_id}, {tgt_id})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing pair ({src_id}, {tgt_id}): {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Error: pipeline_v2.py not found")
        return False

def main():
    """Run pipeline for all file pairs"""
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting pipeline automation for {len(FILE_PAIRS)} file pairs...")
    
    results = []
    for src_id, tgt_id in FILE_PAIRS:
        success = run_pipeline(src_id, tgt_id)
        results.append((src_id, tgt_id, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total pairs processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed pairs:")
        for src_id, tgt_id, success in results:
            if not success:
                print(f"  - ({src_id}, {tgt_id})")
        sys.exit(1)
    else:
        print("\n✓ All pairs processed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()