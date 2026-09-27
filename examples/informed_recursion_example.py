#!/usr/bin/env python3
"""Example: Using Informed Recursion with Query-Structure Routing (M1) and Targeted Verification (M2).

This script demonstrates how to instantiate an RLM with Informed Recursion enabled.
Requires OPENAI_API_KEY (or compatible endpoint configuration) in your environment:
    export OPENAI_API_KEY="your-api-key"
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm import RLM

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable before running this example.")
        return

    # 1. Instantiate RLM with Informed Recursion enabled
    # enable_m1=True: dynamically assesses query structure and suggests recursion depth.
    # enable_m2=True: active contract verification intercept before answer commitment.
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
            "api_key": api_key,
            "base_url": os.getenv("OPENAI_BASE_URL", None),
        },
        environment="local",
        max_depth=2,
        max_iterations=30,
        enable_m1=True,
        enable_m2=True,
        m1_mode="rules",
        paper_mode=True,
        verbose=True,
    )

    # 2. Example long context
    context = """
    Document 1: Patient ID 4022 visited Clinic A on March 12, 2024. Blood pressure was 120/80.
    Document 2: Patient ID 5190 visited Clinic B on April 04, 2024. Blood pressure was 140/90.
    Document 3: Patient ID 4022 had a follow-up on May 15, 2024. Prescribed Medication X.
    Document 4: Patient ID 7711 visited Clinic A on June 01, 2024. Blood pressure was 118/75.
    Document 5: Patient ID 5190 had a follow-up on July 20, 2024. Prescribed Medication Y.
    """

    query = "Find all patients who visited Clinic A and state whether they were prescribed any medication."

    print("Running Informed Recursion query...")
    completion = rlm.completion(
        prompt=context,
        root_prompt=query,
    )

    print("\n" + "=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)
    print(completion.response)

    if completion.metadata and "m1_m2" in completion.metadata:
        meta = completion.metadata["m1_m2"]
        print("\n" + "=" * 60)
        print("INFORMED RECURSION METADATA:")
        print("=" * 60)
        print(f"M1 Dynamic Depth: {meta.get('dynamic_depth')}")
        print(f"M2 Intercept Triggered: {meta.get('m2_fired')}")
        print(f"M2 Changed Answer: {meta.get('m2_changed_answer')}")

if __name__ == "__main__":
    main()
