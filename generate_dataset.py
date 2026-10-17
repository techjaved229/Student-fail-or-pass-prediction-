"""Generate a realistic synthetic student pass/fail dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("data/students.csv")
RANDOM_SEED = 42
TOTAL_RECORDS = 650

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Shaurya", "Atharv", "Dhruv", "Kabir", "Rudra", "Ananya", "Diya", "Aadhya", "Avni", "Myra",
    "Ira", "Anika", "Pari", "Sara", "Kiara", "Navya", "Riya", "Aanya", "Meera", "Saanvi",
    "Rahul", "Rohan", "Kunal", "Nikhil", "Yash", "Aman", "Akash", "Sahil", "Varun", "Harsh",
    "Neha", "Priya", "Pooja", "Sneha", "Kavya", "Nisha", "Simran", "Tanvi", "Isha", "Ritika",
    "Mohit", "Gaurav", "Ankit", "Siddharth", "Abhishek", "Rishi", "Dev", "Lakshya", "Manav", "Aryan",
    "Shreya", "Mansi", "Aditi", "Payal", "Sakshi", "Komal", "Tanya", "Prachi", "Swati", "Jhanvi",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Yadav", "Patel", "Mehta", "Jain", "Mishra",
    "Tiwari", "Pandey", "Chauhan", "Reddy", "Nair", "Iyer", "Das", "Roy", "Khan", "Ansari",
    "Malhotra", "Kapoor", "Bansal", "Agarwal", "Saxena", "Joshi", "Rao", "Pillai", "Menon", "Shetty",
]


def generate_student_names(total_records: int, rng: np.random.Generator) -> list[str]:
    """Create realistic, reproducible names without using generic Student 1 labels."""
    all_names = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    rng.shuffle(all_names)

    if total_records <= len(all_names):
        return all_names[:total_records]

    names = all_names.copy()
    extra_needed = total_records - len(names)
    for index in range(extra_needed):
        names.append(f"{all_names[index % len(all_names)]} {index // len(all_names) + 2}")
    return names


def generate_students(total_records: int = TOTAL_RECORDS, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create reproducible student data with realistic academic variation."""
    rng = np.random.default_rng(random_seed)

    attendance = np.clip(rng.normal(loc=74, scale=16, size=total_records), 0, 100).round(1)
    internal_marks = np.clip(
        attendance * 0.28 + rng.normal(loc=17, scale=8, size=total_records),
        0,
        50,
    ).round(1)
    test_score = np.clip(
        attendance * 0.34 + internal_marks * 0.92 + rng.normal(loc=18, scale=13, size=total_records),
        0,
        100,
    ).round(1)

    academic_score = attendance * 0.30 + internal_marks * 1.00 + test_score * 0.45
    noise = rng.normal(loc=0, scale=8.5, size=total_records)
    pass_threshold = 84
    result = np.where(academic_score + noise >= pass_threshold, "Pass", "Fail")

    df = pd.DataFrame(
        {
            "student_id": [f"ST{index:03d}" for index in range(1, total_records + 1)],
            "student_name": generate_student_names(total_records, rng),
            "attendance": attendance,
            "internal_marks": internal_marks,
            "test_score": test_score,
            "result": result,
        }
    )
    return df


def main() -> None:
    """Generate and save the dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_students()
    df.to_csv(OUTPUT_PATH, index=False)
    pass_count = int((df["result"] == "Pass").sum())
    fail_count = int((df["result"] == "Fail").sum())
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Records: {len(df)} | Pass: {pass_count} | Fail: {fail_count}")


if __name__ == "__main__":
    main()
