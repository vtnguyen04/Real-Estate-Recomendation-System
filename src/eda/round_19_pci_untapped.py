"""
Round 19: PCI (fact_post_contact_interactions) — Untapped Data Source Analysis

Objective: Discover whether PCI table contains signals for currently "blind" test users
who have ZERO contact history in fact_user_events but DO have lead/chat data in PCI.

Key Question: Can PCI data convert blind users → warm users for ALS?

Author: AI Agent (Round 19)
Date: 2026-05-20
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import polars as pl

# ── Config ──
PCI_PATH = "/home/db/rc/datathon/train/fact_post_contact_interactions/**/*.parquet"
TEST_PATH = "/home/db/rc/datathon/test/test_users.parquet"
CACHE_DIR = ".cache"
REPORT_PATH = "src/eda/reports/round_19_report.md"


def main():
    print("=" * 60)
    print("ROUND 22: PCI Data — Untapped Signal for Blind Users")
    print("=" * 60)

    # ── Load data ──
    pci = pl.scan_parquet(PCI_PATH)
    test_users = pl.read_parquet(TEST_PATH)
    contact_users = pl.read_parquet(f"{CACHE_DIR}/contact_pairs.parquet", columns=["user_id"]).unique()

    print(f"\nTest users: {len(test_users):,}")
    print(f"Users with contacts (warm): {len(contact_users):,}")

    # ── 1. PCI overview ──
    print("\n── [1/5] PCI Overview ──")
    pci_stats = pci.select([
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("item_id").n_unique().alias("unique_items"),
        pl.len().alias("total_rows"),
        pl.col("lead_count").sum().alias("total_leads"),
        pl.col("purchased").sum().alias("total_purchased"),
        pl.col("chat_lead").sum().alias("total_chat_leads"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
    ]).collect()
    print(pci_stats)

    # ── 2. Test user coverage ──
    print("\n── [2/5] Test User Coverage in PCI ──")
    pci_test = (
        pci
        .filter(pl.col("user_id").is_in(test_users["user_id"].to_list()))
        .collect()
    )
    pci_test_users = pci_test["user_id"].n_unique()
    print(f"Test users in PCI: {pci_test_users:,} ({pci_test_users/len(test_users)*100:.1f}%)")

    # ── 3. Blind users with PCI data ──
    print("\n── [3/5] Blind Users with PCI Data ──")
    blind_test = test_users.join(contact_users, on="user_id", how="anti")
    print(f"Blind test users (no fact_user_events contacts): {len(blind_test):,}")

    blind_pci = pci_test.filter(pl.col("user_id").is_in(blind_test["user_id"].to_list()))
    blind_pci_users = blind_pci["user_id"].n_unique()
    print(f"Blind users WITH PCI data: {blind_pci_users:,}")
    print(f"PCI rows for blind users: {len(blind_pci):,}")
    print(f"Avg items per blind user in PCI: {len(blind_pci)/max(blind_pci_users,1):.1f}")

    # Signal breakdown
    lead_rows = (blind_pci["lead_count"] > 0).sum()
    chat_rows = (blind_pci["chat_message_count"] > 0).sum()
    purchased_rows = blind_pci["purchased"].sum()
    print(f"\nSignal breakdown for blind users:")
    print(f"  lead_count > 0: {lead_rows:,} rows")
    print(f"  chat_message_count > 0: {chat_rows:,} rows")
    print(f"  purchased = True: {purchased_rows:,} rows")

    # ── 4. PCI as additional ALS signal ──
    print("\n── [4/5] PCI as Additional ALS Training Signal ──")
    pci_lead_pairs = (
        pci
        .filter(pl.col("lead_count") > 0)
        .select(["user_id", "item_id"])
        .unique()
        .collect()
    )
    print(f"Total PCI lead pairs (all users): {len(pci_lead_pairs):,}")
    print(f"PCI lead users: {pci_lead_pairs['user_id'].n_unique():,}")

    # How many NEW pairs vs already in contact_pairs?
    contact_pairs = pl.read_parquet(f"{CACHE_DIR}/als_contact_pairs.parquet", columns=["user_id", "item_id"])
    existing = contact_pairs.select(["user_id", "item_id"]).unique()
    new_pairs = pci_lead_pairs.join(existing, on=["user_id", "item_id"], how="anti")
    print(f"Existing ALS contact pairs: {len(existing):,}")
    print(f"NEW pairs from PCI (not in ALS): {len(new_pairs):,}")
    print(f"New unique users from PCI: {new_pairs['user_id'].n_unique():,}")

    # ── 5. Category & date analysis for blind PCI users ──
    print("\n── [5/5] Category & Date Analysis ──")
    cat_dist = blind_pci.group_by("category").agg(pl.len().alias("n")).sort("n", descending=True)
    print("Category distribution (blind users in PCI):")
    print(cat_dist)

    date_range = blind_pci.select([
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ])
    print(f"\nDate range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")

    # Recency: how many blind users have recent PCI data?
    recent_blind = blind_pci.filter(
        pl.col("date") >= pl.lit("2026-03-01").str.to_date()
    )
    recent_users = recent_blind["user_id"].n_unique()
    print(f"Blind users with PCI data after 2026-03-01: {recent_users:,}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"INS-059: {blind_pci_users:,} blind test users have PCI data (avg {len(blind_pci)/max(blind_pci_users,1):.1f} items)")
    print(f"INS-060: {len(new_pairs):,} NEW (user,item) lead pairs from PCI not in ALS training")
    print(f"→ Can convert {blind_pci_users:,} blind → warm users")
    print(f"→ Can add {len(new_pairs):,} new interaction pairs to ALS matrix")
    print(f"→ Potential warm user increase: {len(contact_users):,} → {len(contact_users) + new_pairs['user_id'].n_unique():,}")


if __name__ == "__main__":
    main()
