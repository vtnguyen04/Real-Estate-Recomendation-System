import sys
import polars as pl
import os

CACHE_DIR = '.cache'
TRAIN_PATH = '/home/db/rc/datathon/train/'

print("Starting extraction of cold user prefs...")

# Warm users = users in contact_pairs
contacts = pl.read_parquet(os.path.join(CACHE_DIR, 'contact_pairs.parquet'))
warm_users = set(contacts['user_id'].unique().to_list())
print(f'Warm users: {len(warm_users):,}')

# Cold user prefs from pageview pairs + dim_listing join
pv = pl.read_parquet(os.path.join(CACHE_DIR, 'als_pageview_pairs.parquet'))
listing = pl.scan_parquet(os.path.join(TRAIN_PATH, 'dim_listing/*.parquet')).select(
    ['item_id', 'city_name', 'category']
).collect()

# Filter to cold users only, join with listing for city/category
cold_pv = pv.filter(~pl.col('user_id').is_in(list(warm_users)))
print(f'Cold user pageview pairs: {len(cold_pv):,}')

cold_pv_enriched = cold_pv.join(listing, on='item_id', how='left')
cold_prefs = (
    cold_pv_enriched
    .group_by('user_id')
    .agg([
        pl.col('city_name').drop_nulls().mode().first().alias('pref_city'),
        pl.col('category').drop_nulls().mode().first().alias('pref_cat'),
    ])
    .filter(pl.col('pref_city').is_not_null() | pl.col('pref_cat').is_not_null())
)

out = os.path.join(CACHE_DIR, 'cold_user_prefs.parquet')
cold_prefs.write_parquet(out)
print(f'Cold user prefs saved: {len(cold_prefs):,} users, {os.path.getsize(out)/1e6:.1f}MB')
print(cold_prefs.head(5))
