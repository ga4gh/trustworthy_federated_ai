#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Waiting for DRS servers to spin up..."

# Array of all your DRS hosts
hosts="drs-central drs-site-a drs-site-b drs-site-c drs-site-d"

for host in $hosts; do
  echo "Checking $host..."
  until curl --silent --fail --output /dev/null "http://$host:4500/ga4gh/drs/v1/service-info"; do
    printf '.'
    sleep 1
  done
  printf '\n%s is up!\n' "$host"
done

echo "All DRS servers are ready. Starting data seeding..."
# ==========================================
# SITE A DATA SEEDING
# ==========================================
echo "Seeding Site A..."

# Train Object
curl --request POST \
  --url http://drs-site-a:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_a_train",
  "description": "Training split for Site A",
  "name": "train.tsv",
  "is_bundle": false,
  "size": 85000,
  "checksums": [{"checksum": "83709ab2755fed31503cd34bd7a58a69", "type": "md5"}],
  "file_access_objects": [{"path": "/data/train.tsv"}]
}'

# Val Object
curl --request POST \
  --url http://drs-site-a:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_a_val",
  "description": "Validation split for Site A",
  "name": "val.tsv",
  "is_bundle": false,
  "size": 15000,
  "checksums": [{"checksum": "93709ab2755fed31503cd34bd7a58b70", "type": "md5"}],
  "file_access_objects": [{"path": "/data/val.tsv"}]
}'

# ==========================================
# SITE B DATA SEEDING
# ==========================================
echo "Seeding Site B..."

# Train Object
curl --request POST \
  --url http://drs-site-b:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_b_train",
  "description": "Training split for Site B",
  "name": "train.tsv",
  "is_bundle": false,
  "size": 85000,
  "checksums": [{"checksum": "83709ab2755fed31503cd34bd7a58a69", "type": "md5"}],
  "file_access_objects": [{"path": "/data/train.tsv"}]
}'

# Val Object
curl --request POST \
  --url http://drs-site-b:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_b_val",
  "description": "Validation split for Site B",
  "name": "val.tsv",
  "is_bundle": false,
  "size": 15000,
  "checksums": [{"checksum": "93709ab2755fed31503cd34bd7a58b70", "type": "md5"}],
  "file_access_objects": [{"path": "/data/val.tsv"}]
}'

# ==========================================
# SITE C DATA SEEDING
# ==========================================
echo "Seeding Site C..."

# Train Object
curl --request POST \
  --url http://drs-site-c:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_c_train",
  "description": "Training split for Site C",
  "name": "train.tsv",
  "is_bundle": false,
  "size": 85000,
  "checksums": [{"checksum": "83709ab2755fed31503cd34bd7a58a69", "type": "md5"}],
  "file_access_objects": [{"path": "/data/train.tsv"}]
}'

# Val Object
curl --request POST \
  --url http://drs-site-c:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_c_val",
  "description": "Validation split for Site C",
  "name": "val.tsv",
  "is_bundle": false,
  "size": 15000,
  "checksums": [{"checksum": "93709ab2755fed31503cd34bd7a58b70", "type": "md5"}],
  "file_access_objects": [{"path": "/data/val.tsv"}]
}'

# ==========================================
# SITE D DATA SEEDING
# ==========================================
echo "Seeding Site D..."

# Train Object
curl --request POST \
  --url http://drs-site-d:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_d_train",
  "description": "Training split for Site D",
  "name": "train.tsv",
  "is_bundle": false,
  "size": 85000,
  "checksums": [{"checksum": "83709ab2755fed31503cd34bd7a58a69", "type": "md5"}],
  "file_access_objects": [{"path": "/data/train.tsv"}]
}'

# Val Object
curl --request POST \
  --url http://drs-site-d:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "site_d_val",
  "description": "Validation split for Site D",
  "name": "val.tsv",
  "is_bundle": false,
  "size": 15000,
  "checksums": [{"checksum": "93709ab2755fed31503cd34bd7a58b70", "type": "md5"}],
  "file_access_objects": [{"path": "/data/val.tsv"}]
}'

# ==========================================
# CENTRAL SERVER DATA SEEDING (Test Object)
# ==========================================
echo "Seeding Central Server Test Data..."

curl --request POST \
  --url http://drs-central:4501/admin/ga4gh/drs/v1/objects \
  --header 'content-type: application/json' \
  --data '{
  "id": "central_test",
  "description": "Global Test Evaluation Set",
  "name": "test.tsv",
  "is_bundle": false,
  "size": 25000,
  "checksums": [{"checksum": "a3709ab2755fed31503cd34bd7a58c71", "type": "md5"}],
  "file_access_objects": [{"path": "/data/test.tsv"}]
}'

echo "All DRS objects successfully registered!"