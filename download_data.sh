#!/bin/bash

# Install the Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Sync the data from the Hugging Face bucket to the lexica directory
hf sync hf://buckets/ecaplan/splits-storage lexica/
