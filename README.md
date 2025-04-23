# OSS Threat Data

A curated dataset of real-world open-source software (OSS) supply chain threat incidents, labeled using the AV-xxx taxonomy.

## Dataset Overview

The dataset includes incidents with the following fields:

- `id`: Unique identifier for the incident.
- `title`: Brief title describing the incident.
- `description`: Detailed explanation of the incident.
- `label`: AV-xxx taxonomy code categorizing the threat.
- `source`: URL to the original source of the incident report.

## Sample Entry

| id    | title                                       | description                                      | label  | source                                      |
|-------|---------------------------------------------|--------------------------------------------------|--------|---------------------------------------------|
| gh001 | Malicious dependency mimics popular package | The package `requests-plus` mimics the well-known `requests` library and steals environment variables upon installation. | AV-200 | https://github.com/example/repo/issues/101 |

## Usage

To load the dataset in Python:

```python
import pandas as pd

df = pd.read_csv('data/oss_threat_dataset_updated.csv')
