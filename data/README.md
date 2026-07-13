# Dataset Extraction & Reproducibility

This document details the exact procedures used to extract, merge, and download the clinical notes dataset required for this project. Our pipeline relies on two primary data sources: the MIMIC-IV clinical database and the MIMIC-IV-Ext-BHC dataset.

## 1. Unstructured Text Source (MIMIC-IV-Ext-BHC)

**Dataset Reference:** [MIMIC-IV-Ext-BHC v1.2.0](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/)

Because this dataset cannot be directly queried via Google BigQuery, it must be downloaded to your local environment or Google Colab instance using `wget`.

### Download Instructions

Run the following command in your terminal or Colab notebook. You will be prompted to enter your PhysioNet credentials.

```bash
wget -r -N -c -np --user <your_physionet_username> --ask-password https://physionet.org/files/labelled-notes-hospital-course/1.2.0/
```
*Note: Ensure you have signed the Data Use Agreement (DUA) for MIMIC-IV on PhysioNet before attempting to download.*

## 2. Structured Cohort Extraction (MIMIC-IV v3.1)

**Dataset Reference:** [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)

To accurately label our target cohort (Joint Arthroplasty patients) and establish the 30-day readmission ground truth, we executed a SQL query on Google Cloud BigQuery.

### Extraction Environment
- **Platform:** Google Cloud BigQuery
- **Access Setup:** You must have a signed PhysioNet DUA for MIMIC-IV and link your PhysioNet account to your own Google Cloud project. For official instructions on adding the MIMIC-IV dataset to your BigQuery workspace, refer to the **[MIMIC Cloud Access Guide](https://mimic.mit.edu/docs/gettingstarted/cloud/)**.

### SQL Query

The following SQL query extracts adult patients who underwent specific joint arthroplasty procedures (identified via ICD-9/ICD-10 codes), calculates whether they were readmitted within 30 days, and explicitly joins with the `discharge` note table to retrieve the exact `note_id`. This `note_id` is essential for performing a precise 1-to-1 merge with the unstructured text downloaded in step 1.

```sql
-- Step 1: Isolate the surgery cohort
WITH surgery_cohort AS (
  SELECT 
    p.subject_id,
    p.hadm_id,
    a.admittime,
    a.dischtime
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd` p
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON p.subject_id = a.subject_id
    AND p.hadm_id = a.hadm_id
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt
    ON p.subject_id = pt.subject_id
  WHERE 
    pt.anchor_age >= 18
    AND a.hospital_expire_flag = 0
    AND (
      (p.icd_version = 10 AND p.icd_code LIKE '0SR9%')
      OR (p.icd_version = 10 AND p.icd_code LIKE '0SRB%')
      OR (p.icd_version = 10 AND p.icd_code LIKE '0SRC%')
      OR (p.icd_version = 10 AND p.icd_code LIKE '0SRD%')
      OR (p.icd_version = 10 AND p.icd_code LIKE '0SRF%')
      OR (p.icd_version = 10 AND p.icd_code LIKE '0SRG%')
      OR (p.icd_version = 9 AND p.icd_code IN ('8151','8152','8153'))
      OR (p.icd_version = 9 AND p.icd_code IN ('8154'))
    )
),

-- Step 2: Compute 30-day readmission flag
readmission AS (
  SELECT 
    s.subject_id,
    s.hadm_id,
    s.admittime,
    s.dischtime,
    CASE 
      WHEN MIN(a2.admittime) IS NOT NULL 
      THEN 1 ELSE 0 
    END AS readmitted_30day
  FROM surgery_cohort s
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a2
    ON s.subject_id = a2.subject_id
    AND a2.admittime > s.dischtime
    AND DATETIME_DIFF(a2.admittime, s.dischtime, DAY) <= 30
  GROUP BY 
    s.subject_id, s.hadm_id, s.admittime, s.dischtime
),

-- Step 3: Extract exact note_id mapping
notes AS (
  SELECT 
    hadm_id,
    note_id
  FROM `physionet-data.mimiciv_note.discharge`
)

-- Final Output Selection
SELECT 
  r.subject_id,
  r.hadm_id,
  r.admittime,
  r.dischtime,
  r.readmitted_30day,
  n.note_id            
FROM readmission r
INNER JOIN notes n    
  ON r.hadm_id = n.hadm_id
ORDER BY r.subject_id
```

Export the results of this query as a CSV file named `cohort_final_v2.csv`.

## 3. Data Merging

Once both the structured cohort (`cohort_final_v2.csv`) and the unstructured BHC text data (`mimic-iv-bhc.csv`) are acquired, you must merge them using our automated script from the root repository directory:

```bash
python src/merge_datasets.py
```

This script performs a 1-to-1 inner join on `note_id` to produce `merged_cohort_text_data.csv`. This merged file then serves as the direct input for our main preprocessing pipeline (`src/preprocess.py`).
