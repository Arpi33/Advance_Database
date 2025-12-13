<h1>LLM-Centric Data Quality Framework</h1>

<p>
This repository contains an end-to-end data quality pipeline for heterogeneous product specifications (camera data),
covering <b>schema matching</b>, <b>entity resolution</b>, and <b>data imputation</b>.
The framework combines prompt-based LLM reasoning (Qwen) with semantic similarity (SBERT + cosine) and strong lexical baselines.
</p>

<h2>Project Modules</h2>
<ul>
  <li><b>Schema Matching</b>: Qwen-2.5-0.5B evaluated in <b>zero-shot vs fine-tuned</b> settings, with analysis by attribute difficulty.</li>
  <li><b>Entity Resolution</b>: SBERT + cosine similarity compared against <b>TF-IDF cosine</b> and <b>Jaccard</b>.</li>
  <li><b>Data Imputation</b>: Entity-aware prompt-based imputation compared against <b>entity-mode</b> and <b>kNN-mode</b> baselines, reported with <b>accuracy + coverage</b>.</li>
</ul>

<h2>Repository Structure (typical)</h2>
<ul>
  <li><code>imputation_lib.py</code> – LLM-based imputation engine</li>
  <li><code>entity_clean.csv</code> – cleaned entity database</li>
  <li><code>target_data_megapixel/</code>, <code>target_data_size/</code> – held-out JSON samples for imputation evaluation</li>
  <li><code>gt_mega.xlsx</code>, <code>gt_size.xlsx</code> – ground-truth values for evaluation</li>
  <li><code>notebooks/</code> – evaluation notebooks (schema / entity / imputation)</li>
</ul>

<h2>Quick Start</h2>
<ol>
  <li>Install dependencies (example):
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>Prepare data:
    <ul>
      <li>Place <code>entity_clean.csv</code> in the project root</li>
      <li>Place evaluation folders (<code>target_data_megapixel</code>, <code>target_data_size</code>) and GT files (<code>gt_*.xlsx</code>)</li>
    </ul>
  </li>
  <li>Run notebooks:
    <pre><code>jupyter notebook</code></pre>
  </li>
</ol>

<h2>Key Results (Summary)</h2>
<ul>
  <li><b>Schema Matching</b>: Fine-tuning strongly improves accuracy compared to zero-shot evaluation.</li>
  <li><b>Entity Resolution</b>: SBERT + cosine substantially outperforms TF-IDF and Jaccard baselines.</li>
  <li><b>Data Imputation</b>: Most challenging stage; performance depends heavily on evidence availability and dataset scale.</li>
</ul>

<h2>Limitations</h2>
<ul>
  <li>Imputation evaluation is constrained by limited ground-truth availability and small held-out samples.</li>
  <li>Results may vary across domains and attribute types; larger datasets are expected to improve robustness.</li>
</ul>

<h2>License</h2>
<p>
Add your license here (e.g., MIT / Apache-2.0) and any model/data usage notes.
</p>
