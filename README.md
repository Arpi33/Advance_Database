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

