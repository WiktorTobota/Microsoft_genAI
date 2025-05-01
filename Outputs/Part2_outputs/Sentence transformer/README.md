---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:76608
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: ..
  sentences:
  - Evidence is presented for the utilization of a shortened version of the Arthritis
    Impact Measurement Scales. The results confirmed that the shortened versions retained
    adequate internal consistencies, test-retest reliabilities, and both concurrent
    and predictive validities over a 2 year period which were similar to the original
    longer versions.
  - Hypercholesterolaemia, a risk factor for atherosclerosis (ATH), has been suggested
    to have a role in the development of osteoarthritis (OA). To test this hypothesis,
    the effect of cholesterol and different cholesterol-lowering treatments on OA
    was investigated in a mouse model resembling human lipoprotein metabolism.
  - We investigated the autoantibody (autoAb) profiles in ANA+ individuals lacking
    systemic autoimmune rheumatic disease (SARD) and early SARD patients to determine
    the key differences between these groups and identify factors that are associated
    with an increased risk of symptomatic progression within the next 2 years in ANA+
    individuals.
- source_sentence: ..
  sentences:
  - Studies comparing 500 mg rituximab and 1,000 mg rituximab doses in rheumatoid
    arthritis have yielded conflicting data on clinical outcomes, but in all of these
    studies a subgroup of patients has had excellent responses at the lower dose.
    Historically, it was considered that rituximab uniformly depleted B cells at both
    doses. Using highly sensitive assays, we have shown that B cell depletion is variable
    and predictive of clinical response. Using the same techniques, we undertook the
    present study to test the hypothesis that the level of B cell depletion, rather
    than the rituximab dose, determines clinical response.
  - To investigate whether in vivo capillary microscopy of the lower lip mucosa can
    be used to assess microvascular disease in systemic sclerosis.
  - The aim was to investigate the reliability and validity of radiographic sacroiliitis
    assessment in anteroposterior (AP) lumbar radiographs compared with conventional
    pelvic radiographs in patients with axial spondyloarthritis (axSpA).
- source_sentence: ..
  sentences:
  - As reported by us, a new myeloid cell population with an oncofetal membrane marker,
    dimeric Lex (di-Lex; III3FucV3 FucnLc6), was found in the epiphyseal bone marrow
    adjacent to the involved joints of patients with severe rheumatoid arthritis (RA).
    Patients with RA received intradermal (id) injections of di-Lex incorporated in
    liposome or of high molecular weight glycoprotein, or tumor associated carbohydrate
    antigen (TCA), containing the same carbohydrate epitope as di-Lex. The epiphyseal
    myeloid cells were reduced or sometimes eliminated during id injection. In random
    trials of id injection, observation under clinical and laboratory conditions showed
    improvement in 63% (17/27) of the patients treated for 6 months with appropriate
    doses of di-Lex (III3FucnLc4), and in 72% (31/43) of those treated with an identical
    protocol for TCA. However, id injection with monomeric Lex had no effect.
  - To investigate the feasibility of collecting rheumatoid arthritis (RA) patient
    self-administered outcome data using touch-screen computers in a routine out-patient
    clinic.
  - To develop an objective method of nailfold capillaroscopy (NFC), applicable to
    a wide age range of paediatric patients. To compare the morphological characteristics
    of the nailfold capillaries in different rheumatology patient groups and controls.
- source_sentence: ..
  sentences:
  - To prepare a website for families and health professionals containing up to date
    information about paediatric rheumatic diseases (PRD).
  - Are walk time and grip strength measures of disease activity or functional ability?
    Ninety-two patients with rheumatoid arthritis were examined initially and 12 months
    later for clinical measures including joint deformity, and answered a functional
    status questionnaire. Walk time and grip strength were strongly related to joint
    deformity and functional questionnaire measures, and appeared insensitive in showing
    changes in disease activity over time. They could serve as objective functional
    measures in studies primarily directed towards changing functional ability, but
    appeared to be poor major outcome measures for trials aimed at altering disease
    activity.
  - To evaluate the effect of orally administered methotrexate (MTX) on the density
    of CC chemokine receptor 2 (CCR2) and CXC chemokine receptor 3 (CXCR3) on circulating
    monocytes, and the coexpression of CXCR3 and CCR2 on CD4 T lymphocytes in patients
    with active chronic rheumatoid arthritis.
- source_sentence: ..
  sentences:
  - To establish whether there is a relationship between serum magnesium (Mg) concentration
    and radiographic knee osteoarthritis (OA).
  - To investigate sicca symptoms in patients with rheumatoid arthritis (RA) with
    respect to constancy, temporal changes of prevalence, and possible risk factors.
  - Clinical trials in early diffuse SSc have consistently shown a placebo group response
    with a declining modified Rodnan skin score (mRSS), with negative outcomes. Our
    objective was to identify strategies using clinical characteristics or laboratory
    values to improve trial design.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '..',
    'To investigate sicca symptoms in patients with rheumatoid arthritis (RA) with respect to constancy, temporal changes of prevalence, and possible risk factors.',
    'Clinical trials in early diffuse SSc have consistently shown a placebo group response with a declining modified Rodnan skin score (mRSS), with negative outcomes. Our objective was to identify strategies using clinical characteristics or laboratory values to improve trial design.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 76,608 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                      | sentence_1                                                                          | label                                                         |
  |:--------|:--------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                          | string                                                                              | float                                                         |
  | details | <ul><li>min: 2 tokens</li><li>mean: 3.49 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 92.53 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.5</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | label            |
  |:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>..</code> | <code>This is a pilot study of zileuton, a selective 5-lipoxygenase inhibitor in systemic lupus erythematosus (SLE).</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>0.0</code> |
  | <code>..</code> | <code>Improvement in analysis and reporting results of osteoarthritis (OA) clinical trials has been recently obtained because of harmonization and standardization of the selection of outcome variables (OMERACT 3 and OARSI). Moreover, OARSI has recently proposed the OARSI responder criteria. This composite index permits presentation of results of symptom modifying clinical trials in OA based on individual patient responses (responder yes/no). The 2 organizations (OMERACT and OARSI) established a task force aimed at evaluating: (1) the variability of observed placebo and active treatment effects using the OARSI responder criteria; and (2) the possibility of proposing a simplified set of criteria. The conclusions of the task force were presented and discussed during the OMERACT 6 conference, where a simplified set of responder criteria (OMERACT-OARSI set of criteria) was proposed.</code> | <code>0.0</code> |
  | <code>.</code>  | <code>The WOMAC (Western Ontario and McMaster Universities) function subscale is widely used in clinical trials of hip and knee osteoarthritis. Reducing the number of items of the subscale would enhance efficiency and compliance, particularly for use in clinical practice applications.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 256
- `per_device_eval_batch_size`: 256
- `disable_tqdm`: False
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 256
- `per_device_eval_batch_size`: 256
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.6667 | 500  | 0.3572        |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.4.1
- Transformers: 4.51.1
- PyTorch: 2.5.1+cu124
- Accelerate: 1.3.0
- Datasets: 3.5.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->