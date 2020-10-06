# Readme 

This repository contains the experimental code for the publication **Detecting Word Sense Disambiguation Biases in Machine Translation for Model-Agnostic Adversarial Attacks** (Emelin, Denis, Ivan Titov, and Rico Sennrich, EMNLP 2020).

The readme is meant to provide an overview of the functionality of the different scripts included in this codebase and their relation to the paper’s contents. For the required and optional arguments of each script, please run python3 `script_name.py` -h. References to the paper are given in **bold**.
  
  
## Requirements

* python 3.x

* nltk
* numpy
* spacy
* langid
* pandas
* sklearn
* pytorch
* pingouin
* babelnetpy
* transformers
* tensorflow_hub
* language_tool_python
* sentence_transformers
  
  
## Resource collection

`resource_collection/clean_corpora.py`  
:green_book: Cleans the raw parallel corpora. See **Appendix A1** for details.

`resource_collection/scrape_babelnet.py`  
:green_book: Collects sense clusters for English homographs from BabelNet and refines them by applying filtering heuristics. See **Section 2.1, Resource collection** for details.

`resource_collection/remove_sense_duplicates.py`  
:green_book: Removes sense duplicates from collected BabelNet sense clusters.

`resource_collection/extract_attractor_terms.py`  
:green_book: Extracts attractor terms from specified training corpora, assigning them to corresponding homograph senses clusters, and computes their disambiguation bias values. See **Section 2.1** for details.

`resource_collection/extract_seed_pairs.py`  
:green_book: Extracts seed sentences containing homographs from held-out and test corpora for the benchmarking of WSD error prediction performance and the generation of adversarial samples. See **Section 2.2** for details.

`resource_collection/extract_homograph_modifiers.py`  
:green_book: Extracts adjectives observed to modify known homograph senses in the English portion of the training corpora, used to constrain the generation of adversarial samples. See **Section 3.1, Attractor selection** for details.

`resource_collection/extract_non_homograph_modifiers.py`  
:green_book: Extracts adjectives observed to modify non-homograph nouns in a specified monolingual English corpus, used to constrain the generation of adversarial samples. See **Section 3.1, Attractor selection** for details.
  
  
## Adversarial sample generation

`adversarial_sample_generation/generate_adversarial_samples.py`:
Used to generate adversarial samples by applying the proposed perturbations to seed sentences and running various filtering heuristics to ensure sample quality. See Section 3.1 for details.

`adversarial_sample_generation/score_seeds_with_bert.py`:
Used to identify and remove seed sentences containing ambiguous homograph mentions. See Section 3.1, Seed sentence selection, for details.

`adversarial_sample_generation/score_samples_with_lm.py`:
Used to estimate sentence perplexity increases in adversarial samples relative to their underlying seed sentences. See Section 3.1, Post-generation filtering, for details.



## Evaluation

`evaluation/evaluate_attack_success.py`:
Used to check whether unperturbed sentences are translated correctly (see Section 2.2 for details) and whether adversarial attacks are successful (see Section 3.2 for details)

`evaluation/check_challenge_overlap.py`:
Used to compute the overlap between WSD error prediction challenge sets. See Section 2.2, Challenge set evaluation, for details.

`evaluation/check_sample_transferability.py`:
Used to compute the Jaccard similarity index between several sets of successful adversarial samples. See Section 4 for details.

`evaluation/create_human_annotation_forms.py`:
Used to generate forms used in the human evaluation of sample ambiguity and naturalness. See Section 3.3 for details.

`evaluation/evaluate_human_annotation.py`:
Used to evaluate the judgments collected from human annotators and to compute inter-annotator agreement scores. See Section 3.3 for details.

`evaluation/evaluate_perturbation_efficacy.py`:
Used to estimate the correlations between successes of adversarial attacks and the perturbation types used to generate them. See Section 3.2 for details.

`evaluation/check_grammaticality_preservation.py`:
Used to automatically detect grammar errors in seed sentences and adversarial samples for measuring grammaticality degradation after adversarial perturbation. See Section 3.3 for details.

`evaluation/generate_adversarial_challenge_set.py`:
Used to create the adversarial challenge set. See section 3.2, Challenge set evaluation, for details.

`evaluation/generate_wsd_challenge_set.py`:
Used to create the WSD error prediction challenge set based on sentence-level disambiguation bias scores. See Section 2.2, Challenge set evaluation, for details.

`evaluation/generate_wsd_challenge_set_from_homographs.py`:
Used to create the WSD error prediction challenge set based on homograph sense cluster frequency. See Section 2.2, Challenge set evaluation, for details.

`evaluation/test_attractor_correlations.py`:
Used to calculate correlation scores based on attractor-specific disambiguation bias values. See Section 3.2 for details.

`evaluation/test_homograph_correlations.py`:
Used to calculate correlation scores based on sentence-level disambiguation bias values. See Section 2.2 for details.

`evaluation/test_homograph_correlations.py`:
Used to calculate correlation scores based on homograph sense cluster frequency. See Section 2.2 for details.
 
`evaluation/write_adversarial_tables_to_text.py`:
Used to write adversarial samples to a plain text file to be translated by baseline NMT models.



## Resources
`./ende_homograph_sense_clusters.json`:
Contains the manually refined homograph sense clusters used in all experiments. See Section 2.1, Resource collection, for details.


## Citation

```
@inproceedings{emelin2020detecting,
  title={Detecting Word Sense Disambiguation Biases in Machine Translation for Model-Agnostic Adversarial Attacks},
  author={Emelin, Denis and Titov, Ivan and Sennrich, Rico},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  pages={xxx--xxxx},
  year={2020}
}
```
