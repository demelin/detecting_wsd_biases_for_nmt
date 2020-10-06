README

This readme file describes the functionality of the different scripts included in this codebase and their relation to the paper’s contents. For the required and optional arguments of each script, please run python3 script_name.py -h.



REQUIREMENTS

python 3.x

nltk
numpy
spacy
langid
pandas
sklearn
pytorch
pingouin
babelnetpy
transformers
tensorflow_hub
language_tool_python
sentence_transformers



RESOURCE COLLECTION

clean_corpora.py: Used to clean the raw parallel corpora. See Appendix A1 for details.

scrape_babelnet.py: Used to obtain sense clusters for English homographs from BabelNet and refine them by applying filtering heuristics. See Section 2.1, Resource collection, for details.

remove_sense_duplicates.py: Used to remove sense duplicates from BabelNet sense clusters.

extract_attractor_terms.py: Used to collect attractor terms from training corpora, assign them to corresponding homograph senses clusters, and compute their disambiguation bias values. See Section 2.1 for details.

extract_seed_pairs.py: Used to collect seed sentences containing homographs from held-out and test corpora for the benchmarking of WSD error prediction performance and the generation of adversarial samples. See Section 2.2 for details.

extract_homograph_modifiers.py: Used to extract adjectives observed to modify known homograph senses in the English side of parallel corpora for constraining the generation of adversarial samples. See Section 3.1, Attractor selection, for details.

extract_non_homograph_modifiers.py: Used to extract adjectives observed to modify non-homograph nouns in a monolingual corpus for constraining the generation of adversarial samples. See Section 3.1, Attractor selection, for details.

homograph_sense_clusters.json: Contains the manually refined homograph sense clusters used in all experiments. See Section 2.1, Resource collection, for details.



ADVERSARIAL SAMPLE GENERATION

generate_adversarial_samples.py: Used to generate adversarial samples by applying the proposed perturbations to seed sentences and running various filtering heuristics to ensure sample quality. See Section 3.1 for details.

score_seeds_with_bert.py: Used to identify and remove seed sentences containing ambiguous homograph mentions. See Section 3.1, Seed sentence selection, for details.

score_samples_with_lm.py: Used to estimate sentence perplexity increases in adversarial samples relative to their underlying seed sentences. See Section 3.1, Post-generation filtering, for details.



EVALUATION

evaluate_attack_success.py:  Used to check whether unperturbed sentences are translated correctly (see Section 2.2 for details) and whether adversarial attacks are successful (see Section 3.2 for details)

check_challenge_overlap.py: Used to compute the overlap between WSD error prediction challenge sets. See Section 2.2, Challenge set evaluation, for details.

check_sample_transferability.py: Used to compute the Jaccard similarity index between several sets of successful adversarial samples. See Section 4 for details.

create_human_annotation_forms.py: Used to generate forms used in the human evaluation of sample ambiguity and naturalness. See Section 3.3 for details.

evaluate_human_annotation.py: Used to evaluate the judgments collected from human annotators and to compute inter-annotator agreement scores. See Section 3.3 for details.

evaluate_perturbation_efficacy.py: Used to estimate the correlations between successes of adversarial attacks and the perturbation types used to generate them. See Section 3.2 for details.

check_grammaticality_preservation.py: Used to automatically detect grammar errors in seed sentences and adversarial samples for measuring grammaticality degradation after adversarial perturbation. See Section 3.3 for details.

generate_adversarial_challenge_set.py: Used to create the adversarial challenge set. See section 3.2, Challenge set evaluation, for details.

generate_wsd_challenge_set.py: Used to create the WSD error prediction challenge set based on sentence-level disambiguation bias scores. See Section 2.2, Challenge set evaluation, for details.

generate_wsd_challenge_set_from_homographs.py: Used to create the WSD error prediction challenge set based on homograph sense cluster frequency. See Section 2.2, Challenge set evaluation, for details.

test_attractor_correlations.py: Used to calculate correlation scores based on attractor-specific disambiguation bias values. See Section 3.2 for details.

test_homograph_correlations.py: Used to calculate correlation scores based on sentence-level disambiguation bias values. See Section 2.2 for details.

test_homograph_correlations.py: Used to calculate correlation scores based on homograph sense cluster frequency. See Section 2.2 for details.
 
write_adversarial_tables_to_text.p: Used to write adversarial samples to a plain text file to be translated by baseline NMT models.



CHALLENGE SETS

domain/domain_wsd_db_challenge_set.txt: Challenge set for WSD error prediction constructed based on sentence-level disambiguation bias scores. See Section 2.2, Challenge set evaluation, for details.

domain/domain_wsd_hf_challenge_set.txt: Challenge set for WSD error prediction constructed based on homograph sense cluster frequency. See Section 2.2, Challenge set evaluation, for details.

domain/domain_adversarial_challenge_set.txt: Challenge set containing adversarial WSD samples, with perturbations applied to the English side only. See section 3.2, Challenge set evaluation, for details.

