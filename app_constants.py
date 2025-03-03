import os
import pandas, json

import nlp_pipeline.nlp_service_inference.config as CONFIG


SLASH = os.sep

curr_file_path =  os.path.abspath(__file__)
main_dir_path =  os.path.dirname(os.path.dirname(curr_file_path))

if 'NLP_SERVICE_FILES_PATH' in os.environ:
    nlp_files_dir = os.path.abspath(os.environ['NLP_SERVICE_FILES_PATH'])
else:
    nlp_files_dir = os.path.dirname(os.path.dirname(main_dir_path)) + SLASH + 'nlp_service_files'
    
if 'SPARK_NLP_SERVICE_FILES_PATH' in os.environ:
    spark_nlp_files_dir = os.environ['SPARK_NLP_SERVICE_FILES_PATH']
else:
    spark_nlp_files_dir = os.path.dirname(os.path.dirname(main_dir_path)) + SLASH + 'nlp_service_files'

chai_inference_input_columns = ['tumor_type','profile_key','pif_key','text','docDate','general','molecular_pathology_report','ngs','ngs_lab']
chai_inference_relevant_columns = ['sentences','ner_chai', 'assertion_chai__ner_chai', 'classification_chai__ner_chai','re_chai__ner_chai', 'chai_logs']
# chai_inference_on_jsl_relevant_columns = ['classification_chai__ner_oncology', 'postprocess_classification_chai__ner_oncology', 'irrelevance_chai__ner_oncology'] if CONFIG.USE_JSL_NER else []
# chai_inference_relevant_columns += chai_inference_on_jsl_relevant_columns


dependencies_dir_path = nlp_files_dir + SLASH + 'input_artifacts' + SLASH + 'chai_inference'
spark_dependencies_dir_path = spark_nlp_files_dir + SLASH + 'input_artifacts' + SLASH + 'chai_inference'

model_dir_path = dependencies_dir_path + SLASH + 'models'

config_dir_path = main_dir_path + SLASH + 'Config'
config_regex_dir_path = config_dir_path + SLASH + 'regex'

config_pycontext_dir_path = config_dir_path + SLASH + 'pycontext'
config_ontology_dir_path = config_dir_path + SLASH + 'ontology'
config_log_dir_path = config_dir_path + SLASH + 'logs'

iom_file_path = config_dir_path + SLASH + 'iom.json'
log_conf_path = config_log_dir_path + SLASH + 'log.conf'
log_outp_path = main_dir_path + SLASH + 'chai_logs/logs.log'
os.environ['CHAI_LOG_OUTP_PATH'] = log_outp_path

cancer_to_topography_mapping_path = config_dir_path + SLASH +  'cancer_to_topography_mapping.json'
cancer_mapping_path = config_dir_path + SLASH + 'cancer_mapping.json'
tumor_output_mapping_path = config_dir_path + SLASH + 'tumor_output_mapping.json'



ontology_file_path_dict = {}
ontology_file_path_dict['default'] = config_ontology_dir_path + SLASH + 'Default'
ontology_file_path_dict['breast cancer'] = config_ontology_dir_path + SLASH + 'breast'
ontology_file_path_dict['lung cancer'] = config_ontology_dir_path + SLASH + 'lung'
ontology_file_path_dict['prostate cancer'] = config_ontology_dir_path + SLASH + 'prostate'
ontology_file_path_dict['biomarkers'] = config_ontology_dir_path + SLASH + 'biomarkers'



## doc selection dir path
doc_selection_dir_path = config_dir_path + SLASH + 'doc_selection'
image_type2_path = doc_selection_dir_path + SLASH + 'image_type2_qcca_BCLC_count_EM_v3.1_final_RG.csv'
image_type1_path = doc_selection_dir_path + SLASH + 'image_type1_qcca_BCLC_count_EM_v1_final.csv'
ngs_molpath_path = doc_selection_dir_path + SLASH + 'combined_overlap_ngs_molpath_v3.csv'
doc_selection_file_names_dict = {}
doc_selection_file_names_dict['image_type2'] = image_type2_path 
doc_selection_file_names_dict['image_type1'] = image_type1_path 
doc_selection_file_names_dict['ngs_molpath'] = ngs_molpath_path 



# ontology files
ontology_file_names_dict = {}
ontology_file_names_dict['tumor'] = 'alias__cancer.json'
ontology_file_names_dict['histology'] = 'Histology_Informatics_KB_V3.6.csv'
ontology_file_names_dict['biomarkers']  = 'biomarkers_p360_dts_ngene_v3_20250206.csv'
ontology_file_names_dict['biomarkers_regex']  = 'biomarkers_name_regex.json'
ontology_file_names_dict['biomarkers_blacklist']  = 'biomarker_regex_blacklist.json'
ontology_file_names_dict['labs'] = 'LabTest_Config.csv'
ontology_file_names_dict['vitals'] = 'VitalsTest_Config.csv'
ontology_file_names_dict['meds'] = 'Chemotherapy_Drug_Names_Master.csv'
ontology_file_names_dict['labs'] = 'LabTest_Config.csv'
ontology_file_names_dict['AJCC7'] = 'AJCC7.csv'
ontology_file_names_dict['biomarkers_mapping_file'] = 'biomarkers_status_normalization.csv'
ontology_file_names_dict['stage_normalization_ordering'] = 'stage_normalization_ordering.csv'
ontology_file_names_dict['ecog_ordering'] = 'ecog_ordering.csv'
ontology_file_names_dict['karnofsky_ordering'] = 'karnofsky_ordering.csv'
ontology_file_names_dict['ambiguous_variants'] = 'ambiguous_mv_ontology.csv'
ontology_file_names_dict['unambiguous_variants'] = 'unambiguous_mv_ontology.csv'
ontology_file_names_dict['surgery'] = 'Surgery_Informatics_KB_V3.6.csv'
# ontology_file_names_dict['medication'] = 'cancer_drugs_public_V9.0.csv' 
ontology_file_names_dict['bio_attribute_normalizer'] = 'bio_attr_norm.csv' 
ontology_file_names_dict['radiation'] = 'Radiation_synonyms_normalization.csv'
ontology_file_names_dict['solid_cancer'] = 'cancer_dx_list_solidcancers_RWD_DTS_withsynonymsforNLP_06032022_V1.csv'
ontology_file_names_dict['alcohol'] = 'alcohol_v1.0.csv'
ontology_file_names_dict['menopause'] = 'Menopause_v1.1.csv'


# regex files
regex_file_path_dict = {}
regex_file_path_dict['biomarker_regex'] = config_regex_dir_path + SLASH + 'regex_biomarkers.json'
regex_file_path_dict['tnm_regex'] = config_regex_dir_path + SLASH + 'regex_tnm.json'
regex_file_path_dict["treatmentresponse"] = config_dir_path + SLASH + "ontology" +SLASH +"treatment_response" +SLASH +"treatment_response_kb.yaml"
regex_file_path_dict["comorbidities"] = config_dir_path + SLASH + "ontology" +SLASH +"Default" +SLASH +"comorbidities.yaml"



# pycontext
pycontext_path_dict = {}
pycontext_path_dict['pycontext_modifiers_path'] = config_pycontext_dir_path + SLASH + 'pycontext_modifiers.json'
pycontext_path_dict['pycontext_targets_path'] = config_pycontext_dir_path + SLASH + 'pycontext_targets.json'



# ulmfit model files
ulmfit_model_file_path_dict = {}



# bert model files
bert_model_file_path_dict = {}

# bert_model_file_path_dict['biomarkers__biomarkers'] = {'dir' : 'biomarkers_v2','task_name' : 'biomarkers', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['metstatus__mets1'] = {'dir' : 'mets_finetuned','task_name' : 'mets1', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['metstatus__mets2'] = {'dir' : 'mets2_vanilla_classification','task_name' : 'mets2', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['metstatus__siteofmets'] = {'dir' : 'siteofmets','task_name' : 'siteofmets', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['stage__stage_irr'] = {'dir' : 'stage/stage_irr','task_name' : 'stage_irr', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['surgery__surgery'] = {'dir' : 'surgery','task_name' : 'surgery', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['biomarkers__bio_name_ner'] = {'dir' : 'bio_name/ner_v5.6', 'task_name': 'bio_name_ner', 'max_seq_length': 256, 'do_lower_case': True}

bert_model_file_path_dict['biomarkers__bio_name_irr'] = {'dir' : 'bio_name/irr_v5.7', 'task_name': 'bio_name_irr', 'max_seq_length': 256, 'do_lower_case': True}

# bert_model_file_path_dict['pcsubtype__pcsubtype'] = {'dir' : 'pcsubtype', 'task_name' : 'pcsubtype', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['tumor__tumor_irr'] = {'dir' : 'tumor/bert_irrelevance', 'task_name' : 'tumor_irr', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['alcohol__alcohol'] = {'dir' : 'alcohol', 'task_name' : 'alcohol', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['smoking__smoking'] = {'dir' : 'smoking', 'task_name' : 'smoking', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['menopause__menopause_cls'] = {'dir' : 'menopause/bert_classification', 'task_name' : 'menopause_cls', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['treatmentresponse__treatmentresponse_ner'] = {'dir' : 'treatmentresponse/ner', 'task_name' : 'treatmentresponse_ner', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['tumor__tumor_ner'] = {'dir' : 'tumor/bert_ner_v4.6', 'task_name' : 'tumor_ner', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['tumor__tumor_nerc'] = {'dir' : 'tumor/bert_nerc_v4.6', 'task_name' : 'tumor_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['jsl_grade__grade_nerc'] = {'dir' : 'jsl_grade/grade_nerc','task_name' : 'grade_nerc', 'max_seq_length' : 256, 'do_lower_case' : True}

# bert_model_file_path_dict['jsl_stage__stage_nerc'] = {'dir' : 'jsl_stage/stage_nerc','task_name' : 'stage_nerc', 'max_seq_length' : 256, 'do_lower_case' : True}

# bert_model_file_path_dict['jsl_performance_status__performance_status_nerc'] = {'dir' : 'jsl_performance_status/performance_status_nerc','task_name' : 'performance_status_nerc', 'max_seq_length' : 256, 'do_lower_case' : True}

bert_model_file_path_dict['oncologyv2__oncologyv2'] = {'dir': 'oncologyv2/ner_v5.2', 'task_name': 'oncologyv2', 'max_seq_length': 128, 'do_lower_case': True}

bert_model_file_path_dict['medication__medication_ner'] = {'dir': 'medication/ner_v4.6', 'task_name': 'medication_ner', 'max_seq_length': 128, 'do_lower_case': True}

bert_model_file_path_dict['radiation__radiation_ner'] = {'dir': 'radiation/radiation_ner_v5.1', 'task_name': 'radiation_ner', 'max_seq_length': 128, 'do_lower_case': True}

bert_model_file_path_dict['surgery__surgery_irr'] = {'dir' : 'surgery/surgery_irr_v4.6','task_name' : 'surgery_irr', 'max_seq_length' : 192, 'do_lower_case' : True}

bert_model_file_path_dict['hist__hist_irr'] = {'dir' : 'hist/hist_irr_v5.2','task_name' : 'hist_irr', 'max_seq_length' : 147, 'do_lower_case' : True}

bert_model_file_path_dict['grade__grade_nerc'] = {'dir' : 'grade/grade_nerc_v4.6','task_name' : 'grade_nerc', 'max_seq_length' : 144, 'do_lower_case' : True}

bert_model_file_path_dict['stage__stage_nerc'] = {'dir' : 'stage/stage_nerc_v5.2','task_name' : 'stage_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['medication__route_nerc'] = {'dir' : 'medication/route_nerc','task_name' : 'route_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['medication__therapy_ongoing_nerc'] = {'dir' : 'medication/therapy_ongoing_nerc','task_name' : 'therapy_ongoing_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['performance_status__performance_status_nerc'] = {'dir' : 'performance_status/performance_status_nerc_v4.6','task_name' : 'performance_status_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['comorbidities__comorbidities_ner'] = {'dir' : 'comorbidities/comorbidities_ner_v5.3','task_name' : 'comorbidities_ner', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['comorbidities__comorbidities_irr'] = {'dir' : 'comorbidities/comorbidities_irr_v5.3','task_name' : 'comorbidities_irr', 'max_seq_length' : 128, 'do_lower_case' : True}

# bert_model_file_path_dict['comorbidities__comorbidities_nerc'] = {'dir' : 'comorbidities/comorbidities_nerc_v4.1','task_name' : 'comorbidities_nerc', 'max_seq_length' : 256, 'do_lower_case' : True}

bert_model_file_path_dict['site__site_ner'] = {'dir' : 'site/site_ner_v4.1', 'task_name' : 'site_ner', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['site__site_nerc'] = {'dir' : 'site/site_nerc_v4.2','task_name' : 'site_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['metastasis__metastasis_ner'] = {'dir' : 'metastasis/metastasis_ner', 'task_name' : 'metastasis_ner', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['metastasis__metastasis_nerc'] = {'dir' : 'metastasis/metastasis_nerc_v4.1','task_name' : 'metastasis_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['race_ethnicity__race_ethnicity_ner'] = {'dir' : 'race_ethnicity/race_ethnicity_ner_v4.1','task_name' : 'race_ethnicity_ner', 'max_seq_length' : 512, 'do_lower_case' : True}

bert_model_file_path_dict['race_ethnicity__race_nerc'] = {'dir' : 'race_ethnicity/race_nerc_v4.1','task_name' : 'race_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['race_ethnicity__ethnicity_nerc'] = {'dir' : 'race_ethnicity/ethnicity_nerc_v4.1','task_name' : 'ethnicity_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['medication__drug_nerc'] = {'dir' : 'medication/drug_nerc_v5.1','task_name' : 'drug_nerc', 'max_seq_length' : 128, 'do_lower_case' : True}

bert_model_file_path_dict['radiation__radiation_modality_irr'] = {'dir' : 'radiation/modality_irr_v5.3','task_name' : 'radiation_modality_irr', 'max_seq_length' : 128, 'do_lower_case' : True}

# crf status2 model files
crf_model_file_path_dict = {}

# crf_model_file_path_dict['biomarkers__biomarkers'] = {'dir' : 'status2_biomarkers_allFeatures_v1','task_name' : 'biomarkers', 'filename':'pd1_pdl1_tmb_status2_model_clq_data_allFeatures_0806.sav'}

# crf_model_file_path_dict['biomarkers__biomarkers_bc'] = {'dir' : 'status2_biomarkers_allFeatures_v1','task_name' : 'biomarkers_bc', 'filename':'er_pr_her2_status2_model_emol_data_allFeatures_0212.sav'}



# xgboost model files
xgboost_model_file_path_dict = {}

xgboost_model_file_path_dict['secondary_tumor__secondary_tumor'] = {'dir':'secondary_tumor','task_name':'secondary_tumor','filename':'xgboost_NSCLC_GT_run_v3_doc_level.pkl',
'word_tokenizer_filename':'word_vectorizer_nsclc_GT_run_v3_doc_level.pkl'}



# dl model files
dl_model_file_path_dict = {}

# dl_model_file_path_dict['biomarkers__biomarkers'] = {
#     'dir':'bio-re',
#     'task_name':'bio-re',
#     'filename':'model.bin',
#     'parameters':'parameters.json',
#     'rel_vocab':'rel_vocab.json',
#     'word_vocab': 'word_vocab.json',
#     'abs_vocab': 'abs_vocab.json'
# }



# spacy model files
spacy_model_file_path_dict = {}

# spacy_model_file_path_dict['surgery__surgery_ner'] = {
#     'dir': 'surgery_ner_type',
#     'task_name':'surgery_ner',
# }

# spacy_model_file_path_dict['metstatus__siteofmets_ner'] = {
#     'dir': 'siteofmets_ner',
#     'task_name':'siteofmets_ner',
# }



# decision tree model files
decision_tree_model_file_path_dict = {}
decision_tree_model_file_path_dict['medication__medication_ner'] = {
    'dir': 'medication/drug_name_decision_tree_v5.1', 'task_name': 'drug_name_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['drug name']
}

decision_tree_model_file_path_dict['oncologyv2__oncologyv2_surgery_hist'] = {
    'dir': 'oncologyv2/surgery_hist_decision_tree_v4.6', 'task_name': 'surgery_hist_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['cancer_surgery', 'histological_type']
}

decision_tree_model_file_path_dict['oncologyv2__oncologyv2_grade'] = {
    'dir': 'oncologyv2/grade_decision_tree_v4.6', 'task_name': 'grade_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['grade']
}

decision_tree_model_file_path_dict['tumor__tumor_ner'] = {
    'dir': 'tumor/tumor_decision_tree_v5.4', 'task_name': 'tumor_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['tumor']
}

decision_tree_model_file_path_dict['oncologyv2__oncologyv2_staging'] = {
    'dir': 'oncologyv2/staging_decision_tree_v5.4', 'task_name': 'staging_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['staging']
}

decision_tree_model_file_path_dict['biomarkers__bio_name_ner'] = {
    'dir': 'bio_name/bio_name_decision_tree_v5.4', 'task_name': 'biomarkers_dt_classifier', 'filename': 'decision_tree_model.bin', 'labels_of_interest':['gene']
}