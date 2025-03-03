import sys
import time
import re
import copy
import json
import bisect
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple
from nlp_pipeline.nlp_service_inference.chai_inference.utils import get_mapping_dict, return_site_of_metastasis, chain_iter, get_stage_from_tnm, normalize_stage
from nlp_pipeline.nlp_service_inference.chai_inference.utils import is_suspicious_metsite

from nlp_pipeline.nlp_service_inference.chai_inference.app.app_constants import ontology_file_path_dict, ontology_file_names_dict, bert_model_file_path_dict, model_dir_path, log_conf_path
from nlp_pipeline.nlp_service_inference.chai_inference.models import get_negation_status
from nlp_pipeline.nlp_service_inference.chai_inference.inference.bert_inference import bert_inference
import pandas
import datetime
import numpy as np
import os
# from nltk import sent_tokenize, word_tokenize
# import logging, logging.config
from nlp_pipeline.nlp_service_inference.chai_inference.inference.crf_status2_inference import nlp_pipeline_format
# from nlp_pipeline.nlp_service_inference.chai_inference.inference.smoking_status_inference import nlp_predict_smoking_status
from nlp_pipeline.nlp_service_inference.chai_inference.inference.secondary_tumor_inference import nlp_predict_secondary_tumor_status

from nlp_pipeline.nlp_service_inference.chai_inference.inference.histology_inference import nlp_predict_histology

from nlp_pipeline.nlp_service_inference.chai_inference.inference.menopause_inference import update_menopause_rule_based_prediction, update_menopause_rule_based_irrelevance
# from nlp_pipeline.nlp_service_inference.chai_inference.inference.surgery_inference import nlp_predict_surgery, normalize_surgery_prediction
from nlp_pipeline.nlp_service_inference.chai_inference.inference.biomarkers_inference import predict
from nlp_pipeline.nlp_service_inference.chai_inference.inference.biomarkers_inference import ner_inference
# from nlp_pipeline.nlp_service_inference.chai_inference.inference.surgery_inference import get_surgery_ner_pred
from nlp_pipeline.nlp_service_inference.chai_inference.preprocessing.helper import cancer_pred_fields
from nlp_pipeline.nlp_service_inference.chai_inference.inference.bert_ner_inference import bert_ner_inference
from nlp_pipeline.nlp_service_inference.chai_inference.inference.stage_inference import get_staging_entities, postprocess_multilabelled_staging

# from gensim import utils as gsu
# import gensim.parsing.preprocessing as gsp

import nlp_pipeline.nlp_service_inference.config as CONFIG

# logging.config.fileConfig(log_conf_path)
# logger = logging.getLogger('Inference')
# logger = logging.getLogger(__name__)

default_ontology_path = ontology_file_path_dict['default'] + '/'

kb_path = default_ontology_path +  ontology_file_names_dict['biomarkers']
norm_path = default_ontology_path + ontology_file_names_dict['bio_attribute_normalizer']

biomarkers_ontology_path = ontology_file_path_dict['biomarkers'] + '/'
kb_regex_path = biomarkers_ontology_path +  ontology_file_names_dict['biomarkers_regex']

normalizer_df = pandas.read_csv(norm_path)
normalizer_df = normalizer_df[['Raw_Value ', 'Attribute', 'Normalized_Value']]
normalizer_df = normalizer_df.apply(lambda x: x.astype(str).str.lower())
norm_tr = normalizer_df[normalizer_df['Attribute']=='test_result']
norm_tr.set_index('Raw_Value ', inplace=True)
test_result_norm_dict = norm_tr.to_dict()['Normalized_Value']
test_result_norm_dict[''] = ''
test_result_norm_dict['irrelevant'] = 'irrelevant'

def new_normalizer_func(text):
    
    text = str(text).strip()
    text = text.lower()
    
    if text == '':
        return ''
    
    norm_dict  = test_result_norm_dict
    
    text = text.lower()
    config_keys = list(norm_dict.keys())
    
    for key in config_keys:
        
        if key == text:
            return norm_dict[key]
    
        
    for key in config_keys:
        if key in text:
            return norm_dict[key]
    
    return None


bm_kb = pandas.read_csv(kb_path)
bm_kb = bm_kb[['Biomarker Name', 'Alias']]
bm_kb = bm_kb.loc[~bm_kb['Biomarker Name'].isnull()]
bm_kb = bm_kb.loc[bm_kb['Biomarker Name'].astype(str).str.strip().str.len()>0]
bm_kb.loc[bm_kb['Alias'].isnull(), 'Alias'] = bm_kb.loc[bm_kb['Alias'].isnull(), 'Biomarker Name']
bm_kb.loc[bm_kb['Alias'].astype(str).str.strip().str.len()==0, 'Alias'] = \
    bm_kb.loc[bm_kb['Alias'].astype(str).str.strip().str.len()==0, 'Biomarker Name']
bm_kb.set_index('Biomarker Name', inplace=True)
bm_kb_dict = bm_kb.to_dict()['Alias']

biomarker_ocr_fix_aliases = json.load(open(kb_regex_path))["biomarker_ocr_fix_aliases"]

def extend_norm_list(bm_kb_dict):
    bm_kb_new = {}
    for key in bm_kb_dict.keys():
        bl_old = bm_kb_dict[key].split(', ')
        bl_old.append(key)
        bl_new = []
        for i in bl_old:
            bl_new.append(i)
            bl_new.append(''.join(i.split('-')))
            bl_new.append(''.join(i.split()))
        for i in biomarker_ocr_fix_aliases.get(key.lower(), []):
            bl_new.append(i)
            bl_new.append(''.join(i.split('-')))
            bl_new.append(''.join(i.split()))
        bl_new = [i.lower() for i in bl_new]
        bl_new = list(set(bl_new))
        bm_kb_new[key.lower()] = bl_new
    return bm_kb_new
        
bm_kb_dict_new = extend_norm_list(bm_kb_dict)

def normalize_bm(name):
    name = name.lower()
    if name in bm_kb_dict_new:
        return name
    else:
        for bm, bm_alias in bm_kb_dict_new.items():
            if name in bm_alias:
                return bm.lower()
        return None


def return_bio_check(name: str):
    
    try:
        bio_list = bm_kb_dict[name.upper()]
        bio_list = bio_list.split(', ')
    except Exception as e:
        bio_list = []
    
    bio_list = [i.lower() for i in bio_list]
    bio_list.append(name.lower())
    
    return bio_list



class Inference(object):
    
    def __init__(self, fields = None, dict_of_sentences:Dict[str,List[Tuple]] = None, biomarker_list = None, regexObj = None, 
                 regexBiomarkersObj = None, ulmfit_models = None, BERT_models = None, tumor_type = None, primary_tumor_mapping_dict = None, 
                 histology_mapping_dict = None, surgery_mapping_dict = None, tnm_stage_mapping_df = None, stage_normalization_df = None, crf_models = None, xgboost_models = None,
                 docObj=None, meta_dos=None, spacy_models=None, dl_models = None, use_onnx = [], decision_tree_models=None):
        self.dict_of_sentences = dict_of_sentences
        self.biomarker_list = biomarker_list
        self.regexObj = regexObj
        self.regexBiomarkersObj = regexBiomarkersObj
        self.fields_with_negation = ['hist','tumor','medication','comorbidity']
        self.crf_status2_biomarker_class = ['pd1','pdl1','tmb','er','pr','her2']
        self.ulmfit_models = ulmfit_models
        self.BERT_models = BERT_models
        self.crf_models = crf_models
        self.xgboost_models = xgboost_models
        self.dl_models = dl_models
        self.decision_tree_models = decision_tree_models
        self.primary_tumor_type_mapping_dict = primary_tumor_mapping_dict
        self.histology_mapping_dict = histology_mapping_dict
        self.surgery_mapping_dict = surgery_mapping_dict
        self.menopause_surgery_list = [k for k,v in regexObj.menopause_map.items() if v=='surgery']
        
        self.spacy_models = spacy_models
        self.use_onnx=use_onnx
        self.tnm_stage_mapping_df = tnm_stage_mapping_df
        self.stage_normalization_df = stage_normalization_df
        self.text = docObj.text
        self.docObj = docObj
        self.meta_dos = meta_dos

        self.biomarker_status1_regex_dict = self.regexBiomarkersObj.biomarker_status1_regex_dict
        self.biomarker_status2_regex_dict = self.regexBiomarkersObj.biomarker_status2_regex_dict
        self.biomarker_status_posneg_regex = self.regexBiomarkersObj.biomarker_status_posneg_regex
        self.amb_mv_regex_dict, self.unamb_mv_regex_dict = self.regexBiomarkersObj.amb_mv_regex_dict, self.regexBiomarkersObj.unamb_mv_regex_dict
        self.tumor_type = tumor_type
        self.irrelevant_list_for_stage = ['kidney','ckd','lymphedema','renal','decubitus', 'cirrhosis', 'copd', 'chronic obstructive pulmonary disease', 'diabetes', 'esrd', 'ulcer', 'fibrosis', 'hemorrhoid', 'cystocele', 'gfr', 'hypeprkalemia',
                                         'uterine prolapse', 'nhl', 'glaucoma', 'hyperlipidemia', 'creatinine', 'cad']
        self.suspicious_list_for_mets = ['/o','r/o', 'mobility deficit/weakness/peripheral', 'rule out', 'concern', 'possible', 'likely', 'discuss', 'negative for', 'not excluded', 'suspicious', 'differential', 'may be', 'maybe', 'exclude', 'risk of','ddx', 'consider', 'suspicious', 'diff dx', 'possibil', 'discuss']
        self.outDict= {}
        self.time_dict= {}
        self.dos_type = "" ## flag to indicate what kind of dos we are returning, exact dos, or approximate using some logic
        self._get_dates()
        self._get_biomarker_key_in_dict_of_sentences()
        self.get_natural_chunk_dict()
        self.get_pred_dict()
        
        # iterating pred_dict keys to account for order within fields

        for field in self.pred_dict.keys():
            if field in fields:
                if field not in ['dates','dateofservice','menopause'] and field not in self.biomarker_list:
                    self.pred_dict[field]()
        
        if 'biomarkers' in self.dict_of_sentences.keys():
            self.pred_dict['biomarkers']()
        
        self.pred_dict['dateofservice']()
        
        if 'menopause' in fields:
            self.pred_dict['menopause']()

    def _get_dates(self):
        self.dates = None
        List = []
        if 'dates' in self.dict_of_sentences.keys():
            for item in self.dict_of_sentences['dates']:
                date_format  = self._return_date_format(item[0],'dates')
                if date_format is not None:
                    tup = item + (date_format,)
                    List.append(tup)
                
        if len(List)>0:
            List = sorted(List, key = lambda x : x[-1])
            self.dates = List

    def _get_biomarker_key_in_dict_of_sentences(self):
        List = []
        for key in self.dict_of_sentences.keys():
            if key in self.biomarker_list:
                List = List + self.dict_of_sentences[key]
        if len(List)>0:
            self.dict_of_sentences['biomarkers'] = List
                
    def get_mid_char(self, s,e):
        return int(s+(e-s)/2)

    def _get_tumor_predictions_on_single_sentence(self, text):
        tumor_predictions = []
        ent_list = []
        sentence_aspect_label_tuple_list = []
        outDict_values = []
        subject = self._returnSubject(context=text)
        if subject != 'self':
            return tumor_predictions

        if "tumor_ner" not in self.BERT_models.keys() or "tumor_nerc" not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['tumor_ner']
        if pandas.isnull(model_dict):
            return tumor_predictions
        
        use_onnx = ('tumor' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=text,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        for ner_chunk_spans in all_ner_chunk_spans:
            for ner_chunk_span in ner_chunk_spans:
                negation_status = get_negation_status(text, ner_chunk_span[0])
                if negation_status:
                    pass
                else:
                    ent_list.append(ner_chunk_span)
                    sentence_aspect_label_tuple_list.append(self.get_tumor_tokenized_sentence_tuple(text, ner_chunk_span[1],
                                                                                            ner_chunk_span[2]))
                    Dict = {}
                    Dict['context'] = text
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['attribute'] = {}
                    Dict['attribute']['tag'] = 'tumor'
                    outDict_values.append(Dict)

    
        results_nerc = self.pred_nerc_tag(outDict_values, "tumor", "tumor", "tumor_nerc")
        for ent, nerc in zip(ent_list, results_nerc):
            if nerc['prediction'].lower() != 'irrelevant':
                tumor_predictions.append((ent[1], ent[2], nerc['prediction']))
        return tumor_predictions

    def _associate_tumor_to_field(self, text, field_start_end_char):
        tumor_predictions = self._get_tumor_predictions_on_single_sentence(text)
        mid_char__tumor = {self.get_mid_char(tup[0], tup[1]): tup[2] for tup in tumor_predictions}
        field_mid_char = self.get_mid_char(field_start_end_char[0], field_start_end_char[1])

        if len(mid_char__tumor) == 0:
            return 'no cancer found'
        elif len(mid_char__tumor) == 1:
            return next(iter(mid_char__tumor.values()))
        else:
            dist__char = {}
            for char, tumor in mid_char__tumor.items():
                d = np.abs(char - field_mid_char)
                if d in dist__char:
                    dist__char[d].append(char)
                else:
                    dist__char[d] = [char]
            return mid_char__tumor[max(dist__char[min(dist__char.keys())])]

    def _get_newline_count(self, text, field_mid_char, neighbor_field_mid_char):
        left_char = min(field_mid_char, neighbor_field_mid_char)
        right_char = max(field_mid_char, neighbor_field_mid_char)
        return text[left_char:right_char].count("\n")
    
    def _is_treatmentresponse_irrelevant(self, text, field_start_end_char, threshold=80):
        """
        If the entity closest to the treatment response phrase is other primary cancer or a comorbidity then return True; else return False
        Approach:
        1. Get all tumor and comorbidity terms along with the midpoint of their start and end characters
        2. Among the above set, get the terms whose midchar is closest to the treatment response phrase midchar. The number of closest terms
         could be 0 (if the above set is empty), 2 (if two terms are equidistant on either side of the tr phrase midchar) or 1 
        3. If the closest set is non-empty and it doesnt contain self.tumor_type then return True; else return False.

        """
        tumor_predictions = self._get_tumor_predictions_on_single_sentence(text)
        comorbidity_regex_match_list = re.finditer(self.regexObj.get_regex_comorbidities()[0],text)
        comorbidity_predictions = [(*regex_match.span(), text[slice(*regex_match.span())]) for regex_match in comorbidity_regex_match_list]
        field_mid_char = self.get_mid_char(field_start_end_char[0], field_start_end_char[1])
        
        mid_char__neighbor_field_set = defaultdict(set) 
        min_dist = threshold
        for start, end, tumor in tumor_predictions + comorbidity_predictions:
            neighbor_field_mid_char = self.get_mid_char(start, end)
            dist = abs(neighbor_field_mid_char-field_mid_char)
            newline_count = self._get_newline_count(text, field_mid_char, neighbor_field_mid_char)
            if dist <= min_dist and newline_count <=1:
                mid_char__neighbor_field_set[dist].add(tumor)
                min_dist = dist
        
        return  len(mid_char__neighbor_field_set[min_dist])>0 and self.tumor_type not in mid_char__neighbor_field_set[min_dist]


    def _get_chunk_spans(self, tokenizer=None, max_seq_length=128, chunk_overlap_factor=0.0):
        max_seq_length = max(max_seq_length-3, 1) #3 for cls, sep tokens.
        max_overlap_seq_length = int(max_seq_length*chunk_overlap_factor) if chunk_overlap_factor>0 else 0

        def _get_overlap_window(max_overlap_seq_length, max_seq_length, window):
            if not window:
                return [], 0
            if max_seq_length <= 0:
                return [], 0
            overlap = []
            overlap_seq_length = 0
            while window:
                sent,s,e,n = window.pop()
                if (overlap_seq_length + n > max_overlap_seq_length) or (overlap_seq_length + n > max_seq_length):
                    return overlap, overlap_seq_length
                overlap = [(sent,s,e,n)] + overlap
                overlap_seq_length += n
            return overlap, overlap_seq_length

        def _get_chunk_span_from_window(window, text):
            if not window:
                return ()
            start_char = window[0][1]
            end_char = window[-1][2]
            n_tokens = sum([span[3] for span in window])
            return (self.text[start_char:end_char],start_char,end_char,n_tokens)

        def _find_sent_token_len(_sent):
            if tokenizer:
                return len(tokenizer.encode(_sent.replace('\n',' ').strip(), add_special_tokens=False))
            else:
                return len(_sent.split())*3

        def _get_sub_sent_chunks(sent_span):
            sentObj = sent_span[0]
            if sent_span[3] <= max_seq_length:
                return [(sentObj.text, *sent_span[1:])]
            # using token.idx instead of token.start_char to comply with spacy token attributes
            token_spans = [(token.text, token.idx) for token in self.docObj if (token.sent.start_char==sentObj.start_char) and (token.text.replace('\n',' ').strip())]
            token_spans = [(*curr_span, next_span[1]) for curr_span,next_span in zip(token_spans,token_spans[1:]+[('',sentObj.end_char)])]
            token_spans = [(self.text[s:e],s,e, _find_sent_token_len(self.text[s:e])) for _,s,e in token_spans]
            return _aggregate_spans(token_spans)

        def _aggregate_spans(spans):
            chunk_spans = []
            window = []
            chunk_length = 0
            for span in spans:
                span_seq_length = span[3]
                if chunk_length and (chunk_length + span_seq_length > max_seq_length):
                    chunk_span = _get_chunk_span_from_window(window, self.text)
                    chunk_spans.append(chunk_span)
                    window,chunk_length = _get_overlap_window(max_overlap_seq_length, max_seq_length-span_seq_length, window)
                window.append(span)
                chunk_length += span_seq_length

            chunk_span = _get_chunk_span_from_window(window, self.text)
            chunk_spans.append(chunk_span)
            return chunk_spans

        sent_spans = [(sent, sent.start_char, sent.end_char, _find_sent_token_len(sent.text)) for sent in self.docObj.sents]
        sub_sent_spans = [sub_sent_span for sent_span in sent_spans for sub_sent_span in _get_sub_sent_chunks(sent_span)]
        return _aggregate_spans(sub_sent_spans)

    
    def get_natural_chunk_dict(self):
        self.natural_chunk_dict = {}
        max_seq_lengths = list(set([self.BERT_models[key]['max_seq_length'] for key in self.BERT_models.keys()]))
        if self.BERT_models.keys():
            _tokenizer_model = list(self.BERT_models.keys())[0]
            _tokenizer = self.BERT_models[_tokenizer_model]['tokenizer']
            for max_seq_length in max_seq_lengths:
                self.natural_chunk_dict[max_seq_length] = self._get_chunk_spans(tokenizer=_tokenizer, max_seq_length=max_seq_length, chunk_overlap_factor=0)


    def get_sub_sent_spans(self, tokenizer, max_seq_length):
        max_seq_length = max(max_seq_length-12, 1) #9 tokens for text_b in classification, 3 for cls, sep tokens.
        if pandas.isnull(tokenizer):
            return []
            
        def _get_chunk_span_from_window(window):
            if not window:
                return ()
            start_char = window[0][1]
            end_char = window[-1][2]
            n_tokens = sum([span[3] for span in window])
            return (self.text[start_char:end_char],start_char,end_char,n_tokens)

        def _find_sent_token_len(_sent):
            if tokenizer:
                return len(tokenizer.encode(_sent.replace('\n',' ').strip(), add_special_tokens=False))
            else:
                return len(_sent.split())*3

        def _get_sub_sent_chunks(sent_span):
            sentObj = sent_span[0]
            if sent_span[3] <= max_seq_length:
                return [(sentObj.text, *sent_span[1:])]
            # using token.idx instead of token.start_char to comply with spacy token attributes
            token_spans = [(token.text, token.idx) for token in self.docObj if (token.sent.start_char==sentObj.start_char) and (token.text.replace('\n',' ').strip())]
            token_spans = [(*curr_span, next_span[1]) for curr_span,next_span in zip(token_spans,token_spans[1:]+[('',sentObj.end_char)])]
            token_spans = [(self.text[s:e],s,e, _find_sent_token_len(self.text[s:e])) for _,s,e in token_spans]
            return _aggregate_spans(token_spans)

        def _aggregate_spans(spans):
            chunk_spans = []
            window = []
            chunk_length = 0
            for span in spans:
                span_seq_length = span[3]
                if chunk_length and (chunk_length + span_seq_length > max_seq_length):
                    chunk_span = _get_chunk_span_from_window(window)
                    chunk_spans.append(chunk_span)
                    window,chunk_length = [], 0
                window.append(span)
                chunk_length += span_seq_length

            chunk_span = _get_chunk_span_from_window(window)
            chunk_spans.append(chunk_span)
            return chunk_spans

        sent_spans = [(sent, sent.start_char, sent.end_char, _find_sent_token_len(sent.text)) for sent in self.docObj.sents]
        sub_sent_spans = [sub_sent_span for sent_span in sent_spans for sub_sent_span in _get_sub_sent_chunks(sent_span)]
        return sub_sent_spans


    def get_central_tokenized_sentence_tuple(self, tokenizer, sub_sent_spans, start_char, end_char, max_seq_length, aspect_encloser=None):
        """
        Given sub_sent_spans (i.e. sentences of <max_seq_length), entity offsets i.e. start_char and end_char and aspect_encloser, this function
        returns the sentence_aspect_label_tuple
        
        Parameters: 
            - sub_sent_spans: list of tuple of setences with their offsets and token length: [(sent, sent_start_char, sent_end_char, token_len)]
            - start_char: entity start char wrt self.text
            - end_char: entity end_char wrt self.text
            - max_seq_length: max_seq_length of the central chunk
            - aspect_encloser (optinal): 
                - default: None will not enclose entity in text_a and text_b
                - tuple(start_encloser, end_encloser); example: ('<', '>')

        Returns:
            - returns a tuple (text_a, text_b, label); label will be an empty string
        """
        
        def _find_sent_token_len(_sent):
            if tokenizer:
                return len(tokenizer.encode(_sent.replace('\n',' ').strip(), add_special_tokens=False))
            else:
                return len(_sent.split())*3

        def _get_sentence_idx(sub_sent_spans, start_char, end_char):
            sent_idxs = [sub_sent_span[1] for sub_sent_span in sub_sent_spans]
            center_sent_start = bisect.bisect_right(sent_idxs, start_char)-1
            center_sent_end = bisect.bisect_right(sent_idxs, end_char)-1
            return center_sent_start, center_sent_end

        def _get_enclosed_text(text_a, ent_start_char, ent_end_char, text_b, encloser):
            text_b = encloser[0]+text_b+encloser[1]
            text_a = text_a[:ent_start_char] + encloser[0] +text_a[ent_start_char:ent_end_char]  + encloser[1] + text_a[ent_end_char:]
            return text_a, text_b

        def _get_window_chunk(tokenizer, context, aspect, window):
            match = re.search(re.escape(aspect), context)
            if not match:
                return context
            aspect_start_char = match.start()
            aspect_end_char = match.end()
            _tokens = tokenizer.tokenize(context)
            _index_tracker = 0
            _context_spliter = 0
            all_tokens = []
            _context = context[:]
            for _token in _tokens:
                _index_tracker = _index_tracker+_context_spliter
                _context = _context[_context_spliter:]
                if _token.startswith('##'):
                    _token = _token[2:]
                _match = re.search(re.escape(_token), _context, flags=re.IGNORECASE)
                all_tokens.append((_match.group(), _index_tracker+_match.start(), _index_tracker+_match.end()))
                _context_spliter = _match.end()
        
            all_token_info = [_token[1] for _token in all_tokens]
            aspect_start_token = bisect.bisect_left(all_token_info, aspect_start_char)-1
            aspect_end_token = bisect.bisect_right(all_token_info, aspect_end_char)-1
            
            aspect_tokens_len = aspect_end_token - aspect_start_token
            window = window - aspect_tokens_len
            
            after_context_end = all_tokens[min(len(all_tokens)-1, aspect_end_token-1+ window//2)][2]
            before_context_start = all_tokens[max(0, aspect_start_token-window//2)][1]
            
            return context[before_context_start:after_context_end]

        def _check_aspect_in_phrase(phrase, aspect):
            return aspect in phrase

        center_text = ""
        text_b = self.text[start_char:end_char]
        # find the sentence offset of ner entity
        center_sent_start, center_sent_end = _get_sentence_idx(sub_sent_spans, start_char, end_char)
        global_center_sent_start = center_sent_start
        global_center_sent_end = center_sent_end
        
        # same sentence contains the entity
        if center_sent_start==center_sent_end:
            center_text = sub_sent_spans[center_sent_start][0]
        else:#entity spans across 2 or more sentences
            prev_sent_end = sub_sent_spans[center_sent_start][2]
            center_text = sub_sent_spans[center_sent_start][0]
            for i in range(center_sent_start+1, center_sent_end+1, 1):
                joining_str = ' '*(sub_sent_spans[i][1]-prev_sent_end)
                center_text = center_text + joining_str + sub_sent_spans[i][0]
                prev_sent_end = sub_sent_spans[i][2]

        #add entity enclosing tokens <> around entity in text_a, and text_b
        if aspect_encloser:
            center_text, text_b = _get_enclosed_text(center_text, start_char-sub_sent_spans[center_sent_start][1], end_char-sub_sent_spans[center_sent_start][1], text_b, aspect_encloser)

        #set window size
        max_seq_length = max(max_seq_length-(3+_find_sent_token_len(text_b)), 1) #3 for cls, sep tokens and; dynamic text_b tokens
        #find the length of centre text
        center_text_token_len = _find_sent_token_len(center_text)
        # if center_text is <= max_seq_len: consider += max_seq_length/s
        if center_text_token_len > max_seq_length:
            center_text = _get_window_chunk(tokenizer, center_text, text_b, max_seq_length)
            return (center_text, text_b, "")
        else:
            prev = False
            while center_text_token_len < max_seq_length:
                if center_sent_start-1<0 and center_sent_end+1>=len(sub_sent_spans):
                    break
                if center_sent_start-1<0:
                    if (center_text_token_len + sub_sent_spans[center_sent_end+1][3] < max_seq_length):
                        center_text_token_len += sub_sent_spans[center_sent_end+1][3]
                        center_sent_end += 1
                    else:
                        break
                elif center_sent_end+1>=len(sub_sent_spans):
                    if (center_text_token_len + sub_sent_spans[center_sent_start-1][3] < max_seq_length):
                        center_text_token_len += sub_sent_spans[center_sent_start-1][3]
                        center_sent_start -= 1
                    else:
                        break
                elif prev==False:
                    if (center_text_token_len + sub_sent_spans[center_sent_end+1][3] < max_seq_length):
                        center_text_token_len += sub_sent_spans[center_sent_end+1][3]
                        center_sent_end += 1
                        prev = True
                    else:
                        break
                elif prev==True:
                    if (center_text_token_len + sub_sent_spans[center_sent_start-1][3] < max_seq_length):
                        center_text_token_len += sub_sent_spans[center_sent_start-1][3]
                        center_sent_start -= 1
                        prev = False
                    else:
                        break
            
            final_text = ""
            prev_sent_end = None
            joining_str = ''
            for i in range(center_sent_start, global_center_sent_start, 1):
                if prev_sent_end:
                    joining_str = ' '*(sub_sent_spans[i][1]-prev_sent_end)
                final_text = final_text + joining_str + sub_sent_spans[i][0]
                prev_sent_end = sub_sent_spans[i][2]
            
            final_text = final_text + joining_str + center_text
            prev_sent_end = sub_sent_spans[global_center_sent_end][2]
            
            for i in range(global_center_sent_end+1, center_sent_end, 1):
                if prev_sent_end:
                    joining_str = ' '*(sub_sent_spans[i][1]-prev_sent_end)
                final_text = final_text + joining_str + sub_sent_spans[i][0]
                prev_sent_end = sub_sent_spans[i][2]
                
            return (final_text, text_b, "")
            

    def _convert_to_dict(self, field = None, field_sentence_list = None ):
        
        if field_sentence_list is None:
            return None
        else:
            List = []
            if field=='metstatus':
                met_site_list, met_site_prob_df  = self.return_met_site_dl(field_sentence_list)
                met_site_list_by_ner_model = self.return_metsite_ner_based(field_sentence_list)
            count = -1
            for item in field_sentence_list:
                count = count + 1
                Dict = {}
                Dict['confidence'] = ""
                if len(item)==6:
                    Dict['prediction'] = item[0]
                elif len(item)>6:
                    Dict['prediction'] = item[6]
                    if len(item)>7:
                        if field != 'metstatus':
                            Dict['confidence'] = {field : item[7]}
                        else:
                            Dict['confidence'] = {'mets1' : item[7], 'mets2' : item[8]}
                else:
                    continue
                Dict['start_char'] = item[1]
                Dict['end_char'] = item[2]
                Dict['context'] = item[3]
                Dict['context_start_char'] = item[4]
                Dict['context_end_char'] = item[5]
                if field in ['hist']:
                    Dict['base_prediction'] = item[0]
                
                Dict['attribute'] = {}
                if field == 'radiation':
                    Dict['confidence'] = item[7]
                    Dict['attribute']['named-entities'] = item[8]
                    Dict['attribute']['aspect'] = item[0]
                
                ## multi cancer
                if field in cancer_pred_fields:
                    field_start_char = item[1] - item[4]
                    field_end_char = item[2] - item[4]
                    Dict['attribute']['cancer_pred'] = {}
                    Dict['attribute']['cancer_pred']['prediction'] = self._associate_tumor_to_field(item[3].lower(),
                                                                                                    (field_start_char, field_end_char))
                    Dict['attribute']['cancer_pred']['confidence'] = 0.8
                    
                    
                Dict['attribute']['subject'] = self._returnSubject(Dict['context'],Dict['context_start_char'],Dict['context_end_char'],Dict['start_char'],Dict['end_char'])

                if 'date' in field:
                    Dict['attribute']['validDate'] = False if self._return_date_format(Dict['prediction'],field) is None else True
                if 'dateofservice' in field:
                    Dict['attribute']['dos_type'] = self.dos_type
                if field in self.fields_with_negation:
                    negation_status = get_negation_status(Dict['context'],item[0])
                    if negation_status:
                        Dict['attribute']['status'] = 'negated'
                if field == 'metstatus':
                    if Dict['prediction']=='yes':
                        rule_based_site_of_mets = return_site_of_metastasis(item[3])
                        
                        if (met_site_list is not None) :
                            common_rule_dl = list(set(rule_based_site_of_mets).intersection(set(met_site_list[count].split(";"))))
                            if len(common_rule_dl)>0:
                                Dict['attribute']['site'] =  ";".join(common_rule_dl)
                                Dict['attribute']['confidence_site'] = dict(met_site_prob_df.iloc[count])
                            elif len(met_site_list_by_ner_model[count]) > 0:
                                Dict['attribute']['site'] =  ";".join(met_site_list_by_ner_model[count])
                                Dict['attribute']['confidence_site'] = {
                                    site_item: 0.8
                                    for site_item in met_site_list_by_ner_model[count]
                                }
                            else:
                                Dict['attribute']['site'] =  ''
                                Dict['attribute']['confidence_site'] = {
                                    'brain': 0, 'lung': 0, 'bone':0,
                                    'site unknown':0.29, 'liver': 0, 
                                    'others': 0, 'lymp node': 0,
                                    'distant lymph node': 0
                                }
                        elif len(met_site_list_by_ner_model[count]) > 0:
                            Dict['attribute']['site'] = ";".join(met_site_list_by_ner_model[count])
                            Dict['attribute']['confidence_site'] = {
                                site_item: 0.8
                                for site_item in met_site_list_by_ner_model[count]
                            }        
                            # print('Since common rule dl is 0 goinf with ner model : ')
                            # print(met_site_list_by_ner_model[count])
                        else:
                            Dict['attribute']['site'] =  ''
                            Dict['attribute']['confidence_site'] = {
                                'brain': 0, 'lung': 0, 'bone':0,
                                'site unknown':0.29, 'liver': 0, 
                                'others': 0, 'lymp node': 0,
                                'distant lymph node': 0
                            }
                
                if field == 'stage':
                    Dict['attribute']['confirmatory_for_initial_diag'] = item[8]
                if field == 'grade':
                    Dict['attribute']['irrelevance'] = 'yes' if item[8].lower()=='no' else 'no'
                if field in ['stage','stage_imputed_from_tnm']:
                    normalized_stage, coarse_stage = normalize_stage(self.stage_normalization_df,Dict['prediction'])

                    Dict['prediction'] = normalized_stage.strip()
                    Dict['attribute']['coarse_stage'] = coarse_stage.strip()
                List.append(Dict)
            output = List
        return output
    

    def _return_date_format(self,date,field):
        try:
            temp = pandas.to_datetime(date)
            if temp < datetime.datetime.now():
                outp = temp
            else:
                if field == 'dateofbirth':
                    outp = temp.replace(year = temp.year - 100)  ## this is because dob of 21/10/21 is taken as 2017 instead of 1917
                else:
                    outp = None
        except Exception:
            try:
                temp = datetime.datetime.strptime(date,'%m/%y')
                if temp < datetime.datetime.now():
                    outp = temp
                    
                else:
                    if field == 'dateofbirth':
                        outp = temp.replace(year = temp.year - 100)  ## this is because dob of 21/10/21 is taken as 2017 instead of 1917
                    else:
                        outp = None
            except Exception:
                outp = None
        if field=='dateofservice':
            if outp is not None:
                difference = datetime.datetime.now() - outp
                difference_in_years = (difference.days + difference.seconds/86400)/365.2425
                if difference_in_years > 20:  ## because dos sojmetimes picks up dob inn case no other date is available, putting a rule to eliminate dos more than 25 years old
                    outp = None
            
        return outp


    def _returnSubject(self, context = None, context_start_char = None, context_end_char = None, start_char = None, end_char = None):
        return "self"


    def _returnSubjectOld(self, context = None, context_start_char = None, context_end_char = None, start_char = None, end_char = None):
        text = context
        ##sentence_list = re.split(";|,",text)
        ##start_char_list = []
        ##context_end_char = context_end_char - context_start_char
        ##start_char = start_char - context_start_char
        ##end_char = end_char - context_start_char
        ##context_start_char = 0
        ##Len = context_start_char
        ##for sent in sentence_list:
        ##    start_char_list.append(Len)
        ##    Len = Len+len(sent)+1

        ##sentence_start_char_chosen = context_start_char
        ##sentence_end_char_chosen  = context_end_char
        ##for i in range(len(sentence_list)-1):
        ##    if (start_char >= start_char_list[i]) & (end_char <= start_char_list[i+1]):
        ##        sentence_start_char_chosen = start_char_list[i]
        ##        sentence_end_char_chosen = start_char_list[i+1]
        ##        break
        ##    else:
        ##        pass
        ##text = text[sentence_start_char_chosen:sentence_end_char_chosen]


        outp = 'self'
        if len(re.findall('\\b%ss?\\b'%('grandmother'),text.lower()))>0:
            outp = 'grandmother'
        elif len(re.findall('\\b%ss?\\b'%('grandfather'),text.lower()))>0:
            outp = 'grandfather'
        elif len(re.findall('\\b%ss?\\b'%('mother'),text.lower()))>0:
            outp = 'mother'
        elif len(re.findall('\\b%ss?\\b'%('father'),text.lower()))>0:
            outp = 'father'
        elif len(re.findall('\\b%ss?\\b'%('sister'),text.lower()))>0:
            outp = 'sister'
        elif len(re.findall('\\b%ss?\\b'%('brother'),text.lower()))>0:
            outp = 'brother'
        elif len(re.findall('\\b%ss?\\b'%('uncle'),text.lower()))>0:
            outp = 'uncle'
        elif len(re.findall('\\b%ss?\\b'%('aunt'),text.lower()))>0:
            outp = 'aunt'
        elif len(re.findall('\\b%ss?\\b'%('daughter'),text.lower()))>0:
            outp = 'daughter'
        elif len(re.findall('\\b%ss?\\b'%('son'),text.lower()))>0:
            outp = 'son'
        elif len(re.findall('\\b%ss?\\b'%('cousin'),text.lower()))>0:
            outp = 'cousin'
        elif len(re.findall('\\b%ss?\\b'%('nephew'),text.lower()))>0:
            outp = 'nephew'
        elif len(re.findall('\\b%ss?\\b'%('niece'),text.lower()))>0:
            outp = 'niece'
        elif len(re.findall('\\b%ss?\\b'%('husband'),text.lower()))>0:
            outp = 'husband'
        elif len(re.findall('\\b%ss?\\b'%('wife'),text.lower()))>0:
            outp = 'wife'
        elif len(re.findall('\\b%ss?\\b'%('friend'),text.lower()))>0:
            outp = 'friend'
        elif len(re.findall('\\b%ss?\\b'%('fhx'),text.lower()))>0:
            outp = 'family history'
        elif len(re.findall('\\b%ss?\\b'%('family'),text.lower()))>0:
            outp = 'family history'
        else:
            pass
        return outp        
    
    

    
    def get_pred_dict(self):
        Dict = OrderedDict()
        Dict['tumor'] = self.get_pred_tumor
        Dict['oncologyv2'] = self.get_pred_bertneroncologyv2
        Dict['medication'] = self.get_pred_drugs
        # Dict['tnmt'] = self.get_pred_TNMT
        # Dict['tnmn'] = self.get_pred_TNMN
        # Dict['tnmm'] = self.get_pred_TNMM
        # Dict['tnm'] = self.get_pred_TNM
        Dict['stage'] = self.get_pred_stage_tnm
        Dict['stage_imputed_from_tnm'] = self.get_pred_stage_imputed_from_tnm
        # Dict['metstatus'] = self.get_pred_mets
        Dict['metastasis'] = self.get_pred_metastasis
        Dict['grade'] = self.get_pred_grade
        Dict['hist'] = self.get_pred_hist
        Dict['secondary_tumor'] = self.get_pred_secondary_tumor
        
        Dict['ecog'] = self.get_pred_ECOG
        Dict['karnofsky'] = self.get_pred_karnofsky
        Dict['dateofbirth'] = self.get_pred_dateofbirth
        Dict['dateofdeath'] = self.get_pred_dateofdeath
        Dict['dateofservice'] = self.get_pred_dateofservice
        Dict["race_ethnicity"] = self.get_pred_race_ethnicity
        # Dict['race'] = self.get_pred_race
        # Dict['ethnicity'] = self.get_pred_ethnicity
        Dict['gender'] = self.get_pred_gender
        Dict['biomarkers'] = self.get_pred_biomarker_name
        Dict['menopause'] = self.get_pred_menopause
        Dict['smoking'] = self.get_pred_smoking

        Dict['surgery'] = self.get_pred_surgery

        Dict['radiation'] = self.get_pred_radiation
        Dict['pcsubtype'] = self.get_pred_pcsubtype
        Dict['alcohol'] = self.get_pred_alcohol
        Dict['treatmentresponse'] = self.get_pred_treatmentresponse
        Dict['performance_status'] = self.get_pred_performance_status
        Dict['death'] = self.get_pred_death
        Dict['comorbidities'] = self.get_pred_comorbidities
        Dict['site'] = self.get_pred_site
        self.pred_dict = Dict



    def get_pred_TNMT(self):
        if 'tnmt' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('tnmt',self.dict_of_sentences['tnmt'])
            if outp is not None:
                self.outDict['tnmt'] = outp
        
    
    def get_pred_TNMN(self):
        
        if 'tnmn' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('tnmn',self.dict_of_sentences['tnmn'])
            if outp is not None:
                self.outDict['tnmn'] = outp


    
    def get_pred_TNMM(self):
        
        if 'tnmm' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('tnmm',self.dict_of_sentences['tnmm'] )
            if outp is not None:
                self.outDict['tnmm'] = outp
    
    
    
    def get_pred_TNM(self):
        
        if 'tnm' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('tnm',self.dict_of_sentences['tnm'])
            if outp is not None:
                self.outDict['tnm'] = outp
    
    
    
    def get_pred_stage_imputed_from_tnm(self):
        if 'tnm' in self.dict_of_sentences.keys():
            List = []
            for item in self.dict_of_sentences['tnm']:
                base_prediction = item[0]
                # print(base_prediction)
                prediction = get_stage_from_tnm(tnm = base_prediction, df = self.tnm_stage_mapping_df, regexObj = self.regexObj )
                if prediction is None:
                    continue
                tup = item + (prediction,)
                List.append(tup)

            if len(List)>0:
                outp = self._convert_to_dict('stage_imputed_from_tnm',List)
                self.outDict['stage_imputed_from_tnm'] = outp


    def get_pred_stage_tnm(self):
        if not 'oncologyv2' in self.outDict:
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = \
            self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list('stage')

        if sentence_aspect_label_tuple_list:
            nerc_results, _ = self.get_bert_classification_inference('stage_nerc', 'stage', sentence_aspect_label_tuple_list)
            assert len(nerc_results) == len(sentence_aspect_label_tuple_list), "Mismatch between input length and output length of stage NERC model"

            tnmt_outDict_values = []
            tnmn_outDict_values = []
            tnmm_outDict_values = []
            outDict_values = []

            if len(nerc_results) == len(ner_out_dict_values):
                for i, item in enumerate(ner_out_dict_values):
                    if nerc_results[i].lower() == "multi-labelled":
                        ml_stage_tnm, sentence_aspect_multi_tuple = postprocess_multilabelled_staging(self, ner_out_dict_values[i])
                        nerc_multi_results, _ = self.get_bert_classification_inference('stage_nerc', 'stage', sentence_aspect_multi_tuple)
                        for i, item in enumerate(ml_stage_tnm):
                            staging_label = get_staging_entities(nerc_multi_results[i])
                            item['prediction'] = nerc_multi_results[i]
                            if nerc_multi_results[i].lower()=="multi-labelled" or nerc_multi_results[i].lower()=="irrelevant":
                                item["attribute"]['status'] = 'irrelevant'
                                item["attribute"]['is_irrelevant'] = True
                                outDict_values.append(item)
                            elif staging_label.lower() == 'tstaging':
                                tnmt_outDict_values.append(item)
                            elif staging_label.lower() == 'nstaging':
                                tnmn_outDict_values.append(item)
                            elif staging_label.lower() == 'mstaging':
                                tnmm_outDict_values.append(item)
                            else:
                                item["attribute"]['status'] = 'relevant'
                                outDict_values.append(item)
                        continue
                        
                    staging_label = get_staging_entities(nerc_results[i])
                    item['prediction'] = nerc_results[i]
                    if staging_label.lower() == 'tstaging':
                        tnmt_outDict_values.append(item)
                    elif staging_label.lower() == 'nstaging':
                        tnmn_outDict_values.append(item)
                    elif staging_label.lower() == 'mstaging':
                        tnmm_outDict_values.append(item)
                    else:
                        item["attribute"]['status'] = 'relevant'
                        item["attribute"]['is_irrelevant'] = (nerc_results[i].lower().strip() == 'irrelevant')
                        outDict_values.append(item)

            if len(outDict_values)>0:
                self.outDict['stage'] = outDict_values
            if len(tnmt_outDict_values)>0:
                self.outDict['tnmt'] = tnmt_outDict_values
            if len(tnmn_outDict_values)>0:
                self.outDict['tnmn'] = tnmn_outDict_values
            if len(tnmm_outDict_values)>0:
                self.outDict['tnmm'] = tnmm_outDict_values
                

    def get_pred_ECOG(self):
        
        if 'ecog' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('ecog',self.dict_of_sentences['ecog'])
            if outp is not None:
                self.outDict['ecog'] = outp


    def get_pred_karnofsky(self):
        
        if 'karnofsky' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('karnofsky',self.dict_of_sentences['karnofsky'])
            if outp is not None:
                self.outDict['karnofsky'] = outp
             
    

    def get_pred_dateofbirth(self):
        
        if 'dateofbirth' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('dateofbirth',self.dict_of_sentences['dateofbirth'])
            idx_del = []
            if outp is not None:
                for i,item in enumerate(outp):
                    dt = item['prediction']
                    dt = self._return_date_format(dt,'dateofbirth')
                    if pandas.isnull(dt):
                        idx_del.append(i)
                    else:
                        outp[i]['prediction'] = str(dt).split()[0]
                
                outp = [item for i,item in enumerate(outp) if i not in idx_del]
                if len(outp) > 0:
                    self.outDict['dateofbirth'] = outp



    def get_pred_dateofdeath(self):
        
        if 'dateofdeath' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('dateofdeath',self.dict_of_sentences['dateofdeath'])
            if outp is not None:
                self.outDict['dateofdeath'] = outp            


            
    def get_pred_dateofservice(self):
        dateofservice_priority_list = ['dateofservicedateofservice', 'dateofservicedateofcollection', 'dateofservicedateofadmission',
                                       'dateofservicedateofdate' , 'dateofservicedos' ,'dateofservicedatesigned'] ## sorted in descending order of priority
        
        for key in dateofservice_priority_list:
            List = []
            if key in self.dict_of_sentences.keys():
                ## First choice
                self.dos_type = key
                for item in self.dict_of_sentences[key]:
                    date_format  = self._return_date_format(item[0],'dateofservice')
                    if date_format is not None:
                        tup = item + (date_format,)
                        List.append(tup)
                    
            if len(List)>0:
                List = sorted(List, key = lambda x : x[-1])
                List = [List[-1][:-1]]  ## If there are multiple values, keep the highest value (for example multiple signed by date, keep highest signed by date)
                outp = self._convert_to_dict('dateofservice',List)
                if outp is not None:
                    self.outDict['dateofservice'] = outp
                    break
            

            ##pass
        if self.dates is not None:
            List = [self.dates[-1][:-1]]  ## last element removed to keep string value in first position and remove timestamp value in last position
            outp = self._convert_to_dict('dateofservice',List)
            self.outDict['dateofservice_maxdate'] = outp



    def get_pred_hist(self):
        if not 'oncologyv2' in self.outDict:
            return

        if 'hist_irr' not in self.BERT_models.keys():
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list(
            'hist')
        
        hist_irr_results, _ = self.get_bert_classification_inference('hist_irr', 'hist', sentence_aspect_label_tuple_list)

        assert len(hist_irr_results) == len(ner_out_dict_values), "Histlogy Irr and NER prediction length mismatch"

        asserted_ner_outdict_values = []

        for i in range(len(ner_out_dict_values)):
            item = ner_out_dict_values[i]
            item['attribute']['is_irrelevant'] = (hist_irr_results[i].lower().strip() == "irrelevant")
            asserted_ner_outdict_values.append(item)

        if len(ner_out_dict_values) > 0:
            self.outDict['hist'] = asserted_ner_outdict_values


    def get_tumor_tokenized_sentence_tuple(self, text, start, end):
        token = '<TUMOR>'
        text_a =  text[0:start] + token + text[end:]
        return (text_a, token, "")

    def get_tumor_relevance(self, sentence_aspect_label_tuple_list):
        if len(sentence_aspect_label_tuple_list) == 0:
            return [], None

        if 'tumor_irr' in self.BERT_models.keys():
            model_dict = self.BERT_models['tumor_irr']
            if not pandas.isnull(model_dict):
                #####################profiling##############################
                algo_fifth_step = time.time()
                #####################profiling##############################
                if ('tumor' in self.use_onnx) and model_dict.get('use_onnx', False):
                    results, prob_df = bert_inference(
                        sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                        model=model_dict['model'], device=model_dict['device'],
                        label_list=model_dict['label_list'],
                        max_seq_length=model_dict['max_seq_length'],
                        tokenizer=model_dict['tokenizer'],
                        output_mode=model_dict['output_mode'],
                        model_type=model_dict['model_type'],
                        use_onnx=True)
                else:
                    results, prob_df = bert_inference(
                        sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                        model=model_dict['model'], device=model_dict['device'],
                        label_list=model_dict['label_list'],
                        max_seq_length=model_dict['max_seq_length'],
                        tokenizer=model_dict['tokenizer'],
                        output_mode=model_dict['output_mode'],
                        model_type=model_dict['model_type'],
                        use_onnx=False)

                #####################profiling##############################
                bert_model_surgery = time.time() - algo_fifth_step
                self.time_dict['bert_model_tumor_irr'] = bert_model_surgery
                #####################profiling##############################

                return results, prob_df
            else:
                raise ("Tumor Irrelevance Model missing")

        return [], None



    def get_pred_tumor(self):
        # """ Tumor prediction model using NER model"""
        if "tumor_ner" not in self.BERT_models.keys() or "tumor_nerc" not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['tumor_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]
        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################

        use_onnx = ('tumor' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx,
            recalibration_dt_model_dict_list=[self.decision_tree_models.get('tumor_ner')]
        )

        #####################profiling##############################
        bert_model_tumor = time.time() - algo_eigth_step
        self.time_dict['bert_model_tumor_ner'] = bert_model_tumor
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "tumor - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence, recalibration_result
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['prediction'] = ner_chunk_span[0]
                    Dict['attribute']['tag'] = 'tumor'
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                    Dict['context_start_char'],
                                                                    Dict['context_end_char'],
                                                                    Dict['start_char'], Dict['end_char'])
                    Dict['attribute']['status'] = 'relevant'
                    if ner_chunk_span[5]==1:
                        Dict['attribute']['recalibration_prediction'] = 'relevant'
                    elif ner_chunk_span[5]==0:
                        Dict['attribute']['recalibration_prediction'] = 'irrelevant'
                    elif ner_chunk_span[5]==-1:
                        Dict['attribute']['recalibration_prediction'] = ''
                    outDict_values.append(Dict)
        
        outDict_values = self.pred_nerc_tag(outDict_values, "tumor", "tumor", "tumor_nerc")

        if len(outDict_values) > 0:
            self.outDict['tumor'] = outDict_values

    def get_pred_secondary_tumor(self):
        ##print("#### Inside get pred secondary tumor #######")
        nsclc_syns = ["non-small-cell carcinoma","non small-cell carcinoma","non-small cell carcinoma","non small cell carcinoma","non-small-cell lung cancer","non small-cell lung cancer","non-small cell lung cancer","non small cell lung cancer","nsclc","non-small cell lung cancer"]
        sclc_syns = ["sclc","small-cell carcinoma","small cell carcinoma","small cell lung cancer","small cell lung carcinoma","small-cell lung cancer"]
        # nsclc_stages = ['IB', 'II', 'I', 'IV', 'IA', 'IIIC', 'IIIB', 'IVB', 'IIIA', 'IIA', 'IIB', 'III', 'IVA', 'IC',  'IE',  'IIC', 'IIIc', 'Ia', 'Ib', '1B', 'IIIa','1','2','3','4']
        # nsclc_stages = [x.lower().strip() for x in nsclc_stages]
        sclc_stages = ['extensive','limited']
        
        nsclc_histologies = ['adenocarcinoma','squamous']
        sclc_histologies= ['small cell', 'oat', 'small-cell', 'neuroendocrine']
        prediction_list = ['nsclc', 'sclc']
            
        List = []
        tumor_list = []
        relevant_tumor_output = [item for item in self.outDict.get('tumor',[]) 
                                      if item['attribute']['status'].lower()!="irrelevant" and item['prediction'].lower()!="irrelevant"]
        for item in relevant_tumor_output:
            base_prediction = item['base_prediction'].lower().strip()
            if ((base_prediction in self.primary_tumor_type_mapping_dict.keys())
                or (CONFIG.USE_INPUT_TUMOR and 'lung' in self.tumor_type)):
                prediction = ""
                if (CONFIG.USE_INPUT_TUMOR and 'lung' in self.tumor_type):
                    primary_tumor = 'lung cancer'
                else:
                    primary_tumor = self.primary_tumor_type_mapping_dict[base_prediction].lower()
                tumor_list.append(primary_tumor)
                if (('lung' in primary_tumor)
                    or (CONFIG.USE_INPUT_TUMOR and 'lung' in self.tumor_type)):
                    if np.any([base_prediction==x for x in nsclc_syns]):
                        prediction = 'nsclc'
                    elif np.any([base_prediction==x for x in sclc_syns]):
                        prediction ='sclc'
                    else:
                        pass
                    if prediction in prediction_list:
                        item_tup = (item["base_prediction"], item["start_char"], item["end_char"], item["context"], item["context_start_char"], item["context_end_char"])
                        tup = item_tup + (prediction,)
                        List.append(tup)
                            
                 
        relevant_stage_output = [item for item in self.outDict.get('stage',[]) 
                                      if item['attribute']['status'].lower()!="irrelevant"]
        relevant_stage_output = [item for item in relevant_stage_output 
                                      if not np.any([x in item['context'].lower() for x in self.irrelevant_list_for_stage])]
        if len(relevant_stage_output) > 0:
            if ((len(tumor_list)>0 and 'lung cancer' in tumor_list)
                or (CONFIG.USE_INPUT_TUMOR and 'lung' in self.tumor_type)):

                for item in relevant_stage_output:
                    base_prediction = item["prediction"].lower().strip()
                    context = item["context"]
                    prediction = ""
                    if base_prediction in sclc_stages:
                        prediction = 'sclc'
                    elif not base_prediction.startswith("stage 0"):
                        prediction = 'nsclc'
                    else:
                        pass
                    if prediction in prediction_list:
                        item_tup = (item["base_prediction"], item["start_char"], item["end_char"], item["context"], item["context_start_char"], item["context_end_char"])
                        tup = item_tup + (prediction,)
                        List.append(tup)
                        

        hist_list = []
        if 'hist' in self.outDict.keys():
            if ((len(tumor_list)>0 and 'lung cancer' in tumor_list)
                or (CONFIG.USE_INPUT_TUMOR and 'lung' in self.tumor_type)):

                
                for item in self.outDict['hist']:
                    base_prediction = ' '.join(str(item['base_prediction']).lower().strip().split())

                    prediction = ""
                    if base_prediction in self.histology_mapping_dict.keys():
                        prediction = self.histology_mapping_dict[base_prediction]
                        if prediction.lower().strip().startswith('non-small cell carcinoma'):
                            prediction = 'nsclc'
                        elif (prediction.lower().strip().startswith('small cell carcinoma') 
                        or prediction.lower().strip().startswith('small cell neuroendocrine')):
                            prediction = 'sclc'
                        else:
                            if np.any([y in x for y in nsclc_histologies for x in [base_prediction]]):
                                prediction = 'nsclc'
                            else:
                                pass

                    if prediction in prediction_list:
                        ## tup = item + (prediction,)
                        ## List.append(tup)
                        item_cp = copy.deepcopy(item)
                        item_cp['prediction'] = prediction
                        hist_list.append(item_cp)

        outp = [{}]
        if len(List)>0:
            outp = self._convert_to_dict('secondary_tumor',List)

        if hist_list:
                if outp == [{}]:
                    outp = hist_list
                else:
                    outp.extend(hist_list)

        output = {}
        model_dict = self.xgboost_models['secondary_tumor']

        if not pandas.isnull(model_dict):
            prediction = None
            prediction, prob_score = nlp_predict_secondary_tumor_status(self.text, model_dict['model'],model_dict['word_tokenizer'])
            
            if prediction:           
                output = {'prediction':prediction, 'confidence':prob_score, 'attribute':{'subject':'self'}}
        
        if outp != [{}]:
            op_list = []
            op_list.extend(outp)
            op_list.extend([output])
            self.outDict['secondary_tumor'] = op_list



    def get_pred_medication(self):
        """
        Getting predictions from medication models:
        1) Medication and attributes using SpacyNer Model
        2) Irrelevance model (BERT Model)
        3) Drug-Attribute Relationship model (LSTM Model)
        4) Drug-Date_Type Relationship model (LSTM Model)
        5) Aggregation code
        """
        if self.dl_models['medication_drug_attribute_model'] and \
            self.dl_models['medication_drug_date_model'] and \
            self.BERT_models['medication'] and \
            self.spacy_models['medication_ner']:
            temp_sentences = []
            medication_sentence_dicts_from_regex = []
            for idx, item in enumerate(self.dict_of_sentences.get('medication', [])):
                sentence = item[3]
                sentence_char_start = item[4]
                sentence_char_end = item[5]
                if sentence not in temp_sentences:
                    medication_sentence_dicts_from_regex.append(
                        {
                            'text': sentence,
                            'sentence_id': idx,
                            'sentence_char_start': sentence_char_start,
                            'sentence_char_end': sentence_char_end
                        }
                    )
                    temp_sentences.append(sentence)

            # print("Number of sentences from medication regex on document: {}".format(len(medication_sentence_dicts_from_regex)))
            if len(medication_sentence_dicts_from_regex) > 0:
                # run spacy ner model
                id2sentence_item, input_to_irrelevance_model = prepare_id2sentence_dict_medication(
                    medication_sentence_dicts_from_regex,
                    spacy_model=self.spacy_models['medication_ner']['model']
                )

                # running irrelevance model
                input_to_irrelevance_model_keys = list(input_to_irrelevance_model.keys())
                sentence_aspect_label_tuple_list = [
                    (input_to_irrelevance_model[key]['context'], input_to_irrelevance_model[key]['keyword'], "")
                    for key in input_to_irrelevance_model_keys
                ]
                # the below is done because if there is no drug identified by the 
                if len(sentence_aspect_label_tuple_list) > 0:
                    irrelevance_model_dict = self.BERT_models['medication']
                    if ('medication' in self.use_onnx) and irrelevance_model_dict.get('use_onnx', False):
                        irrelevance_results, irrelevance_prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list, 
                            model=irrelevance_model_dict['model'],
                            device = irrelevance_model_dict['device'], 
                            label_list = irrelevance_model_dict['label_list'],
                            max_seq_length = irrelevance_model_dict['max_seq_length'], 
                            tokenizer=irrelevance_model_dict['tokenizer'],
                            output_mode=irrelevance_model_dict['output_mode'], 
                            model_type=irrelevance_model_dict['model_type'],
                            use_onnx=True
                        )
                    else:
                        irrelevance_results, irrelevance_prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list, 
                            model=irrelevance_model_dict['model'],
                            device = irrelevance_model_dict['device'], 
                            label_list = irrelevance_model_dict['label_list'],
                            max_seq_length = irrelevance_model_dict['max_seq_length'], 
                            tokenizer=irrelevance_model_dict['tokenizer'],
                            output_mode=irrelevance_model_dict['output_mode'], 
                            model_type=irrelevance_model_dict['model_type'],
                            use_onnx=False
                        )
                    irrelevance_prob_list = [
                        row[irrelevance_results[idx]]
                        for idx, (_, row) in enumerate(irrelevance_prob_df.iterrows())
                    ]
                    for irr_out, irr_prob, key in zip(list(irrelevance_results), irrelevance_prob_list, input_to_irrelevance_model_keys):
                        input_to_irrelevance_model[key]['irr_output'] = irr_out
                        input_to_irrelevance_model[key]['irr_prob'] = irr_prob

                    # updating the drug entities in the id2sentence_item data with the irrelevance information
                    for key, val in input_to_irrelevance_model.items():
                        sentence_id, ent_id = key.split('_')
                        sentence_id = int(sentence_id)
                        ent_id = int(ent_id)
                        temp_idx = [i for i, ent in enumerate(id2sentence_item[sentence_id]['entities']) if ent['ent_id'] == ent_id][0]
                        id2sentence_item[sentence_id]['entities'][temp_idx]['irr_output'] = val['irr_output']
                        id2sentence_item[sentence_id]['entities'][temp_idx]['irr_prob'] = val['irr_prob']

                    # getting df for relation predictions
                    drug_attribute_df, drug_date_df = prepare_data_for_medication_pipeline(id2sentence_item)

                    # run drug_attribute_relation model
                    # below are the relationship models dictionaries and their required attributes
                    drug_attribute_lstm_model = self.dl_models['medication_drug_attribute_model']
                    drug_date_lstm_model = self.dl_models['medication_drug_date_model']

                    # getting predictions for drug_attribute_df data
                    drug_attribute_df = drug_relation_inference(
                        model=drug_attribute_lstm_model['model'], df=drug_attribute_df,
                        max_sentence_length=drug_attribute_lstm_model['max_sentence_length'],
                        word_vocab=drug_attribute_lstm_model['word_vocab'],
                        rel_vocab=drug_attribute_lstm_model['rel_vocab'],
                        abs_vocab=drug_attribute_lstm_model['abs_vocab'],
                        rel_pad="[PAD]", abs_pad="[PAD]", word_pad="[PAD]",
                        rel_unk="[UNK]", abs_unk="[UNK]", word_unk="[UNK]",
                        batch_size=32, device=drug_attribute_lstm_model['device'],
                        make_examples_using_sp_chars_func=True, 
                    )

                    # getting predictions for drug_date_df data
                    drug_date_df = drug_relation_inference(
                        model=drug_date_lstm_model['model'], df=drug_date_df,
                        max_sentence_length=drug_date_lstm_model['max_sentence_length'],
                        word_vocab=drug_date_lstm_model['word_vocab'],
                        rel_vocab=drug_date_lstm_model['rel_vocab'],
                        abs_vocab=drug_date_lstm_model['abs_vocab'],
                        rel_pad="[PAD]", abs_pad="[PAD]", word_pad="[PAD]",
                        rel_unk="[UNK]", abs_unk="[UNK]", word_unk="[UNK]",
                        batch_size=32, device=drug_date_lstm_model['device'],
                        make_examples_using_sp_chars_func=False, 
                    )
                    # print(len(drug_attribute_examples), len(drug_date_examples))
                    # aggregate results for each sentence
                    medication_output_dict = aggregate_outputs_after_relations_processing(
                        drug_attribute_df, drug_date_df, id2sentence_item
                    )
                    medication_output_list = prepare_medication_fields_dict_format(medication_output_dict)
                    # print(medication_output_list)
                    self.outDict['medication'] = medication_output_list
                    # give output in proper format


    def get_pred_comorbidity(self):
        if 'comorbidity' in self.dict_of_sentences.keys():
            outp = self._convert_to_dict('comorbidity',self.dict_of_sentences['comorbidity'])
            if outp is not None:
                self.outDict['comorbidity'] = outp        


    def return_metsite_ner_based(self, met_sentence_list):
        """
        Takes in a list of sentences and returns metsites
        extracted using the ner based metsite model.
        """
        # getting the spacy model 
        sp_metsite_model_dict = self.spacy_models['siteofmets_ner']
        if pandas.isnull(sp_metsite_model_dict):
            return [[] for _ in met_sentence_list]
        sp_metsite_model = sp_metsite_model_dict['model']
        outputs = []
        for sentence_tuple in met_sentence_list:
            sentence = sentence_tuple[3].strip()
            temp_ents_list = []
            sp_object = sp_metsite_model(sentence)
            for ent in sp_object.ents:
                if ent.label_ == 'metsite_relevant':
                    ent_text = ent.text.lower().strip()
                    if is_suspicious_metsite(ent_text) == False:
                        temp_ents_list.append(ent_text)
            
            outputs.append(list(set(temp_ents_list))) 
        return outputs


    ##def return_met_site_dl(self, metsite_classifier = None, text = None):
    def return_met_site_dl(self, met_sentence_list):
        # classes = ['bone','brain','liver','lung','site unknown','others','lymph node', 'distant lymph node']
        sentence_aspect_label_tuple_list = [(item[3].lower().strip(' ').strip('.').strip(),None,"") for item in  met_sentence_list]
        outp = None
        prob_df = None
        if 'siteofmets' in self.BERT_models.keys():
            model_dict = self.BERT_models['siteofmets']
            if not pandas.isnull(model_dict):
                if ('metstatus' in self.use_onnx) and model_dict.get('use_onnx', False):
                    results,prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                     model = model_dict['model'], 
                                                     device = model_dict['device'], 
                                                     label_list = model_dict['label_list'], 
                                                     max_seq_length = model_dict['max_seq_length'] ,
                                                     tokenizer = model_dict['tokenizer'], 
                                                     output_mode = model_dict['output_mode'], 
                                                     model_type = model_dict['model_type'], 
                                                     use_onnx=True)
                else:
                    results,prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, model = model_dict['model'], 
                                                     device = model_dict['device'], 
                                                     label_list = model_dict['label_list'], 
                                                     max_seq_length = model_dict['max_seq_length'] ,
                                                     tokenizer = model_dict['tokenizer'], 
                                                     output_mode = model_dict['output_mode'], 
                                                     model_type = model_dict['model_type'], 
                                                     use_onnx=False)
                if len(results)==len(met_sentence_list):
                    outp = results
        return outp, prob_df  



    def get_pred_mets(self):
        met_stages = ['iv','4']
        reg_met_stages = re.compile("|".join(['\\b%s\\b'%(str(x)) for x in met_stages]), re.IGNORECASE)
        local_tnmn = ['n1','n2']
        # met_model_1 = self.ulmfit_models['met_model1']
        # met_model_2 = self.ulmfit_models['met_model2']
        
        def get_micro_macromet_status(context):
            string_micromacromet = "(" + "?P<mets_prediction_7>" + "\\bmicromet[\w]*\\b|\\bmacromet[\w]*\\b" + ")"
            regex_micromacromet = re.compile(string_micromacromet,re.IGNORECASE)
            if len(re.findall(regex_micromacromet,context))>0:
                return True
            else:
                return False
        List = []
        if 'tnmm' in self.outDict.keys():
            for item in self.outDict['tnmm']:
                if 'm1' in item["prediction"].lower():
                    prediction = "yes"
                elif 'm0' in item["prediction"].lower():
                    prediction = "no"
                    context = item["context"]
                    if "tnmn" in self.outDict.keys():
                        tnmn_list_values = [item["prediction"].lower() for item in self.outDict["tnmn"] if item["context"] == context]
                        tnmn_list_values = [_pred for _pred in tnmn_list_values if any([_tnmn in _pred for _tnmn in local_tnmn])]
                        if tnmn_list_values:
                            prediction = "local"
                else:
                    continue
                item_tup = (item["base_prediction"], item["start_char"], item["end_char"], item["context"], item["context_start_char"], item["context_end_char"])
                tup = item_tup + (prediction,)
                List.append(tup)
        if 'stage' in self.outDict.keys():
            for item in self.outDict['stage']:
                context = item["context"]
                if np.any([x in context.lower() for x in self.irrelevant_list_for_stage]):
                    continue
                if len(re.findall(reg_met_stages, item["prediction"]))>0:
                    prediction = "yes"
                else:
                    prediction = "no"
                item_tup = (item["base_prediction"], item["start_char"], item["end_char"], item["context"], item["context_start_char"], item["context_end_char"])
                tup = item_tup + (prediction,)
                List.append(tup)
        
        if 'metstatus' in self.dict_of_sentences.keys():
            sentence_aspect_label_tuple_list = [(item[3].lower().strip(' ').strip('.').strip(),None,"") for item in  self.dict_of_sentences['metstatus']]
            if 'mets1' in self.BERT_models.keys() and 'mets2' in self.BERT_models.keys():
                model_dict_1 = self.BERT_models['mets1']
                model_dict_2 = self.BERT_models['mets2']
                if model_dict_1 is not None and model_dict_2 is not None:
                    
                    #####################profiling##############################
                    algo_sixth_step = time.time()
                    #####################profiling##############################
                    if ('metstatus' in self.use_onnx) and model_dict_1.get('use_onnx', False):
                         results_1, prob_df1 = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                              model = model_dict_1['model'], 
                                                              device = model_dict_1['device'], 
                                                              label_list = model_dict_1['label_list'], 
                                                              max_seq_length = model_dict_1['max_seq_length'] ,
                                                              tokenizer = model_dict_1['tokenizer'], 
                                                              output_mode = model_dict_1['output_mode'], 
                                                              model_type = model_dict_1['model_type'], 
                                                              use_onnx=True)
                    else:
                        results_1, prob_df1 = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                             model = model_dict_1['model'], 
                                                             device = model_dict_1['device'], 
                                                             label_list = model_dict_1['label_list'], 
                                                             max_seq_length = model_dict_1['max_seq_length'] ,
                                                             tokenizer = model_dict_1['tokenizer'], 
                                                             output_mode = model_dict_1['output_mode'], 
                                                             model_type = model_dict_1['model_type'], 
                                                             use_onnx=False)
                    
                    #####################profiling##############################

                    bert_model_met_m1 = time.time()-algo_sixth_step
                    self.time_dict['bert_model_met_m1']=bert_model_met_m1
                    #####################profiling##############################


                    #####################profiling##############################
                    algo_seventh_step = time.time()
                    #####################profiling##############################
                    if ('metstatus' in self.use_onnx) and model_dict_2.get('use_onnx', False):
                        results_2, prob_df2 = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                             model = model_dict_2['model'], 
                                                             device = model_dict_2['device'], 
                                                             label_list = model_dict_2['label_list'],
                                                             max_seq_length = model_dict_2['max_seq_length'] ,
                                                             tokenizer = model_dict_2['tokenizer'], 
                                                             output_mode = model_dict_2['output_mode'], 
                                                             model_type = model_dict_2['model_type'], 
                                                             use_onnx=True)
                    else:
                        results_2, prob_df2 = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                             model = model_dict_2['model'], 
                                                             device = model_dict_2['device'], 
                                                             label_list = model_dict_2['label_list'], 
                                                             max_seq_length = model_dict_2['max_seq_length'] ,
                                                             tokenizer = model_dict_2['tokenizer'], 
                                                             output_mode = model_dict_2['output_mode'], 
                                                             model_type = model_dict_2['model_type'], 
                                                             use_onnx=False)
                    
                    #####################profiling##############################

                    bert_model_met_m2 = time.time()-algo_seventh_step
                    self.time_dict['bert_model_met_m2']=bert_model_met_m2
                    #####################profiling##############################


                    if len(results_1)==len(sentence_aspect_label_tuple_list) & len(results_2)==len(sentence_aspect_label_tuple_list):
                        count = -1
                        for item in self.dict_of_sentences['metstatus']:
                            count = count + 1
                            prediction = ""
                            negation_status = get_negation_status(item[3],item[0])

                            confidence_dict1 = dict(prob_df1.iloc[count])
                            confidence_dict2 = dict(prob_df2.iloc[count])
                            if negation_status:
                                prediction = "no"
                            else:
                                if get_micro_macromet_status(item[3]):
                                    prediction = "local"
                                elif np.any([x in item[3].lower() for x in self.suspicious_list_for_mets]):
                                    prediction = "indeterminate"
                                        
                                else:
                                    pred_1 = results_1[count]
                                    pred_2 = results_2[count]

                                    if pred_1 == 'positive/local':
                                        prediction = pred_2

                                    else:
                                        prediction = pred_1
                            tup = item + (prediction, confidence_dict1, confidence_dict2)
                            List.append(tup)                    
            
        if len(List)>0:        
            outp = self._convert_to_dict('metstatus',List)
            if outp is not None:
                self.outDict['metstatus'] = outp


    def get_pred_grade(self):
        if not 'oncologyv2' in self.outDict:
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list(
            'grade')

        if sentence_aspect_label_tuple_list:
            outDict_values = []
            nerc_results, _ = self.get_bert_classification_inference('grade_nerc', 'grade', sentence_aspect_label_tuple_list)
            assert len(nerc_results) == len(ner_out_dict_values), "Mismatch between input length and output length of grade NERC model"
            if len(nerc_results) == len(ner_out_dict_values):
                count = 0
                for item in ner_out_dict_values:
                    item['prediction'] = nerc_results[count]
                    item['attribute']['is_irrelevant'] = (nerc_results[count].lower().strip() == 'irrelevant')
                    outDict_values.append(item)
                    count += 1
            if len(outDict_values)>0:
                self.outDict['grade'] = outDict_values


    def get_pred_race(self):
        if 'race' in self.dict_of_sentences.keys():
            List = []
            for item in self.dict_of_sentences['race']:
                if 'egfr' in item[3].lower():  ## to deal with cases like  eGFR Non-African-American 88 mL/min where african amertican is a lab test reference, not race of patient
                    continue
                tup = item 
                List.append(tup)
            if len(List)>0:
                outp = self._convert_to_dict('race',List)
                if outp is not None:
                    self.outDict['race'] = outp


    def get_pred_gender(self):
        list_of_male_gender_words = ['male','boy','man','gentleman','mr\.','he','his','him']
        list_of_female_gender_words = ['female','girl','woman','lady','miss','mrs','ms\.','she','her']

        
        if 'gender' in self.dict_of_sentences.keys():
            List = []
            for item in self.dict_of_sentences['gender']:
                base_prediction = item[0].lower().strip()
                if base_prediction in list_of_male_gender_words:
                    prediction = "male"
                elif base_prediction in list_of_female_gender_words:
                    prediction = "female"
                else:
                    continue
                tup = item + (prediction,)
                List.append(tup)
            if len(List)>0:
                outp = self._convert_to_dict('gender',List)
                if outp is not None:
                    self.outDict['gender'] = outp


    def get_pred_menopause(self):
        
        if 'menopause' in self.dict_of_sentences.keys():
            # classes = ['premenopausal', 'postmenopausal', 'indeterminate', 'irrelevant']
            
            outDict_values = []
            sentence_aspect_label_tuple_list = [(item[3].lower(),None,"") for item in  self.dict_of_sentences['menopause']]
            if 'menopause_cls' in self.BERT_models.keys():
                model_dict = self.BERT_models['menopause_cls']
                if not pandas.isnull(model_dict):
                    
                    #####################profiling##############################
                    algo_step = time.time()
                    #####################profiling##############################
                    if ('menopause' in self.use_onnx) and model_dict.get('use_onnx', False):
                        results, prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                          model = model_dict['model'], 
                                                          device = model_dict['device'],
                                                          label_list = model_dict['label_list'], 
                                                          max_seq_length = model_dict['max_seq_length'] ,
                                                          tokenizer = model_dict['tokenizer'], 
                                                          output_mode = model_dict['output_mode'], 
                                                          model_type = model_dict['model_type'], 
                                                          use_onnx=True)
                    else:
                        results, prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                          model = model_dict['model'], 
                                                          device = model_dict['device'], 
                                                          label_list = model_dict['label_list'], 
                                                          max_seq_length = model_dict['max_seq_length'] ,
                                                          tokenizer = model_dict['tokenizer'], 
                                                          output_mode = model_dict['output_mode'], 
                                                          model_type = model_dict['model_type'], 
                                                          use_onnx=False)
                            
                    #####################profiling##############################
                    bert_model_menopause = time.time() - algo_step
                    self.time_dict['bert_model_menopause'] = bert_model_menopause
                    #####################profiling##############################


                    if len(results) == len(self.dict_of_sentences['menopause']):
                        count = 0
                        for item in self.dict_of_sentences['menopause']:
                            Dict = {}
                            prediction = results[count]
                            Dict['prediction'] = prediction
                            Dict['base_prediction'] = item[0]
                            Dict['start_char'] = item[1]
                            Dict['end_char'] = item[2]
                            Dict['context'] = item[3]
                            Dict['context_start_char'] = item[4]
                            Dict['context_end_char'] = item[5]
                            Dict['confidence'] = dict(prob_df.iloc[count]).get(prediction)
                            Dict['attribute'] = {}
                            
                            Dict = update_menopause_rule_based_prediction(Dict, self.menopause_surgery_list)
                            Dict['attribute']['irrelevance'] = update_menopause_rule_based_irrelevance(Dict)
                            Dict['attribute']['negation'] = 'no'
                            Dict['attribute']['subject'] = 'self'

                            count = count + 1
                            outDict_values.append(Dict)

            if len(outDict_values) > 0:
                self.outDict['menopause'] = outDict_values


    def get_pred_smoking(self):
        if 'smoking' in self.dict_of_sentences.keys():
            sentence_aspect_label_tuple_list = [(item[3].lower(), None, "") for item in self.dict_of_sentences['smoking']]
            outDict_values = []
            if 'smoking' in self.BERT_models.keys():
                model_dict = self.BERT_models['smoking']
                if not pandas.isnull(model_dict):

                    #####################profiling##############################
                    algo_eigth_step = time.time()
                    #####################profiling##############################
                    use_onnx = ('smoking' in self.use_onnx) and model_dict.get('use_onnx', False)
                    results, prob_df = bert_inference(
                        sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                        model=model_dict['model'],
                        device=model_dict['device'],
                        label_list=model_dict['label_list'],
                        max_seq_length=model_dict['max_seq_length'],
                        tokenizer=model_dict['tokenizer'],
                        output_mode=model_dict['output_mode'],
                        model_type=model_dict['model_type'],
                        use_onnx=use_onnx)

                    #####################profiling##############################
                    bert_model_smoking = time.time() - algo_eigth_step
                    self.time_dict['bert_model_smoking'] = bert_model_smoking

                    #####################profiling##############################

                    if len(results) == len(self.dict_of_sentences['smoking']):
                        count = 0
                        for item in self.dict_of_sentences['smoking']:
                            Dict = {}
                            prediction = results[count]
                            Dict['prediction'] = prediction
                            Dict['base_prediction'] = item[0]
                            Dict['start_char'] = item[1]
                            Dict['end_char'] = item[2]
                            Dict['context'] = item[3]
                            Dict['context_start_char'] = item[4]
                            Dict['context_end_char'] = item[5]
                            Dict['confidence'] = dict(prob_df.iloc[count]).get(prediction)
                            Dict['attribute'] = {}
                            Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                                   Dict['context_start_char'],
                                                                                   Dict['context_end_char'],
                                                                                   Dict['start_char'], Dict['end_char'])

                            count = count + 1
                            outDict_values.append(Dict)

            if len(outDict_values) > 0:
                self.outDict['smoking'] = outDict_values

    
    def get_oncologyv2_dependent_sentence_aspect_label_tuple_list(self, field):
        sentence_aspect_list = []
        field_out_dict_values = []
        if 'oncologyv2' not in self.outDict:
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'surgery':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'cancer_surgery':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'performance_status':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'performance_status':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'stage':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'staging':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'grade':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'grade':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'hist':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'histological_type':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        if field.lower() == 'death':
            for item in self.outDict['oncologyv2']:
                if item['prediction'].lower() == 'mortality_status':
                    _tuple = (item['context'], item['base_prediction'], "")
                    sentence_aspect_list.append(_tuple)
                    field_out_dict_values.append(item)
            return sentence_aspect_list, field_out_dict_values

        return sentence_aspect_list, field_out_dict_values


    def get_pred_performance_status(self):
        if 'oncologyv2' not in self.outDict:
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list('performance_status')

        if sentence_aspect_label_tuple_list:
            outDict_values = []
            if 'performance_status_nerc' in self.BERT_models.keys():
                model_dict = self.BERT_models['performance_status_nerc']
                if not pandas.isnull(model_dict):
                    #####################profiling##############################
                    algo_fifth_step = time.time()
                    #####################profiling##############################
                    if ('performance_status' in self.use_onnx) and model_dict.get('use_onnx', False):
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'], device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=True)
                    else:
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'], device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=False)

                    #####################profiling##############################
                    bert_model_performance_status_nerc = time.time() - algo_fifth_step
                    self.time_dict['bert_model_performance_status_nerc'] = bert_model_performance_status_nerc
                    #####################profiling##############################

                    if len(results) == len(ner_out_dict_values):
                        count = 0
                        for item in ner_out_dict_values:
                            item['prediction'] = results[count]
                            outDict_values.append(item)
                            count += 1
            if len(outDict_values) > 0:
                self.outDict['performance_status'] = outDict_values

    def get_pred_death(self):
        if not 'oncologyv2' in self.outDict:
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list(
            'death')
        if len(ner_out_dict_values) > 0:
            self.outDict['death'] = ner_out_dict_values

    
    def get_pred_bertneroncologyv2(self):
        if "oncologyv2" not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['oncologyv2']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('oncologyv2' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx,
            recalibration_dt_model_dict_list=[self.decision_tree_models.get(key) for key in self.decision_tree_models.keys() if key.startswith('oncologyv2')])

        #####################profiling##############################
        bert_model_neroncology = time.time() - algo_eigth_step
        self.time_dict['oncologyv2'] = bert_model_neroncology
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "oncologyv2 - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['prediction'] = ner_chunk_span[3]
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                        Dict['context_start_char'],
                                                                        Dict['context_end_char'],
                                                                        Dict['start_char'], Dict['end_char'])
                    
                    Dict['attribute']['is_irrelevant'] = False
                    if ner_chunk_span[5]==1:
                        Dict['attribute']['recalibration_prediction'] = 'relevant'
                    elif ner_chunk_span[5]==0:
                        Dict['attribute']['recalibration_prediction'] = 'irrelevant'
                    elif ner_chunk_span[5]==-1:
                        Dict['attribute']['recalibration_prediction'] = ''
                    
                    outDict_values.append(Dict)

        if len(outDict_values) > 0:
            self.outDict['oncologyv2'] = outDict_values


    def pred_nerc_tag(self, outDict_values, field, tag, model_name, relevancy=False, aspect_encloser=("", "")):
        if model_name not in self.BERT_models.keys():
            return outDict_values
        model_dict = self.BERT_models[model_name]
        
        sentence_aspect_label_tuple_list = []
        ner_out_dict_values = []
        indices = []
        sub_sent_spans = self.get_sub_sent_spans(model_dict['tokenizer'], model_dict['max_seq_length'])
        for idx, item in enumerate(outDict_values):
            if item['attribute']['tag'].lower() == tag:
                _tuple = self.get_central_tokenized_sentence_tuple(model_dict['tokenizer'], sub_sent_spans, item['start_char'], item['end_char'], model_dict['max_seq_length'], aspect_encloser=aspect_encloser)
                
                if len(model_dict['tokenizer'].tokenize(_tuple[0])) + len(model_dict['tokenizer'].tokenize(_tuple[1]))>model_dict['max_seq_length']:
                    print("might be wrong, check")
                if _tuple[1] not in _tuple[0]:
                    print(_tuple, item['base_prediction'])

                sentence_aspect_label_tuple_list.append(_tuple)
                ner_out_dict_values.append(item)
                indices.append(idx)

        if sentence_aspect_label_tuple_list:
            nerc_results, _ = self.get_bert_classification_inference(model_name, field, sentence_aspect_label_tuple_list)
            assert len(nerc_results) == len(ner_out_dict_values), f"Mismatch between input length and output length of {model_name} model"
            if len(nerc_results) == len(ner_out_dict_values):
                count = 0
                for i, item in enumerate(ner_out_dict_values):
                    item['attribute']['is_irrelevant'] = (nerc_results[count].lower().strip() == "irrelevant")
                    if not relevancy:
                        item['prediction'] = nerc_results[count]
                    outDict_values[indices[i]] = item
                    count += 1

        return outDict_values

    def get_pred_race_ethnicity(self):
        if ("race_ethnicity_ner" not in self.BERT_models.keys()) or ("race_nerc" not in self.BERT_models.keys()) or ("ethnicity_nerc" not in self.BERT_models.keys()):
            return

        model_dict = self.BERT_models['race_ethnicity_ner']
        if pandas.isnull(model_dict):
            return

        # chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
        #                                     , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0.0)
        chunk_spans = self.natural_chunk_dict.get(model_dict['max_seq_length'], None)
        if not chunk_spans:
            return
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('race_ethnicity' in self.use_onnx) and model_dict.get('use_onnx', False)  # field name
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        #####################profiling##############################
        bert_model_time = time.time() - algo_eigth_step
        self.time_dict['race_ethnicity_ner'] = bert_model_time
        #####################profiling##############################

        race_outdict_values = []
        ethnicity_outdict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "race_ethnicity_ner - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            sub_sent_spans = self.get_sub_sent_spans(model_dict['tokenizer'], model_dict['max_seq_length'])
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['prediction'] = ner_chunk_span[3]
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                        Dict['context_start_char'],
                                                                        Dict['context_end_char'],
                                                                        Dict['start_char'], Dict['end_char'])
                    if Dict['prediction'].lower() == 'race':
                        race_outdict_values.append(Dict)
                    elif Dict['prediction'].lower() == 'ethnicity':
                        ethnicity_outdict_values.append(Dict)

        # handle the nerc
        # race_sentence_aspect_label_tuple_list = [(item['context'], item['base_prediction'], "") for item in race_outdict_values]
        # ethnicity_sentence_aspect_label_tuple_list = [(item['context'], item['base_prediction'], "") for item in ethnicity_outdict_values]
        race_nerc_outdict_values = []
        ethnicity_nerc_outdict_values = []
        
        # def get_nerc_outputs(sentence_aspect_label_tuple_list, outDict_values, model_name):
        #     outdict_values = []
        #     if sentence_aspect_label_tuple_list:
        #         nerc_results, _ = self.get_bert_classification_inference(model_name, 'race_ethnicity', sentence_aspect_label_tuple_list)
        #         assert len(nerc_results) == len(outDict_values), f"{model_name} - input and output length mismatch to nerc model"
        #         if len(nerc_results) == len(outDict_values):
        #             for label, item in zip(nerc_results, outDict_values):
        #                 item['prediction'] = label
        #                 outdict_values.append(item)
                        
        #     return outdict_values
        
        race_nerc_outdict_values = self.pred_nerc_tag(race_outdict_values, 'race_ethnicity', 'race', 'race_nerc', relevancy=False, aspect_encloser = ('<entity>', '<entity>'))
        ethnicity_nerc_outdict_values = self.pred_nerc_tag(ethnicity_outdict_values, 'race_ethnicity', 'ethnicity', 'ethnicity_nerc', relevancy=False, aspect_encloser = ('<entity>', '<entity>'))
        if len(race_nerc_outdict_values) > 0:
            self.outDict['race'] = race_nerc_outdict_values
        if len(ethnicity_nerc_outdict_values) > 0:
            self.outDict['ethnicity'] = ethnicity_nerc_outdict_values
            

    def get_pred_drugs(self):
        if "medication_ner" not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['medication_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('medication' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx,
            recalibration_dt_model_dict_list=[self.decision_tree_models.get('medication_ner')])

        #####################profiling##############################
        bert_model_nermedication = time.time() - algo_eigth_step
        self.time_dict['medication_ner'] = bert_model_nermedication
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "medication_ner - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['prediction'] = ner_chunk_span[0]
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['tag'] = ner_chunk_span[3]
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                        Dict['context_start_char'],
                                                                        Dict['context_end_char'],
                                                                        Dict['start_char'], Dict['end_char'])
                    
                    Dict['attribute']['is_irrelevant'] = False
                    if ner_chunk_span[5]==1:
                        Dict['attribute']['recalibration_prediction'] = 'relevant'
                    elif ner_chunk_span[5]==0:
                        Dict['attribute']['recalibration_prediction'] = 'irrelevant'
                    elif ner_chunk_span[5]==-1:
                        Dict['attribute']['recalibration_prediction'] = ''
                    outDict_values.append(Dict)
        
        nerc_fields = {"route of administration": "route_nerc", "therapy ongoing?": "therapy_ongoing_nerc", "drug name": "drug_nerc"}
        for tag, model_name in nerc_fields.items():
            if tag=="drug name":
                outDict_values = self.pred_nerc_tag(outDict_values, "medication", tag, model_name, relevancy=True)
            else:
                outDict_values = self.pred_nerc_tag(outDict_values, "medication", tag, model_name)

        if len(outDict_values) > 0:
            self.outDict['medication'] = outDict_values


    def get_pred_treatmentresponse(self):
        if 'treatmentresponse_ner' not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['treatmentresponse_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('treatmentresponse' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        #####################profiling##############################
        bert_model_treatmentresponse = time.time() - algo_eigth_step
        self.time_dict['bert_model_treatmentresponse_ner'] = bert_model_treatmentresponse
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "treatment response - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['prediction'] = ner_chunk_span[3]
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                        Dict['context_start_char'],
                                                                        Dict['context_end_char'],
                                                                        Dict['start_char'], Dict['end_char'])
                    # turn off the heuristic irrelevance till further notice
                    # field_start_char = ner_chunk_span[1]
                    # field_end_char = ner_chunk_span[2]
                    # Dict['attribute']['is_irrelevant'] = self._is_treatmentresponse_irrelevant(text=Dict['context'].lower(),
                    #                                                                         field_start_end_char=(field_start_char, field_end_char))
                    Dict['attribute']['is_irrelevant'] = False
                    outDict_values.append(Dict)

        if len(outDict_values) > 0:
            self.outDict['treatmentresponse'] = outDict_values


    def get_pred_surgery(self):
        if not 'oncologyv2' in self.outDict:
            return
        
        if not 'surgery_irr' in self.BERT_models.keys():
            return

        sentence_aspect_label_tuple_list, ner_out_dict_values = self.get_oncologyv2_dependent_sentence_aspect_label_tuple_list('surgery')

        surgery_irr_results, _ = self.get_bert_classification_inference('surgery_irr', 'surgery', sentence_aspect_label_tuple_list)

        assert len(surgery_irr_results) == len(ner_out_dict_values), "Surgery Irr and NER prediction length mismatch"

        asserted_ner_outdict_values = []

        for i in range(len(ner_out_dict_values)):
            item = ner_out_dict_values[i]
            item['attribute']['is_irrelevant'] = (surgery_irr_results[i].lower().strip() == "irrelevant")
            asserted_ner_outdict_values.append(item)

        if len(ner_out_dict_values)>0:
            self.outDict['surgery'] = asserted_ner_outdict_values


    def get_pred_radiation(self):
        if 'radiation_ner' not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['radiation_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('radiation' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx,
            # recalibration_dt_model_dict_list=[self.decision_tree_models.get('radiation_ner')]
        )

        #####################profiling##############################
        bert_model_radiation = time.time() - algo_eigth_step
        self.time_dict['bert_model_radiation_ner'] = bert_model_radiation
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "radiation - input and output length mismatch to ner model"
        if len(all_ner_chunk_spans) == len(chunk_spans):
            for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                    # sent_span - chunk, start_char, end_char, n_tokens
                    Dict = {}
                    Dict['prediction'] = ner_chunk_span[3]
                    Dict['base_prediction'] = ner_chunk_span[0]
                    Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                    Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                    Dict['context'] = sent_span[0]
                    Dict['context_start_char'] = sent_span[1]
                    Dict['context_end_char'] = sent_span[2]
                    Dict['confidence'] = ner_chunk_span[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['tag'] = ner_chunk_span[3]
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                        Dict['context_start_char'],
                                                                        Dict['context_end_char'],
                                                                        Dict['start_char'], Dict['end_char'])
                    Dict['attribute']['is_irrelevant'] = False
                    if ner_chunk_span[5]==1:
                        Dict['attribute']['recalibration_prediction'] = 'relevant'
                    elif ner_chunk_span[5]==0:
                        Dict['attribute']['recalibration_prediction'] = 'irrelevant'
                    elif ner_chunk_span[5]==-1:
                        Dict['attribute']['recalibration_prediction'] = ''
                    outDict_values.append(Dict)

        outDict_values = self.pred_nerc_tag(outDict_values, "radiation", "modality", "radiation_modality_irr", relevancy=True)

        if len(outDict_values) > 0:
            self.outDict['radiation'] = outDict_values


    def get_pred_biomarker_and_attributes(self):
        
        def get_status1_in_context(context,biomarker):
            
            matches1 = list(chain_iter(self.biomarker_status1_regex_dict,context))
            matches_posneg = []
            if biomarker in self.biomarker_status_posneg_regex.keys():
                matches_posneg = list(chain_iter(self.biomarker_status_posneg_regex[biomarker],context))
            if len(matches1)>0 or len(matches_posneg)>0:
                return True
            else:
                return False
            
        
        def get_status2_pred(context, biomarker):
            output = None
            if biomarker in self.biomarker_status2_regex_dict.keys():
                if biomarker in self.crf_status2_biomarker_class:
                    if 'biomarkers' in self.crf_models.keys() and biomarker in ['pd1','pdl1','tmb']:
                            model_dict = self.crf_models['biomarkers']
                            if not pandas.isnull(model_dict):
                                base_result, normalized_result, prob_score = nlp_pipeline_format(context, biomarker, model_dict['model'])
                                if normalized_result:
                                    output = {'prediction':normalized_result, 'base_prediction':base_result, 'confidence':prob_score}
                    
                    elif 'biomarkers_bc' in self.crf_models.keys() and biomarker in ['er','pr','her2']:
                            model_dict = self.crf_models['biomarkers_bc']
                            if not pandas.isnull(model_dict):
                                base_result, normalized_result, prob_score = nlp_pipeline_format(context, biomarker, model_dict['model'])
                                if normalized_result:
                                    output = {'prediction':normalized_result, 'base_prediction':base_result, 'confidence':prob_score}
            
                else:
                    regex_list = self.biomarker_status2_regex_dict[biomarker]
                    matches = chain_iter(regex_list,context)
                    matches = list(matches)
                    if len(matches)==1:
                            for match in matches:
                                keys = list(match.groupdict().keys())
                                Key = [x for x in keys if 'status2' in x][0]
                                prediction = match.groupdict()[Key]
                                output = {'prediction' : prediction, 'base_prediction' : prediction, 'confidence' : ""}
            return output
        
        def get_mv_pred(context, biomarker,status1_value):
            output = []
            
            if biomarker in self.unamb_mv_regex_dict.keys():
                mv_matches = list(chain_iter(self.unamb_mv_regex_dict[biomarker],context))
                if len(mv_matches) > 0:
                    for match in mv_matches:
                        for key in match.groupdict():
                            field = key.split("_")[0].lower()
                            start_char = match.span(key)[0]
                            end_char = match.span(key)[1]
                            prediction = match.groupdict()[key]
                            prediction = prediction.strip().lower()
                            if prediction:
                                output.append({'prediction':prediction, 'base_prediction':prediction,'confidence':"",'status':status1_value})
                
            if biomarker in self.amb_mv_regex_dict.keys() and not output:
                mv_matches = list(chain_iter(self.amb_mv_regex_dict[biomarker],context))
                if len(mv_matches) > 0:
                    for match in mv_matches:
                        for key in match.groupdict():
                            field = key.split("_")[0].lower()
                            start_char = match.span(key)[0]
                            end_char = match.span(key)[1]
                            prediction = match.groupdict()[key]
                            prediction = prediction.strip().lower()
                            if prediction:
                                output.append({'prediction':prediction, 'base_prediction':prediction,'confidence':"",'status':status1_value})
                                
            if output:
                output = list({v['prediction']:v for v in output}.values())
                if len(output)==1:
                    return output
                else:
                    return None
            else:
                return None

        def preprocess_biomarker_sentences(dict_of_sentences):
            if not dict_of_sentences:
                return dict_of_sentences

            # remove ros due to high false positives because of review of systems
            synonyms_to_exclude = ['ros']
            dict_of_sentences = [item for item in dict_of_sentences if item[0].lower().strip() not in synonyms_to_exclude]

            # remove met snippets
            sent_raw_dict = {}
            for item in dict_of_sentences:
                sent_raw_dict.setdefault(item[3].lower().strip(),[]).append(item[0].lower().strip())
            only_met_sents = [k for k,v in sent_raw_dict.items() if (len(set(v))==1) and ('met' in v)]
            return [item for item in dict_of_sentences
                    if not ((item[0].lower().strip()=='met') and (item[3].lower().strip() in only_met_sents))]


        if 'biomarkers' in self.dict_of_sentences.keys():
            new_bm_dict_of_sentences = preprocess_biomarker_sentences(self.dict_of_sentences['biomarkers'])
            if not new_bm_dict_of_sentences:
                del self.dict_of_sentences['biomarkers']
                return

            self.dict_of_sentences['biomarkers'] = new_bm_dict_of_sentences

            sentence_aspect_label_tuple_list = [(item[3].lower(),item[0].lower(),"") for item in  self.dict_of_sentences['biomarkers']]
            if 'biomarkers' in self.BERT_models.keys():
                model_dict = self.BERT_models['biomarkers']
                ner_model_dict = self.BERT_models['bio-ner']
                re_model_dict = self.dl_models['biomarkers']
                
                if model_dict and ner_model_dict and re_model_dict:
                    ner_model, ner_tok, label_list = ner_model_dict['model'], ner_model_dict['tokenizer'], ner_model_dict['label_list']


                    all_ner_res = []
                    sent_batch = [sent_pack[0].lower() for sent_pack in sentence_aspect_label_tuple_list]
                    if ('biomarkers' in self.use_onnx) and ner_model_dict.get('use_onnx', False):
                        all_ner_res = ner_inference(text=sent_batch, model=ner_model, tok=ner_tok, label_list=label_list, use_onnx=True)
                    else:
                        all_ner_res = ner_inference(text=sent_batch, model=ner_model, tok=ner_tok, label_list=label_list, use_onnx=False)

                    for ner_res in all_ner_res:
                        for i, token_tag_dict in enumerate(ner_res):
                            token = list(token_tag_dict.keys())[0]
                            if token not in ['.','+','-']:
                                continue
                            elif token in ['+','-']:
                                if (i) and (i != len(ner_res)-1):
                                    prev_dict = ner_res[i-1]
                                    next_dict = ner_res[i+1]
                                    prev_tok = list(prev_dict.keys())[0]
                                    next_tok = list(next_dict.keys())[0]
                                    if (prev_tok in ['(','[','{']) and (next_tok in [')',']','}']):
                                        ner_res[i][token] = 'CATEGORICAL'
                            else:
                                if (i) and (i != len(ner_res)-1):
                                    prev_dict = ner_res[i-1]
                                    next_dict = ner_res[i+1]
                                    prev_tag = list(prev_dict.values())[0]
                                    next_tag = list(next_dict.values())[0]
                                    if (prev_tag == 'NUMERICAL') and (next_tag == 'NUMERICAL'):
                                        ner_res[i][token] = 'NUMERICAL'

                    ner_dict = {}
                    gene_dict_of_sentences = []
                    for sent_pack, ner_res in zip(sentence_aspect_label_tuple_list, all_ner_res):

                        if sent_pack[0].lower() not in ner_dict.keys():
                            ner_dict[sent_pack[0].lower()] = ner_res

                            curr_item = [_item for _item in self.dict_of_sentences['biomarkers'] if _item[3].lower() == sent_pack[0].lower()][0]
                            local_sent_text = curr_item[3]
                            prev_tag, prev_start, prev_end = '',0,0
                            
                            for token_tag_dict in ner_res:
                                token,tag = list(token_tag_dict.items())[0]
                                
                                start,end = re.search(re.escape(token), local_sent_text, re.IGNORECASE).span()
                                if (prev_tag == tag) and (tag == 'GENE'):
                                    prev_end += end
                                    local_sent_text = local_sent_text[end:]
                                else:
                                    if (prev_tag != '') and (prev_tag == 'GENE'):
                                        raw_pred = curr_item[3][prev_start:prev_end]
                                        norm_pred = normalize_bm(''.join(raw_pred.split()))
                                        if norm_pred:
                                            gene_dict_of_sentences.append((
                                                raw_pred, curr_item[4]+prev_start, curr_item[4]+prev_end
                                                ,curr_item[3], curr_item[4], curr_item[5], norm_pred
                                            ))
                                    prev_tag, prev_start, prev_end = tag, prev_end+start, prev_end+end
                                    local_sent_text = local_sent_text[end:]
                            if (prev_tag != '') and (prev_tag == 'GENE'):
                                raw_pred = curr_item[3][prev_start:prev_end]
                                norm_pred = normalize_bm(''.join(raw_pred.split()))
                                if norm_pred:
                                    gene_dict_of_sentences.append((
                                        raw_pred, curr_item[4]+prev_start, curr_item[4]+prev_end
                                        ,curr_item[3], curr_item[4], curr_item[5], norm_pred
                                            ))

                    # drop false positives from regex
                    self.dict_of_sentences['biomarkers'] = [item
                                                            for item in self.dict_of_sentences['biomarkers']
                                                            if item[0].strip().lower() not in
                                                            ['','er','pr','met','egfr','ros','alk']]
                    self.dict_of_sentences['biomarkers'] = list(set(self.dict_of_sentences['biomarkers'] + gene_dict_of_sentences))
                    if not self.dict_of_sentences['biomarkers']:
                        del self.dict_of_sentences['biomarkers']
                        return

                    sentence_aspect_label_tuple_list = [(item[3].lower(),item[0].lower(),"") for item in  self.dict_of_sentences['biomarkers']]

                    all_re_res = []
                    sent_map_re = []

                    start= time.time()

                    head_token_start_list, head_token_end_list = [], []
                    child_token_start_list, child_token_end_list = [], []
                    text_list = []
                    asp_list = []
                    all_token_list= []

                    bio_ent_dict = {}
                    val_ent_dict = {}

                    for sent_pack in sentence_aspect_label_tuple_list:

                        sent_res = ner_dict[sent_pack[0].lower()]
                        re_res_sent = []
                        bio_entities = []
                        value_entities = []

                        token_list = [] 

                        for j, token_tag_dict in enumerate(sent_res):
                            token = list(token_tag_dict.keys())[0]
                            token_list.append(token)
                            tag = list(token_tag_dict.values())[0]
                            if (tag == 'GENE') or (token.lower()==sent_pack[1].lower()):
                                bio_entities.append((token, j, tag))
                            elif tag in ['CATEGORICAL', 'NUMERICAL', 'NUMERICAL_UNIT', 'VARIANT', 'GENE_RESULT', 'EXON_VALUE']:
                                value_entities.append((token, j, tag))

                        skip_bio_token = []
                        skip_value_token = []

                        if sent_pack[0].lower() not in bio_ent_dict.keys():

                            bio_ent_dict[sent_pack[0].lower()] = bio_entities
                            val_ent_dict[sent_pack[0].lower()] = value_entities

                        for a, bio_ent in enumerate(bio_entities):

                            if a in skip_bio_token:
                                continue

                            if (bio_ent[0].lower()!=sent_pack[1].lower()) and (bio_ent[0].lower() not in sent_pack[1].lower()):
                                continue

                            head_token_start = bio_ent[1]
                            head_token_end = bio_ent[1]
                            head_token_ner = bio_ent[0]
                            head_token_end_ner = bio_ent[0]


                            for k in range(a+1, len(bio_entities)):

                                if bio_entities[k][1] == (head_token_end + 1):
                                    head_token_end= bio_entities[k][1]
                                    head_token_end_ner = bio_entities[k][0]
                                    skip_bio_token.append(k)
                                else:
                                    break



                            for b, val_ent in enumerate(value_entities):

                                if b in skip_value_token:
                                    continue

                                child_token_start = val_ent[1]
                                child_token_end = val_ent[1]
                                child_token_ner = val_ent[0]
                                child_token_end_ner = val_ent[0]

                                for k in range(b+1, len(value_entities)):

                                    if value_entities[k][1] == (child_token_end + 1):
                                        child_token_end= value_entities[k][1]
                                        child_token_end_ner = value_entities[k][0]
                                        skip_value_token.append(k)
                                    else:
                                        break

                                head_token_start_list.append(head_token_start)
                                head_token_end_list.append(head_token_end)
                                child_token_start_list.append(child_token_start)
                                child_token_end_list.append(child_token_end)
                                text_list.append(sent_pack[0].lower())
                                asp_list.append(sent_pack[1].lower())
                                all_token_list.append(token_list)


                    if len(head_token_start_list):

                        re_res, h_list, c_list = predict(model = re_model_dict['model'], sentence=text_list,
                                    word_vocab=re_model_dict['word_vocab'], abs_vocab= re_model_dict['abs_vocab'],
                                    rel_vocab = re_model_dict['rel_vocab'],
                                    max_sentence_length = re_model_dict['parameters']['max_sentence_length'],
                                    head_token_start_list = head_token_start_list,
                                    head_token_end_list=head_token_end_list,
                                    child_token_start_list=child_token_start_list,
                                    child_token_end_list = child_token_end_list,
                                    token_list = all_token_list)

                        results_df = pandas.DataFrame()
                        results_df['sent'] = text_list
                        results_df['aspect'] = asp_list

                        bio_ent_list = [bio_ent_dict[i.lower()] for i in text_list]
                        val_ent_list = [val_ent_dict[i.lower()] for i in text_list]

                        results_df['bio_ent'] = bio_ent_list
                        results_df['val_ent'] = val_ent_list

                        results_df['head'] = h_list
                        results_df['child'] = c_list

                        results_df['re_res'] = re_res


                        results = results_df[results_df['re_res'] != 0]

                        sent_int = results['sent'].drop_duplicates(keep='first').tolist()

                        for s in sent_int:

                            all_asp = list(set(results[results['sent'] == s]['aspect'].to_list()))

                            for asp in all_asp:

                                re_res_sent = []

                                asp_df = results[results['sent'] == s]
                                asp_df = asp_df[asp_df['aspect'] == asp]

                                for kk in range(len(asp_df)):
                                    h1 = asp_df.iloc[kk]['head']
                                    #h1 = ' '.join(h1)
                                    c1 = asp_df.iloc[kk]['child']
                                    #c1 = ' '.join(c1)
                                    tags_for_pred = []

                                    value_entities = asp_df.iloc[kk]['val_ent']

                                    for relp in c1:
                                        for myvalent in value_entities:
                                            if myvalent[0].lower()== relp.lower():
                                                tags_for_pred.append(myvalent[2])
                                                break

                                    values = (' '.join(h1), ' '.join(c1), ';'.join(tags_for_pred))
                                    if values not in re_res_sent:
                                        re_res_sent.append(values)

                                if len(re_res_sent) and (re_res_sent not in all_re_res):
                                    all_re_res.append(re_res_sent)
                                    sent_map_re.append(s.lower())

                    if not pandas.isnull(model_dict):

                        #####################profiling##############################
                        algo_fifth_step = time.time()
                        #####################profiling##############################
                        if ('biomarkers' in self.use_onnx) and model_dict.get('use_onnx', False):
                            results, prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                                model = model_dict['model'], 
                                                                device = model_dict['device'], 
                                                                label_list = model_dict['label_list'], 
                                                                max_seq_length = model_dict['max_seq_length'] ,
                                                                tokenizer = model_dict['tokenizer'], 
                                                                output_mode = model_dict['output_mode'], 
                                                                model_type = model_dict['model_type'], 
                                                                use_onnx=True)
                        else:
                            results, prob_df = bert_inference(sentence_aspect_label_tuple_list = sentence_aspect_label_tuple_list, 
                                                              model = model_dict['model'], 
                                                              device = model_dict['device'], 
                                                              label_list = model_dict['label_list'],
                                                              max_seq_length = model_dict['max_seq_length'] ,
                                                              tokenizer = model_dict['tokenizer'], 
                                                              output_mode = model_dict['output_mode'], 
                                                              model_type = model_dict['model_type'], 
                                                              use_onnx=False)

                        #####################profiling##############################
                        bert_model_biomarkers = time.time()-algo_fifth_step
                        self.time_dict['bert_model_biomarkers'] = bert_model_biomarkers
                        #####################profiling##############################


                        if len(results)==len(sentence_aspect_label_tuple_list):
                            outDict = {}
                            count = 0
                            for index, sent_asp_pair in enumerate(sentence_aspect_label_tuple_list):
                                output_status2 = None
                                output_mv = None
                                output_gr = None
                                status1 = None
                                done_flag = 0

                                normalized_bio_name = ''.join(sent_asp_pair[1].lower().split())
                                norm_new = normalize_bm(normalized_bio_name)
                                if not norm_new:
                                    continue

                                item = [_item 
                                        for _item in self.dict_of_sentences['biomarkers']
                                        if (_item[3].lower(),_item[0].lower(),"") == sent_asp_pair
                                    ][0]

                                # if normalized_bio_name in outDict.keys():
                                #     dict_list = outDict[normalized_bio_name]
                                if norm_new in outDict.keys():
                                    dict_list = outDict[norm_new]
                                    for dc in dict_list:
                                        if dc['context'].lower() == sent_asp_pair[0].lower():
                                            done_flag = 1
                                            break
                                if done_flag:
                                    continue

                                # if normalized_bio_name not in outDict.keys():
                                #     outDict[normalized_bio_name] = []
                                if norm_new not in outDict.keys():
                                    outDict[norm_new] = []


                                Dict = {}
                                # Dict['prediction'] = normalized_bio_name
                                Dict['prediction'] = norm_new
                                Dict['start_char'] = item[1]
                                Dict['end_char'] = item[2]
                                Dict['context'] = item[3]
                                Dict['context_start_char'] = item[4]
                                Dict['context_end_char'] = item[5]
                                Dict['confidence'] = ""
                                Dict['attribute'] = {}



                                rel_list = []
                                ner_list = []

                                bio_list = return_bio_check(normalized_bio_name)
                                for res_pack, sent_pack in zip(all_re_res, sent_map_re):
                                    for k in range(len(res_pack)):
                                        if (''.join(res_pack[k][0].lower().strip(' ').split(' ')) in bio_list) and (sent_pack.lower()==item[3].lower()):
                                            if (res_pack[k] not in rel_list):
                                                rel_list.append(res_pack[k])



                                for ind, sent_pack in enumerate(sentence_aspect_label_tuple_list):
                                    if sent_pack[0].lower() == item[3].lower():
                                        ner_list.append(tuple(ner_dict[sent_pack[0].lower()]))
                                        break

                                Dict['attribute']['relationships'] = list(rel_list)
                                Dict['attribute']['named-entities'] = list(ner_list)[0]
                                Dict['attribute']['subject'] = self._returnSubject(Dict['context'],Dict['context_start_char'],Dict['context_end_char'],Dict['start_char'],Dict['end_char'])
                                Dict['attribute']['status'] = {}
                                confidence_dict = dict(prob_df.iloc[index])

                                candidates = rel_list
                                final_pred, final_pred_status1 = '', ''
                                final_status_2 = {'prediction':'', 'base_prediction': '','confidence': ''}
                                final_gene_result = ''
                                var_dict = {'protein_alteration': '', 'test_result': '', 'biomarker_variant_type': ''}
                                final_variant = []
                                use_flag = 0
                                numerical_unit = ''



                                for ans in candidates:
                                    ans = list(ans)
                                    ans[0]= ''.join(ans[0].strip(' ').split(' '))

                                    if len(list(set(ans[2].split(';')))) == 1:  #### only 1 unique label in set ####

                                        if len(ans[2].split(';')) == 1:  #### one label only

                                            if (ans[0].lower() in bio_list) and (ans[2] in ['CATEGORICAL']):
                                                final_pred = final_pred + ' ' + ans[1]
                                                final_pred = final_pred.strip(' ')
                                            elif (ans[0].lower() in bio_list) and (ans[2] in ['NUMERICAL', 'NUMERICAL_UNIT']):
                                                final_status_2['prediction'] = final_status_2['prediction'] + ' ' + ans[1]
                                                final_status_2['prediction'] = final_status_2['prediction'].strip(' ')
                                            elif (ans[0].lower() in bio_list) and (ans[2] in ['VARIANT']):
                                                #var_dict = {'prediction': '', 'status':'', 'gene_result': '' }
                                                var_dict['protein_alteration'] = var_dict['protein_alteration'] + ' ' + ans[1]
                                                var_dict['protein_alteration'] = var_dict['protein_alteration'].strip(' ')

                                                final_variant.append(var_dict)
                                            elif (ans[0].lower() in bio_list) and (ans[2] in ['GENE_RESULT']):
                                                var_dict['biomarker_variant_type'] = var_dict['biomarker_variant_type'] + ' ' + ans[1]
                                                var_dict['biomarker_variant_type'] = var_dict['biomarker_variant_type'].strip(' ')
                                                if var_dict['biomarker_variant_type'].endswith('ed'):
                                                    if var_dict['test_result'] == '':
                                                        var_dict['test_result'] = 'positive'
                                                final_variant.append(var_dict)

                                        else:                        
                                            label_split = ans[2].split(';')
                                            word_split = ans[1].split(';')

                                            if (ans[0].lower() in bio_list) and (label_split[0] in ['CATEGORICAL']):
                                                for a, b in zip(label_split, word_split):
                                                    final_pred = final_pred + ' ' + b.lower()


                                    else:
                                        if (ans[0].lower() in bio_list):
                                            label_split= ans[2].split(';')
                                            word_split = ans[1].split(' ')
                                            flag = 0


                                            for a, b in zip(label_split, word_split):
                                                if a in ['VARIANT']:
                                                    if var_dict['test_result'] or var_dict['protein_alteration'] or var_dict['biomarker_variant_type']:
                                                        final_variant.append(var_dict)
                                                    var_dict = {'protein_alteration': '', 'test_result': '', 'biomarker_variant_type': ''}
                                                    var_dict['protein_alteration'] = var_dict['protein_alteration'] + ' ' + b.lower()
                                                elif a in ['GENE_RESULT']:
                                                    var_dict['biomarker_variant_type'] = var_dict['biomarker_variant_type'] + ' ' + b.lower()

                                                elif a in ['CATEGORICAL']:
                                                    var_dict['test_result'] = var_dict['test_result'] + ' ' + b.lower()

                                                elif a in ['NUMERICAL']:

                                                    final_status_2['prediction'] = final_status_2['prediction'] + ' ' + b.lower()
                                                    final_status_2['prediction']=final_status_2['prediction'].strip(' ')

                                                elif a in ['NUMERICAL_UNIT']:

                                                    numerical_unit = numerical_unit + ' ' + b.lower()
                                                    numerical_unit = numerical_unit.strip(' ')

                                            if var_dict['biomarker_variant_type'].endswith('ed'):
                                                if var_dict['test_result'] == '':
                                                    var_dict['test_result'] = 'positive'

                                            final_variant.append(var_dict)

                                if get_status1_in_context(Dict['context'], normalized_bio_name):
                                    if len(re.findall("\\bif|african|trial|study|associated|reported\\b", str(item[3]).lower()))>0:  ## postprocessing , if if in sentence, status is irrelevant/egfr african american, contains any mention of trial / study(clinical trial)
                                        Dict['attribute']['status']['test_result'] = {'prediction' : 'irrelevant', 'base_prediction' : results[index], 'confidence' : confidence_dict} #### confidence_dict
                                        status1 = Dict['attribute']['status']['test_result']['prediction']

                                    else:

                                        if final_pred.strip(' ') != '':  ##### some re model pred is there ####
                                            final_pred_status1 = final_pred.strip(' ')
                                        else:
                                            #### no pred form re model ######

                                            if results[index] != 'irrelevant':    ##### aspect says not irrelevant 
                                                use_flag = 1
                                                for comp_pred in final_variant:
                                                    if comp_pred['test_result'].strip(' ') != '':
                                                        final_pred_status1 =  ''
                                                    else:
                                                        final_pred_status1 = 'irrelevant'
                                                        break

                                            else:
                                                #### aspect says irrelevant ######
                                                final_pred_status1 = results[index]

                                        if (new_normalizer_func(final_pred_status1) in  ['']) and (norm_new.lower().strip() in ['met']):
                                            final_pred_status1 = 'irrelevant'

                                        if new_normalizer_func(final_pred_status1) in  ['positive', '']:
                                            final_pred_status1 = results[index]
                                        
                                        final_pred_status1 = new_normalizer_func(final_pred_status1)




                                        if normalized_bio_name!='er' :
                                            Dict['attribute']['status']['test_result'] = {'prediction' : final_pred_status1 ,'base_prediction' : final_pred_status1, 'confidence' : confidence_dict}
                                            status1 = Dict['attribute']['status']['test_result']['prediction']
                                        else:
                                            if len(re.findall("\\bvisit\\b", str(item[3]).lower()))>0:  ## to remove ER visit, which is irrelevennt
                                                Dict['attribute']['status']['test_result'] = {'prediction' : 'irrelevant', 'base_prediction' : results[index], 'confidence' : confidence_dict}
                                                status1 = Dict['attribute']['status']['test_result']['prediction']
                                            else:


                                                Dict['attribute']['status']['test_result'] = {'prediction' : final_pred_status1, 'base_prediction' : final_pred_status1, 'confidence' : confidence_dict}
                                                status1 = Dict['attribute']['status']['test_result']['prediction']

                                if ('status1' not in Dict['attribute']['status'].keys()) or ('test_result' in Dict['attribute']['status'].keys() and Dict['attribute']['status']['test_result']['prediction'] != "irrelevant"):
                                    output_status2 = get_status2_pred(Dict['context'],Dict['prediction'])
                                    output_mv = get_mv_pred(Dict['context'], Dict['prediction'],status1)

                                if final_status_2['prediction'] != '':
                                    output_status2 = final_status_2

                                if (output_status2 is not None ): #and (final_pred_status1 != 'irrelevant'):

                                    output_status2['test_result_1_unit'] = numerical_unit
                                    output_status2['base_prediction'] = output_status2['prediction'] + ' ' + numerical_unit
                                    output_status2['base_prediction'] = output_status2['base_prediction'].strip(' ')
                                    Dict['attribute']['status']['test_result_1_numeric'] = output_status2


                                if not bool(Dict['attribute']['status']):
                                    Dict['attribute']['status'] = {'test_result':{'prediction':'irrelevant'}}
                                
                                
                                new_final_variant = []
                                
                                for int_dict in final_variant:
                                    if new_normalizer_func(int_dict['test_result']) in ['positive', '']:
                                        if 'test_result' in Dict['attribute']['status'].keys():
                                            if Dict['attribute']['status']['test_result']['prediction'] != 'irrelevant':
                                                int_dict['test_result'] = Dict['attribute']['status']['test_result']['prediction']
                                                new_final_variant.append(int_dict)
                                    else:
                                        int_dict['test_result'] = new_normalizer_func(int_dict['test_result'])
                                        new_final_variant.append(int_dict)
                                            
                                final_variant = new_final_variant           

                                if (len(final_variant)):
                                    try:
                                        if Dict['attribute']['status']['test_result']['prediction'] not in ['', 'irrelevant']:
                                            for int_dict in final_variant:
                                                if (int_dict['test_result'].strip(' ') == ''):
                                                    if 'test_result' in Dict['attribute']['status'].keys():
                                                        int_dict['test_result'] = Dict['attribute']['status']['test_result']['prediction']
                                            
                                    except Exception as e:
                                        pass

                                if use_flag != 1:
                                    output_mv = final_variant




                                if (output_mv is not None) and (status1 != 'irrelevant'):
                                    Dict['attribute']['interactions'] = {}
                                    output_mv = [dict(s) for s in set(frozenset(d.items()) for d in output_mv)]
                                    Dict['attribute']['interactions'] = output_mv

                                count = count + 1

                                if normalized_bio_name.lower().strip() not in ['tumor']:
                                    # outDict[normalized_bio_name].append(Dict)
                                    outDict[norm_new].append(Dict)

                            if 'tumor' in outDict.keys():
                                del outDict['tumor']

                            for key in outDict.keys():
                                if len(outDict[key]) < 20: # increased the limit from 4 to 20 for larger documents
                                    pass
                                else:
                                    item_count = 0
                                    for item in copy.deepcopy(outDict[key]):
                                        outDict[key][item_count]['attribute']['status'] = {'test_result':{'prediction':'irrelevant'}}
                                        item_count+=1
                                self.outDict[key] = outDict[key]



    def get_pred_pcsubtype(self):
        castrate_resistant = 'castrate_resistant'
        castrate_sensitive = 'castrate_sensitive'
        indeterminate = 'indeterminate'
        irrelevant = 'irrelevant'

        if 'pcsubtype' in self.dict_of_sentences.keys():

            sentence_aspect_label_tuple_list = [(item[3].lower(), None, "") for item in
                                                self.dict_of_sentences['pcsubtype']]
            outDict_values = []
            if 'pcsubtype' in self.BERT_models.keys():
                model_dict = self.BERT_models['pcsubtype']
                if not pandas.isnull(model_dict):

                    #####################profiling##############################
                    algo_eigth_step = time.time()
                    #####################profiling##############################
                    if ('pcsubtype' in self.use_onnx) and model_dict.get('use_onnx', False):
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'],
                            device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=True)
                    else:
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'],
                            device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=False)

                    #####################profiling##############################
                    bert_model_pcsubtype = time.time() - algo_eigth_step
                    self.time_dict['bert_model_pcsubtype'] = bert_model_pcsubtype

                    #####################profiling##############################

                    if len(results) == len(self.dict_of_sentences['pcsubtype']):
                        count = 0
                        for item in self.dict_of_sentences['pcsubtype']:
                            Dict = {}
                            prediction = results[count]
                            Dict['prediction'] = prediction
                            Dict['base_prediction'] = item[0]
                            Dict['start_char'] = item[1]
                            Dict['end_char'] = item[2]
                            Dict['context'] = item[3]
                            Dict['context_start_char'] = item[4]
                            Dict['context_end_char'] = item[5]
                            Dict['confidence'] = dict(prob_df.iloc[count]).get(prediction)
                            Dict['attribute'] = {}

                            #TODO all the fields below in the attribute dictionary are placeholders
                            Dict['attribute']['cancer_pred'] = {}
                            Dict['attribute']['cancer_pred']['prediction'] = self.tumor_type
                            Dict['attribute']['cancer_pred']['confidence'] = ''

                            Dict['attribute']['event_date'] = ''

                            if prediction in (castrate_resistant, castrate_sensitive):
                                Dict['attribute']['assertion'] = 'yes'
                            else:
                                Dict['attribute']['assertion'] = 'no'

                            if prediction == irrelevant:
                                Dict['attribute']['irrelevance'] = 'yes'
                            else:
                                Dict['attribute']['irrelevance'] = 'no'

                            if prediction in (castrate_resistant, castrate_sensitive, irrelevant):
                                Dict['attribute']['negation'] = 'no'
                            else:
                                if get_negation_status(Dict['context'], Dict['base_prediction']):
                                    Dict['attribute']['negation'] = 'yes'
                                else:
                                    Dict['attribute']['negation'] = 'no'

                            if prediction in (castrate_resistant, castrate_sensitive, indeterminate):
                                Dict['attribute']['subject'] = 'self'
                            else:
                                Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                                   Dict['context_start_char'],
                                                                                   Dict['context_end_char'],
                                                                                   Dict['start_char'], Dict['end_char'])

                            count = count + 1
                            outDict_values.append(Dict)

            if len(outDict_values) > 0:
                self.outDict['pcsubtype'] = outDict_values



    def get_pred_alcohol(self):
        if 'alcohol' in self.dict_of_sentences.keys():

            sentence_aspect_label_tuple_list = [(item[3].lower(), None, "") for item in
                                                self.dict_of_sentences['alcohol']]
            outDict_values = []
            if 'alcohol' in self.BERT_models.keys():
                model_dict = self.BERT_models['alcohol']
                if not pandas.isnull(model_dict):

                    #####################profiling##############################
                    algo_eigth_step = time.time()
                    #####################profiling##############################
                    if ('alcohol' in self.use_onnx) and model_dict.get('use_onnx', False):
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'],
                            device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=True)
                    else:
                        results, prob_df = bert_inference(
                            sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                            model=model_dict['model'],
                            device=model_dict['device'],
                            label_list=model_dict['label_list'],
                            max_seq_length=model_dict['max_seq_length'],
                            tokenizer=model_dict['tokenizer'],
                            output_mode=model_dict['output_mode'],
                            model_type=model_dict['model_type'],
                            use_onnx=False)

                    #####################profiling##############################
                    bert_model_alcohol = time.time() - algo_eigth_step
                    self.time_dict['bert_model_alcohol'] = bert_model_alcohol

                    #####################profiling##############################

                    if len(results) == len(self.dict_of_sentences['alcohol']):
                        count = 0
                        for item in self.dict_of_sentences['alcohol']:
                            Dict = {}
                            prediction = results[count]
                            Dict['prediction'] = prediction
                            Dict['base_prediction'] = item[0]
                            Dict['start_char'] = item[1]
                            Dict['end_char'] = item[2]
                            Dict['context'] = item[3]
                            Dict['context_start_char'] = item[4]
                            Dict['context_end_char'] = item[5]
                            Dict['confidence'] = dict(prob_df.iloc[count]).get(prediction)
                            Dict['attribute'] = {}
                            Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                               Dict['context_start_char'],
                                                                               Dict['context_end_char'],
                                                                               Dict['start_char'], Dict['end_char'])
                            count = count + 1
                            outDict_values.append(Dict)

            if len(outDict_values) > 0:
                self.outDict['alcohol'] = outDict_values



    def get_pred_biomarker_name(self):
        outDict_values = []
        seen_preds = set()

        def preprocess_biomarker_sentences(dict_of_sentences):
            if not dict_of_sentences:
                return dict_of_sentences

            # remove ros due to high false positives because of review of systems
            synonyms_to_exclude = ['ros']
            dict_of_sentences = [item for item in dict_of_sentences if item[0].lower().strip() not in synonyms_to_exclude]

            # remove met snippets
            sent_raw_dict = {}
            for item in dict_of_sentences:
                sent_raw_dict.setdefault(item[3].lower().strip(),[]).append(item[0].lower().strip())
            only_met_sents = [k for k,v in sent_raw_dict.items() if (len(set(v))==1) and ('met' in v)]
            return [item for item in dict_of_sentences
                    if not ((item[0].lower().strip()=='met') and (item[3].lower().strip() in only_met_sents))]


        ########################## ner contribution ##########################
        if ('bio_name_ner' in self.BERT_models) and (not pandas.isnull(self.BERT_models['bio_name_ner'])):
            model_dict = self.BERT_models['bio_name_ner']

            chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                                , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
            sentences = [span[0] for span in chunk_spans]

            #####################profiling##############################
            algo_eigth_step = time.time()
            #####################profiling##############################
            use_onnx = ('biomarkers' in self.use_onnx) and model_dict.get('use_onnx', False)
            all_ner_chunk_spans, all_token_spans = bert_ner_inference(
                sentences=sentences,
                model=model_dict['model'],
                device=model_dict['device'],
                label_list=model_dict['label_list'],
                max_seq_length=model_dict['max_seq_length'],
                tokenizer=model_dict['tokenizer'],
                output_mode=model_dict['output_mode'],
                model_type=model_dict['model_type'],
                use_onnx=use_onnx,
                recalibration_dt_model_dict_list=[self.decision_tree_models.get('bio_name_ner')]
            )

            #####################profiling##############################
            bert_model_bio_name = time.time() - algo_eigth_step
            self.time_dict['bert_model_bio_name_ner'] = bert_model_bio_name
            #####################profiling##############################

            if len(all_ner_chunk_spans) == len(chunk_spans):
                for ner_chunk_spans,sent_span in zip(all_ner_chunk_spans,chunk_spans):
                    for ner_chunk_span in ner_chunk_spans:
                        # ner_chunk_span - prediction, start_char, end_char, tag, confidence
                        # sent_span - chunk, start_char, end_char, n_tokens
                        if len(ner_chunk_span[0].strip()) < 2:
                            continue

                        Dict = {}
                        Dict['prediction'] = ner_chunk_span[0]
                        Dict['start_char'] = ner_chunk_span[1] + sent_span[1]
                        Dict['end_char'] = ner_chunk_span[2] + sent_span[1]
                        Dict['context'] = sent_span[0]
                        Dict['context_start_char'] = sent_span[1]
                        Dict['context_end_char'] = sent_span[2]
                        Dict['confidence'] = ner_chunk_span[4]
                        Dict['attribute'] = {}
                        Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                            Dict['context_start_char'],
                                                                            Dict['context_end_char'],
                                                                            Dict['start_char'], Dict['end_char'])
                        if ner_chunk_span[5]==1:
                            Dict['attribute']['recalibration_prediction'] = 'relevant'
                        elif ner_chunk_span[5]==0:
                            Dict['attribute']['recalibration_prediction'] = 'irrelevant'
                        elif ner_chunk_span[5]==-1:
                            Dict['attribute']['recalibration_prediction'] = ''
                        if (Dict['prediction'].lower(), Dict['start_char'], Dict['end_char']) in seen_preds:
                            continue
                        seen_preds.add((Dict['prediction'].lower(), Dict['start_char'], Dict['end_char']))
                        outDict_values.append(Dict)

        ########################## regex contribution ##########################
        if 'biomarkers' in self.dict_of_sentences.keys():
            new_bm_dict_of_sentences = preprocess_biomarker_sentences(self.dict_of_sentences['biomarkers'])
            if not new_bm_dict_of_sentences:
                del self.dict_of_sentences['biomarkers']
            else:
                self.dict_of_sentences['biomarkers'] = new_bm_dict_of_sentences

        if 'biomarkers' in self.dict_of_sentences.keys():
            # drop false positives from regex
            self.dict_of_sentences['biomarkers'] = [item
                                                    for item in self.dict_of_sentences['biomarkers']
                                                    if item[0].strip().lower() not in
                                                    ['','er','pr','met','egfr','ros','alk']]
            if not self.dict_of_sentences['biomarkers']:
                del self.dict_of_sentences['biomarkers']

        if 'biomarkers' in self.dict_of_sentences.keys():
            for item in self.dict_of_sentences['biomarkers']:
                Dict = {}
                Dict['prediction'] = item[0]
                Dict['start_char'] = item[1]
                Dict['end_char'] = item[2]
                Dict['context'] = item[3]
                Dict['context_start_char'] = item[4]
                Dict['context_end_char'] = item[5]
                Dict['confidence'] = 0.5
                Dict['attribute'] = {}
                Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                    Dict['context_start_char'],
                                                                    Dict['context_end_char'],
                                                                    Dict['start_char'], Dict['end_char'])
                Dict['attribute']['recalibration_prediction'] = ''
                if (Dict['prediction'].lower(), Dict['start_char'], Dict['end_char']) in seen_preds:
                    continue
                seen_preds.add((Dict['prediction'].lower(), Dict['start_char'], Dict['end_char']))
                outDict_values.append(Dict)

        # To run Irr model on NER + Regex output
        if ('bio_name_irr' in self.BERT_models):
            sentence_aspect_tuple_list = []

            for item in outDict_values:
                _tuple = (item["context"], item["prediction"], "")
                sentence_aspect_tuple_list.append(_tuple)

            biomarker_name_irr_results, _ = self.get_bert_classification_inference('bio_name_irr', 'biomarkers', sentence_aspect_tuple_list)

            for i, item in enumerate(outDict_values):
                outDict_values[i]["attribute"]["is_irrelevant"] = biomarker_name_irr_results[i].lower().strip() == "irrelevant"

        # This function checks if a biomarker is around met
        def check_biomarker_with_regex(entity):
            for bm in self.regexBiomarkersObj.biomarker_name_regex:
                for regex in self.regexBiomarkersObj.biomarker_name_regex[bm]:
                    if re.search(regex, entity) and re.search(regex, entity).group(0).strip() != "met":
                        return True
            return False
        
        # checking some relevant terms around met
        def check_met_relevant_terms(word_list):
            relevant_term_list = ['variant', 'skipping', 'intron', 'splice', 'exon', 'receptor', 'mutation', 
                                  'tyrosine', 'kinase', 'gene', 'proto-oncogene', 'protein', 'deletion', 'codon']
            return any([i.lower() in relevant_term_list for i in word_list])
        
        # Marking biomarker therapy cases as irrelevant
        # Checking biomarkers around "met" and other relevant terms(this is supposed to be a case sensitive match)
        # check if relevant terms come on the right and biomarkers on both sides in a window fof 3 words
        for i, item in enumerate(outDict_values):
            if not item["attribute"]["is_irrelevant"]:
                context = item["context"]
                end = item["end_char"]
                context_after_bm = context[end:].replace(",", " ")
                context_after_bm_words = context_after_bm.split(" ")[:3]
                context_after_bm_words = " ".join(context_after_bm_words)

                # Catching the cases where biomarkers are present in therapy context
                therapy_regex = re.compile("|".join([r"\b{}\b".format(i) for i in ["therapy", "tki", "inhibitor", "inhibition"]]), re.IGNORECASE)
                if re.search(therapy_regex, context_after_bm_words):
                    outDict_values[i]["attribute"]["is_irrelevant"] = True

            context = item["context"]
            start = item["start_char"]
            end = item["end_char"]

            if item["prediction"] == "met" and not(outDict_values[i]["attribute"]["is_irrelevant"]):
                left_string = context[:start].replace(",", " ")
                right_string = context[end:].replace(",", " ")

                left_words = left_string.split(" ")[-3:]
                right_words = right_string.split(" ")[:3]

                left_string = " ".join(left_words)
                right_string = " ".join(right_words)

                if check_biomarker_with_regex(left_string) or check_biomarker_with_regex(right_string) or check_met_relevant_terms(right_words):
                    outDict_values[i]["attribute"]["is_irrelevant"] = False
                else:
                    outDict_values[i]["attribute"]["is_irrelevant"] = True

        # Handling ALK issues separately
        alk_regex = re.compile(r"(?i)ALK\s*(phosphate|phos)", re.IGNORECASE)

        for i, item in enumerate(outDict_values):
            if item["prediction"].lower().strip() == "alk" and re.search(alk_regex, item["context"]) is not None:
                outDict_values[i]["attribute"]["is_irrelevant"] = True
            elif item["prediction"].lower().strip() in ['alkaline phosphatase', 'phosphatase, alkaline', 'alka', 'alp', 'alk phos', 'alkphos', 'alkaline phenyl phosphatase', 'alkp']:
                outDict_values[i]["attribute"]["is_irrelevant"] = True

        if len(outDict_values) > 0:
            self.outDict['biomarkers'] = outDict_values



    def get_bert_classification_inference(self, model_key, field, sentence_aspect_label_tuple_list):
        if len(sentence_aspect_label_tuple_list)==0:
            return [], None

        if model_key in self.BERT_models.keys():
            model_dict = self.BERT_models[model_key]
            if not pandas.isnull(model_dict):
                #####################profiling##############################
                algo_fifth_step = time.time()
                #####################profiling##############################
                if (field in self.use_onnx) and model_dict.get('use_onnx', False):
                    results, prob_df = bert_inference(
                        sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                        model=model_dict['model'], device=model_dict['device'],
                        label_list=model_dict['label_list'],
                        max_seq_length=model_dict['max_seq_length'],
                        tokenizer=model_dict['tokenizer'],
                        output_mode=model_dict['output_mode'],
                        model_type=model_dict['model_type'],
                        use_onnx=True)
                else:
                    results, prob_df = bert_inference(
                        sentence_aspect_label_tuple_list=sentence_aspect_label_tuple_list,
                        model=model_dict['model'], device=model_dict['device'],
                        label_list=model_dict['label_list'],
                        max_seq_length=model_dict['max_seq_length'],
                        tokenizer=model_dict['tokenizer'],
                        output_mode=model_dict['output_mode'],
                        model_type=model_dict['model_type'],
                        use_onnx=False)

                #####################profiling##############################
                bert_model_comorbidities = time.time() - algo_fifth_step
                self.time_dict[f'bert_model_{model_key}'] = bert_model_comorbidities
                #####################profiling##############################

                return results, prob_df
            else:
                raise (f"Field: {field}, model: {model_key} missing")

        return [], None


    def get_pred_comorbidities(self):
        if 'comorbidities_ner' not in self.BERT_models.keys():
            return

        if 'comorbidities_irr' not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['comorbidities_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self._get_chunk_spans(tokenizer=model_dict['tokenizer']
                                            , max_seq_length=model_dict['max_seq_length'], chunk_overlap_factor=0)
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('comorbidities' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        #####################profiling##############################
        bert_model_comorbidities = time.time() - algo_eigth_step
        self.time_dict['bert_model_comorbidities_ner'] = bert_model_comorbidities
        #####################profiling##############################

        outDict_values = []
        blacklist_terms = ['tremor', 'bleeding', 'dysuria']
        assert len(all_ner_chunk_spans) == len(chunk_spans), "Comorbidities - input and output length mismatch for NER model"

        if len(all_ner_chunk_spans) == len(chunk_spans):
            sentence_aspect_label_tuple_list = []
            List = []
            for ner_chunk_spans, sent_span in zip(all_ner_chunk_spans, chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # List - prediction, start_char, end_char, tag, confidence, sent_chunk, sent_start_char,
                    # sent_end_char, n_tokens
                    if ner_chunk_span[0].lower().strip() in blacklist_terms:
                        continue
                    List.append((ner_chunk_span[0], ner_chunk_span[1], ner_chunk_span[2], ner_chunk_span[3],
                                 ner_chunk_span[4], sent_span[0], sent_span[1], sent_span[2], sent_span[3]))
                    sentence_aspect_label_tuple_list.append((sent_span[0], ner_chunk_span[0], ""))

            irr_results, _ = self.get_bert_classification_inference('comorbidities_irr', 'comorbidities', sentence_aspect_label_tuple_list)

            assert len(irr_results) == len(List), "Comorbidities - input and output length mismatch for irrelevance model"
            if len(irr_results) == len(List):
                count = 0
                for item in List:
                    Dict = {}
                    Dict['prediction'] = item[0]
                    Dict['base_prediction'] = item[0]
                    Dict['start_char'] = item[1] + item[6]
                    Dict['end_char'] = item[2] + item[6]
                    Dict['context'] = item[5]
                    Dict['context_start_char'] = item[6]
                    Dict['context_end_char'] = item[7]
                    Dict['confidence'] = item[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                       Dict['context_start_char'],
                                                                       Dict['context_end_char'],
                                                                       Dict['start_char'], Dict['end_char'])
                    Dict['attribute']['status'] = irr_results[count]
                    count = count + 1
                    outDict_values.append(Dict)

        if len(outDict_values) > 0:
            self.outDict['comorbidities'] = outDict_values


    def get_tokenized_sentence_tuple(self, text, start, end):
        txt_a = text[:start] + '<' + text[start:end] + '>' + text[end:]
        txt_b = '<' + text[start:end] + '>'
        return (txt_a, txt_b, "")
    

    def get_pred_site(self):
        if 'site_ner' not in self.BERT_models.keys():
            return

        if 'site_nerc' not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['site_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self.natural_chunk_dict.get(model_dict['max_seq_length'], None)
        if not chunk_spans:
            return
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('site' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        #####################profiling##############################
        bert_model_site = time.time() - algo_eigth_step
        self.time_dict['bert_model_site_ner'] = bert_model_site
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "Site - input and output length mismatch for NER model"

        if len(all_ner_chunk_spans) == len(chunk_spans):
            sub_sent_spans = self.get_sub_sent_spans(model_dict['tokenizer'], model_dict['max_seq_length'])
            sentence_aspect_label_tuple_list = []
            List = []
            for ner_chunk_spans, sent_span in zip(all_ner_chunk_spans, chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # List - prediction, start_char, end_char, tag, confidence, sent_chunk, sent_start_char,
                    # sent_end_char, n_tokens
                    List.append((ner_chunk_span[0], ner_chunk_span[1], ner_chunk_span[2], ner_chunk_span[3],
                                 ner_chunk_span[4], sent_span[0], sent_span[1], sent_span[2], sent_span[3]))
                    sentence_aspect_label_tuple_list.append(self.get_central_tokenized_sentence_tuple(model_dict['tokenizer'], sub_sent_spans, ner_chunk_span[1]+sent_span[1], ner_chunk_span[2]+sent_span[1], model_dict['max_seq_length'], aspect_encloser=('<','>')))

            normalize_results, _ = self.get_bert_classification_inference('site_nerc', 'site', sentence_aspect_label_tuple_list)

            assert len(normalize_results) == len(List), "Site - input and output length mismatch for NERC model"
            if len(normalize_results) == len(List):
                count = 0
                for item in List:
                    Dict = {}
                    Dict['prediction'] = str(normalize_results[count])
                    Dict['base_prediction'] = item[0]
                    Dict['start_char'] = item[1] + item[6]
                    Dict['end_char'] = item[2] + item[6]
                    Dict['context'] = item[5]
                    Dict['context_start_char'] = item[6]
                    Dict['context_end_char'] = item[7]
                    Dict['confidence'] = item[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                       Dict['context_start_char'],
                                                                       Dict['context_end_char'],
                                                                       Dict['start_char'], Dict['end_char'])
                    count = count + 1
                    outDict_values.append(Dict)

        if len(outDict_values) > 0:
            self.outDict['site'] = outDict_values


    def get_pred_metastasis(self):
        if 'metastasis_ner' not in self.BERT_models.keys():
            return

        if 'metastasis_nerc' not in self.BERT_models.keys():
            return

        model_dict = self.BERT_models['metastasis_ner']
        if pandas.isnull(model_dict):
            return

        chunk_spans = self.natural_chunk_dict.get(model_dict['max_seq_length'], None)
        if not chunk_spans:
            return
        sentences = [span[0] for span in chunk_spans]

        #####################profiling##############################
        algo_eigth_step = time.time()
        #####################profiling##############################
        use_onnx = ('metastasis' in self.use_onnx) and model_dict.get('use_onnx', False)
        all_ner_chunk_spans, all_token_spans = bert_ner_inference(
            sentences=sentences,
            model=model_dict['model'],
            device=model_dict['device'],
            label_list=model_dict['label_list'],
            max_seq_length=model_dict['max_seq_length'],
            tokenizer=model_dict['tokenizer'],
            output_mode=model_dict['output_mode'],
            model_type=model_dict['model_type'],
            use_onnx=use_onnx)

        #####################profiling##############################
        bert_model_metastasis = time.time() - algo_eigth_step
        self.time_dict['bert_model_metastasis_ner'] = bert_model_metastasis
        #####################profiling##############################

        outDict_values = []
        assert len(all_ner_chunk_spans) == len(chunk_spans), "Metastasis - input and output length mismatch for NER model"

        if len(all_ner_chunk_spans) == len(chunk_spans):
            sub_sent_spans = self.get_sub_sent_spans(model_dict['tokenizer'], model_dict['max_seq_length'])
            sentence_aspect_label_tuple_list = []
            List = []
            for ner_chunk_spans, sent_span in zip(all_ner_chunk_spans, chunk_spans):
                for ner_chunk_span in ner_chunk_spans:
                    # List - prediction, start_char, end_char, tag, confidence, sent_chunk, sent_start_char,
                    # sent_end_char, n_tokens
                    List.append((ner_chunk_span[0], ner_chunk_span[1], ner_chunk_span[2], ner_chunk_span[3],
                                 ner_chunk_span[4], sent_span[0], sent_span[1], sent_span[2], sent_span[3]))
                    sentence_aspect_label_tuple_list.append(self.get_central_tokenized_sentence_tuple(model_dict['tokenizer'], sub_sent_spans, ner_chunk_span[1]+sent_span[1], ner_chunk_span[2]+sent_span[1], model_dict['max_seq_length'], aspect_encloser=None))

            normalize_results, _ = self.get_bert_classification_inference('metastasis_nerc', 'metastasis', sentence_aspect_label_tuple_list)

            assert len(normalize_results) == len(List), "Metastasis - input and output length mismatch for NERC model"
            if len(normalize_results) == len(List):
                count = 0
                for item in List:
                    Dict = {}
                    Dict['prediction'] = str(normalize_results[count])
                    Dict['base_prediction'] = item[0]
                    Dict['start_char'] = item[1] + item[6]
                    Dict['end_char'] = item[2] + item[6]
                    Dict['context'] = item[5]
                    Dict['context_start_char'] = item[6]
                    Dict['context_end_char'] = item[7]
                    Dict['confidence'] = item[4]
                    Dict['attribute'] = {}
                    Dict['attribute']['subject'] = self._returnSubject(Dict['context'],
                                                                       Dict['context_start_char'],
                                                                       Dict['context_end_char'],
                                                                       Dict['start_char'], Dict['end_char'])
                    count = count + 1
                    outDict_values.append(Dict)

        if len(outDict_values) > 0:
            self.outDict['metastasis'] = outDict_values



if __name__=='__main__':
    infernece = Inference()

