# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
# import logging
import os
import sys
from io import open
import pandas

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import xml.etree.ElementTree as ET
import numpy as np
from torch import Tensor
from sklearn.metrics import roc_curve, auc

# logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SurgeryProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train_2806.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate_2806.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['irrelevant', 'positive']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
#                 text_b = None
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class RadiationProcessor(DataProcessor):
    """Processor for RT. classes : relevant and irrelevant """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "valid.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['relevant','irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context_final']
        # print(DF.columns)
        DF['label'] = DF['gt']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        ##DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


def semeval2014term_to_aspectsentiment_hr(filename, remove_conflicting=True):
    sentimap = {
        'positive': 'POS',
        'negative': 'NEG',
        'neutral': 'NEU',
        'conflict': 'CONF',
    }

    def transform_aspect_term_name(se):
        return se

    with open(filename) as file:

        sentence_elements = ET.parse(file).getroot().iter('sentence')

        sentences = []
        aspect_term_sentiments = []
        classes = set([])

        for j, s in enumerate(sentence_elements):
            # review_text = ' '.join([el.text for el in review_element.iter('text')])

            sentence_text = s.find('text').text
            aspect_term_sentiment = []
            for o in s.iter('aspectTerm'):
                aspect_term = transform_aspect_term_name(o.get('term'))
                classes.add(aspect_term)
                sentiment = sentimap[o.get('polarity')]
                if sentiment != 'CONF':
                    aspect_term_sentiment.append((aspect_term, sentiment))
                else:
                    if remove_conflicting:
                        pass
                        # print('Conflicting Term found! Removed!')
                    else:
                        aspect_term_sentiment.append((aspect_term, sentiment))

            if len(aspect_term_sentiment) > 0:
                aspect_term_sentiments.append(aspect_term_sentiment)
                sentences.append(sentence_text)

        cats = list(classes)
        cats.sort()

    idx2aspectlabel = {k: v for k, v in enumerate(cats)}
    sentilabel2idx = {"NEG": 1, "NEU": 2, "POS": 3, "CONF": 4}
    idx2sentilabel = {k: v for v, k in sentilabel2idx.items()}

    return sentences, aspect_term_sentiments, (idx2aspectlabel, idx2sentilabel)


def generate_qa_sentence_pairs_nosampling(sentences, aspecterm_sentiments):
    sentence_pairs = []
    labels = []

    for ix, ats in enumerate(aspecterm_sentiments):
        s = sentences[ix]
        for k, v in ats:
            sentence_pairs.append((s, k))
            labels.append(v)

    return sentence_pairs, labels


class SemEval2014AtscProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.xml"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "test.xml"), "dev")

    def get_labels(self):
        """See base class."""
        return ["POS", "NEG", "NEU"]

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        sentences, aspects, idx2labels = semeval2014term_to_aspectsentiment_hr(corpus, remove_conflicting=True)

        sentences, labels = generate_qa_sentence_pairs_nosampling(sentences, aspects)

        examples = []

        for i, sentence_pair in enumerate(sentences):

            guid = "%s-%s" % (set_type, i)
            try:
                text_a = sentence_pair[0]
                text_b = sentence_pair[1]
                label = labels[i]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class BiomarkersProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['intermediate', 'unstable', 'stable', 'low', 'other', 'pending', 'high', 'expressed', 'overexpressed', 'indeterminate', 'qns', 'positive', 'irrelevant', 'negative']
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    
class BiomarkerAndAttributesNerProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['O', 'B-GENE', 'I-GENE', 'B-GENE_RESULT', 'B-CATEGORICAL', 'B-NUMERICAL', 'B-NUMERICAL_UNIT', 'I-NUMERICAL', 'I-GENE_RESULT', 'I-CATEGORICAL', 'B-TEST_TYPE', 'B-VARIANT', 'I-TEST_TYPE', 'I-VARIANT', 'I-NUMERICAL_UNIT',
 'B-DNA_ALTERATION', 'I-DNA_ALTERATION', 'B-EXON_VALUE', 'I-EXON_VALUE']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BiomarkersIrrProcessor(DataProcessor):
    """Processor for the classifying snippets based on the biomarkers phrase as relevant or irrelevant"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")
    def get_labels(self):
        """See base class."""
        return ['irrelevant', 'relevant']
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]
        DF.reset_index(inplace=True)
        examples = []
        for key, row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BiomarkersNerProcessor(DataProcessor):
    """Processor for the NER to identify biomarker name"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['O', 'B-GENE', 'I-GENE']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []

        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
class GradeProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['yes','no']
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF['line'] = DF['Text Snippet']
        DF['result'] = DF['tumor_grade_present']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    


class MedsTemporalityProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "validate.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['past','current','future','irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        # print(DF.columns)
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
 
    
class SmokingProcessor(DataProcessor):
    """Processor for the classifying snippets based on the smoking status into following classes
    never_smoked, irrelevant, past_smoker, current_smoker"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['Current Smoker', 'Never Smoked', 'Past Smoker']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
class TreatmentResponseNERProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for treatment response into following classes
    'O', 'PROGRESSION'
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O","B-PROGRESSION","I-PROGRESSION"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row["text_b"]
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TumorNERProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for tumor into 3 classes: ["O", "B-tumor", "I-tumor"]
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O", "B-tumor", "I-tumor"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MedicationNERProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for medication into the following classes:
    "B-Drug Name", "B-Drug Dosage/Treatment Day", "B-Drug Dosage Units", "B-Route of Administration", "O", "I-Drug Dosage/Treatment Day", "I-Route of Administration", "B-Therapy Ongoing?", "I-Drug Name", "B-Days/Cycle", "I-Days/Cycle", "I-Therapy Ongoing?", "I-Drug Dosage Units"
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["B-Drug Name", "B-Drug Dosage/Treatment Day", "B-Drug Dosage Units", "B-Route of Administration", "O", "I-Drug Dosage/Treatment Day", "I-Route of Administration", "B-Therapy Ongoing?", "I-Drug Name", "B-Days/Cycle", "I-Days/Cycle", "I-Therapy Ongoing?", "I-Drug Dosage Units"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RadiationNERProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for radiation into the following classes:
    "O", "B-Modality", "I-Modality", "B-Laterality", "I-Laterality"
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O", "B-Modality", "I-Modality", "B-Laterality", "I-Laterality"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        
class Mets1Processor(DataProcessor):
    """Processor for the metastasis 1 class : positive/local (either local or distal met), irrelevant and indeterminate . Negation is taken care of separately using negex"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "valid.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['positive/local', 'irrelevant', 'indeterminate']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['text']
        # print(DF.columns)
        DF['label'] = DF['final_labels']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    


class Mets2Processor(DataProcessor):
    """Processor for the metastasis 2 class : distal(yes) vs local """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "valid.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['yes','local']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['text']
        # print(DF.columns)
        DF['label'] = DF['final_labels']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MetSiteProcessor(DataProcessor):
    """Processor for site of met class, a multi label classification problem"""
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")
        
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "valid.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['bone','brain','liver','lung','site unknown','others','lymph node', 'distant lymph node']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        all_labels = self.get_labels()
        
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                curr_labels = row['label'].split(",") ## as multiple labels per snippet, comma separated
                labels = []
                for label in all_labels:
                    if label in curr_labels:
                        labels.append(float(1))
                    else:
                        labels.append(float(0))
                        
            except IndexError:
                continue

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels))
        return examples
    
class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class StageProcessor(DataProcessor):
    """Processor for the stage at initial dx class : confirmatory initial stage, non confirmatory initial stage and irrelevant"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "valid.csv"), "dev")

    def get_labels(self):
        """See base class."""
        ##return ["positive","negative","irrelevant","pending","qns","others"]
        return ['confirmatory','non_confirmatory','irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""


        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        # print(DF.columns)
        DF['label'] = DF['gt']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['key'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MedicationProcessor(DataProcessor):
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "medication_relevent_model_train_file_2809.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "medication_relevent_model_val_file_2809.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['relevant', 'irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF['line'] = DF['context'].apply(lambda x : str(x).lower())
        DF['result'] = DF['GroundTruthLabel'].apply(lambda x : str(x).lower())
        ##DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['keyword'])) & (~pandas.isnull(DF['result']))]
        DF = DF.loc[(~pandas.isnull(DF['line']))  & (~pandas.isnull(DF['result']))]
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        ##DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                ##text_b = row['keyword']
                label = row['result']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PCSubtypeProcessor(DataProcessor):
    """Processor for the classifying snippets based on the prostate cancer subtype:
        Castration Sensitive, Castration Resistant, Indeterminate, Irrelevant """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "dev.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['castrate_resistant', 'castrate_sensitive', 'irrelevant', 'indeterminate']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF['line'] = DF['text']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class TumorIrrProcessor(DataProcessor):
    """Processor for the classifying snippets based on the tumor phrase as relevant or irrelevant"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['irrelevant', 'relevant']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AlcoholProcessor(DataProcessor):
    """Processor for the classifying snippets based on alcohol consumption:
         'no current drinker', 'abuse', 'irrelevant', 'never used alcohol', 'current drinker' """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["Abuse", "Current Drinker", "No Current Drinker"]

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF['line'] = DF['text']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = None
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MenopauseProcessor(DataProcessor):
    """Processor for Menopausal Status. classes : premenopausal, postmenopausal, indeterminate, irrelevant """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "mp_classify_v2_train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "mp_classify_v2_val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['premenopausal', 'postmenopausal', 'indeterminate', 'irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    

class BERTNEROncologyv2Processor(DataProcessor):
    """Processor for the classifying tokens from snippets into B,I,O tags to  get NER entities for
    Grade, Stage, Procedure Type, Performance Status, Death Entity, Histology
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['O', 'B-Mortality_Status', 'I-Mortality_Status', 'B-Staging', 'B-Histological_Type', 'I-Histological_Type', 'I-Staging', 'B-Grade', 'I-Grade', 'B-Cancer_Surgery', 'I-Cancer_Surgery', 'B-Performance_Status', 'I-Performance_Status']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[~pandas.isnull(DF['text'])]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RaceEthnicityNERProcessor(DataProcessor):
    """Processor for the classifying tokens from snippets into B,I,O tags to  get NER entities for
    Race, Ethnicity
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O", "I-ETHNICITY", "I-RACE", "B-RACE", "B-ETHNICITY"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[~pandas.isnull(DF['text'])]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RaceNERCProcessor(DataProcessor):
    """Processor for the identifing NERC labels for 
    Race
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["white", "american indian or alaska native", "black or african american", "asian", "native hawaiian or other pacific islander", "other race", "irrelevant"]   # internal prediction, same order id2label
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[~pandas.isnull(DF['text'])]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class EthnicityNERCProcessor(DataProcessor):
    """Processor for the identifing NERC labels for 
    Ethnicity
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["not hispanic or latino", "hispanic or latino", "ashkenazi jewish", "irrelevant"]   # internal prediction, same order id2label
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[~pandas.isnull(DF['text'])]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class SurgeryIrrProcessor(DataProcessor):
    """Processor for Grade NER-C. classes : ['Low Grade', 'G2-Moderately Differentiated', 'GX-Grade Cannot be Assessed', 'G1-Well Differentiated', 'G3-4 Poorly Differentiated or Undifferentiated', 'G4-Undifferentiated', 'G3-Poorly Differentiated', 'High Grade'] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["irrelevant", "relevant"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class GradeNERCProcessor(DataProcessor):
    """Processor for Grade NER-C. classes : ['Low Grade', 'G2-Moderately Differentiated', 'GX-Grade Cannot be Assessed', 'G1-Well Differentiated', 'G3-4 Poorly Differentiated or Undifferentiated', 'G4-Undifferentiated', 'G3-Poorly Differentiated', 'High Grade'] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['G1-Well Differentiated', 'G2-Moderately Differentiated', 'G3-Poorly Differentiated', 'G4-Undifferentiated', 'GX-Grade Cannot be Assessed', 'Gleason Grade Group 1', 'Gleason Grade Group 2', 'Gleason Grade Group 3', 'Gleason Grade Group 4', 'Gleason Grade Group 5', 'Gleason Score 7', "irrelevant"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StageNERCProcessor(DataProcessor):
    """Processor for Stage NER-C. classes : ['M0', 'N0', 'Stage IV', 'Stage IVB', 'N1', 'T3', 'T2', 'Stage IA', 'T1c', 'N2', 'Stage I', 'Stage IVA', 'Stage IIIA', 'Stage IIA', 'T1b', 'Stage IIB', 'T4', 'Stage IIIB', 'NX', 'M1', 'M1c', 'M1b', 'T1', 'N3', 'Stage III', 'TX', 'M1a', 'T2a', 'Stage IB', 'Stage IIIC', 'Stage II', 'Extensive', 'T1a', 'T2b', 'Stage IA2', 'MX', 'Limited', 'Stage IA3', 'Stage Unknown', 'T4b', 'N1a', 'N2a', 'T2c', 'T4a', 'Stage IIC', 'T3b', 'T3a', 'Ta', 'Stage IVC', 'N1b', 'pM1c', 'pM1b', 'N2b', 'pT1b', 'pT1', 'pT2', 'pN0', 'pT2a', 'Stage IA1', 'pT3', 'pT1a', 'pT1c', 'pT3b', 'Multi-labelled', 'Irrelevant', 'Others']"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['stage iiib', 'n2a', 'stage iiic', 't4a', 'm0', 'n0', 'stage ia2', 't1b', 'stage iib', 'stage iic', 't4b', 'n2', 't4d', 'ptis', 'm1a(1)', 'n1', 'tx', 'mx', 'stage iva', 't1c', 'stage ia', 'pn3c', 'stage ivb', 'stage ii', 'pn2', 'pt2a', 'stage iia', 'stage i', 'pn3b', 'stage iv', 'multi-labelled', 't3', 'stage iiia', 'stage iii', 'pn2b', 'm1', 'stage ia3', 't4', 'nx', 't1', 'cm0(i+)', 't2a', 'm1c', 'm1b', 't2', 'pn1c', 'pta', 'n3', 't2b', 'stage b', 'stage ib', 'n3c', 'figo stage iiia1(i)', 'pt1c1', 'n1b', 'stage ia1', 'pn1', 'pt0', 'pn2a', 'pt1c3', 'pn0 (i-)', 'pm1c(1)', 'stage ivc', 'ta', 'pm1b(0)', 'pt4c', 'pn3', 'pm1', 't3c', 'm1b(0)', 'pn1b', 'pt3', 'm1a', 'pn0', 'pt3b', 'stage iiia2', 'm1d', 'n1a', 'pt1a', 'pt4d', 'n2b', 't1mi', 'pm1a', 'n1c', 'pm1d(1)', 'pt3a', 'pm1c', 'pnx', 't1a', 'm1c(1)', 't2c', 'm1d(0)', 'pn2c', 'pn1a', 'stage 0is', 'tis', 't3a', 'stage iiia1', 'pn0(i+)', 'pt2b', 'pt4a', 'pm1d(0)', 'stage 0', 'pn3a', 'pm1c(0)', 'n3a', 'pm1d', 't1c3', 't0', 'stage iiid', 'pn0 (mol+)', 'm1c(0)', 'stage ic', 'n3b', 'pt4b', 'n2c', 'pt3c', 'pt2', 'pm1b', 'limited', 'figo stage iiia1(ii)', 'pm1a(1)', 'pt1b', 'pt4', 'pn1mi', 't1c2', 'm1d(1)', 'stage d', 'pt1mi', 't3b', 'pt1c2', 'stage c', 'stage 0a', 't4c', 'pt1c', 'figo stage ic3', 'extensive', 'pm1b(1)', 'm1b(1)', 'n1mi', 'n0(i+)', 'pn0 (mol-)', 'pt1', 'figo stage ic1', 't1c1', 'm1a(0)', 'pm1a(0)', 'pt2c', 'stage a', 'figo stage ic2', 'irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PerformanceStatusNERCProcessor(DataProcessor):
    """Processor for Performance Status NER-C. classes : ['0 (90% - 100%)', '3 (30% - 40%)', '4 (10% - 20%)', '1 (70% - 80%)', '2 (50% - 60%)', 'Irrelevant', '5 (0%)'] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['1 (ECOG)', '0 (ECOG)', 'Irrelevant', '2 (ECOG)', 'Karnofsky Performance Status 90', 'Karnofsky Performance Status 70', '4 (ECOG)', '3 (ECOG)', 'Karnofsky Performance Status 50', 'Karnofsky Performance Status 100', 'Karnofsky Performance Status 30', 'Karnofsky Performance Status 60', 'Karnofsky Performance Status 20', 'Karnofsky Performance Status 80', 'Karnofsky Performance Status 10', 'Karnofsky Performance Status 40', 'Karnofsky Performance Status 0', '5 (ECOG)']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RouteNERCProcessor(DataProcessor):
    """Processor for Route NER-C. classes : ["nasal route", "oral route", "subcutaneous route", "prolonged release subcutaneous implant", "intramuscular route", "intravenous route", "intravesical route", "irrelevant"] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["nasal route", "oral route", "subcutaneous route", "prolonged release subcutaneous implant", "intramuscular route", "intravenous route", "intravesical route", "irrelevant"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    
class TherapyNERCProcessor(DataProcessor):
    """Processor for Route NER-C. classes : ["yes", "irrelevant", "no"] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["yes", "irrelevant", "no"]
                
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ComorbiditiesNERProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for comorbidities into following classes
    'O', 'COMORBIDITY'
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O","I-comorbidity","B-comorbidity"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ComorbiditiesIrrProcessor(DataProcessor):
    """Processor for the classifying snippets based on the comorbidities phrase as relevant or irrelevant"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['irrelevant', 'relevant']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ComorbiditiesNERCProcessor(DataProcessor):
    """Processor for the classifying snippets based on the comorbidities phrase as one of the 30 normalized classes"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['leukemia (other than aml)', 'unspecified chronic kidney disease', 'kidney disease with unknown stage or severity', 'unspecified cirrhosis of liver', 'human immunodeficiency virus (hiv)', 'lymphoma', 'hemiplegia', 'chronic kidney disease, stage 4', 'diabetes without chronic complications', 'myocardial infarction (any history of)', 'congestive heart failure (history of treatment for)', 'liver disease, mild', 'Unspecified diabetes mellitus', 'autoimmune disease', 'chronic kidney disease, stage 2', 'liver disease, severe', 'chronic kidney disease, stage 1', 'metastatic solid tumor (other than tumor type) (any history of)', 'aids/hiv with opportunistic infection', 'connective tissue disease', 'liver disease, moderate', "cerebrovascular accident (history of including tia's)", 'hiv with no opportunistic infection', 'chronic obstructive pulmonary disease (including chronic bronchitis and emphysema)', "alzheimer's or other dementia", 'chronic kidney disease, stage 5', 'surgical treatment for peripheral vascular disease (history of)', 'ulcer disease (do not include gerd or ulcerative colitis)', 'chronic kidney disease, stage 3', 'diabetes with chronic complications']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SiteNERProcessor(DataProcessor):
    """Processor for the identifying entitines within a snippets based on the token classification into following classes
    'O', 'SITE'
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O","B-SITE","I-SITE"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SiteNERCProcessor(DataProcessor):
    """Processor for the classifying snippets based on the site phrase as one of the normalized classes
    ['OTHER_SITE', 'IMAGING_ANATOMIC_SITE', 'METASTATIC_SITE', 'RADIATION_TREATMENT_ANATOMIC_SITE']
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['IMAGING_ANATOMIC_SITE', 'OTHER_SITE', 'METASTATIC_SITE', 'RADIATION_TREATMENT_ANATOMIC_SITE']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MetastasisNERProcessor(DataProcessor):
    """Processor for the identifying entitines within a snippets based on the token classification into following classes
    'O', 'MET'
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["O","B-MET","I-MET"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = None
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MetastasisNERCProcessor(DataProcessor):
    """Processor for the classifying snippets based on the site phrase as one of the normalized classes
    ['distant', 'local']
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['distant', 'irrelevant', 'local']

    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    

class HistologyIrrProcessor(DataProcessor):
    """Processor for the classifying snippets based on the comorbidities phrase as relevant or irrelevant"""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")
    def get_labels(self):
        """See base class."""
        return ['irrelevant', 'relevant']
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus, lineterminator='\n')
        DF = DF.loc[(~pandas.isnull(DF['text_a'])) & (~pandas.isnull(DF['text_b'])) & (~pandas.isnull(DF['labels']))]
        DF.reset_index(inplace=True)
        examples = []
        for key, row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TumorNERCProcessor(DataProcessor):
    """Processor for the classifying snippets based on the token classification for tumor into 85 classes
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ['melanoma of the skin', 'cancer of lung and bronchus', 'hodgkin lymphoma', 'prostate cancer', 'urinary bladder cancer', 'myelodysplastic syndrome', 'multiple myeloma', 'gall bladder cancer', 'chronic myeloid leukemia', 'myeloproliferative disease', 'follicular lymphoma', 'endometrial cancer', 'leukemia, other', 'thyroid cancer', 'hepatocellular carcinoma', 'cancer of vulva', 'cancer of other male genital organs', 'menigeal cancer', 'testicular cancer', 'acute monocytic leukemia', 'cancer of vagina', 'ocular cancer', 'cancer of other urinary organs', 'cranial nerves other nervous system cancer', 'chronic lymphocytic leukemia', 'plasma cell leukemia', 'malignant neoplasm, unspecified', 'sarcoma', 'esophageal cancer', 'cervical cancer', 'cancer of thymus', 'small intestine cancer', 'cancer of palate', 'cancer of floor mouth', 'gum and other mouth cancer', 'skin cancer', 'mature t-cell and nk-cell neoplasms', 'pancreatic cancer', 'cancer of other connective and soft tissue', 'uterine cancer', 'cancer of pyriform sinus', 'adrenal gland', 'tonsil cancer', 'salivary gland cancer', 'parotid gland cancer', 'urothelial cancer', 'cancer of liver and intrahepatic bile ducts', 'rectal cancer', 'cancer of kidney, except renal pelvis', 'biliary tract cancer', 'colorectal cancer', 'cancer of retroperitoneum and peritoneum', 'tongue cancer', 'kaposi sarcoma', 'cancer of other female reproductive organs', 'other & unspecified malignant neoplasm of lymphoid, hematopoietic and related tissue', 'cancer of bones and joints', 'oropharyngeal cancer', 'colon cancer', 'ovarian cancer', 'angiosarcoma', 'breast cancer', 'cancer of renal pelvis', 'penile cancer', 'extramedullary plasmacytoma', 'anal cancer', 'solitary plasmacytoma', 'accessory sinuses', 'cancer of lip', 'cancer of appendix', 'non-hodgkin lymphoma', 'gastric cancer', 'cancer of nasal cavity and middle ear', 'cancer of peripheral nerves and autonomic nervous system', 'glioblastoma', 'head and neck cancer', 'laryngeal cancer', 'cancer of the heart, mediastinum and pleura', 'cancer of other endocrine glands', 'malignant mast cell neoplasm', 'hypopharyngeal cancer', 'mature b-cell neoplasms', 'nasopharyngeal cancer', 'brain cancer', 'irrelevant']
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""

        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF = DF.loc[(~pandas.isnull(DF['text'])) & (~pandas.isnull(DF['labels']))]

        DF.reset_index(inplace=True)
        examples = []

        for key, row in DF.iterrows():

            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['text_a']
                text_b = row['text_b']
                label = row['labels']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class DrugNERCProcessor(DataProcessor):
    """Processor for Route NER-C. classes : ["cancer_meds", conmeds", "irrelevant"] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["cancer_meds", "conmeds", "irrelevant"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ModalityIrrProcessor(DataProcessor):
    """Processor for Route NER-C. classes : ["irrelevant", "relevant"] """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.csv"), "dev")

    def get_labels(self):
        """See base class."""
        return ["irrelevant", "relevant"]
    
    def _create_examples(self, corpus, set_type):
        """Creates examples for the training and dev sets."""
        try:
            DF = pandas.read_csv(corpus)
        except Exception:
            DF = pandas.read_csv(corpus,lineterminator = '\n')
        DF.columns = [str(x).lower() for x in DF.columns]
        DF['line'] = DF['context']
        DF['label'] = DF['label']
        DF['keyword'] = DF['key_word']
        DF = DF.loc[(~pandas.isnull(DF['line'])) & (~pandas.isnull(DF['label'])) & (~pandas.isnull(DF['keyword']))]
        
        DF['line'] = DF['line'].apply(lambda x : str(x).lower())
        DF['label'] = DF['label'].apply(lambda x : str(x).lower())
        DF['keyword'] = DF['keyword'].apply(lambda x : str(x).lower())
        
        DF.reset_index(inplace = True)
        examples = []
        for key,row in DF.iterrows():
            guid = "%s-%s" % (set_type, key)
            try:
                text_a = row['line']
                text_b = row['keyword']
                label = row['label']
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    tokenized_examples = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        elif output_mode == "multilabelclassification":  ## expect a vector of 1 and 0 for each sample
            label_id = example.label
        elif output_mode == "tokenclassification":
            label_id = None
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     if output_mode != 'multilabelclassification':
        #         logger.info("label: %s (id = %d)" % (example.label, label_id))
        #     else:
        #         logger.info("label: %s (id = %s)" % (";".join([str(x) for x in example.label]), ";".join([str(x) for x in label_id])))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
        tokenized_examples.append(" ".join(
                    [str(x) for x in tokens]))

    return features, tokenized_examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()



def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def acc_and_f1macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1_macro": f1_macro,
    }


def label_wise_aucroc(preds, labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_labels = labels.shape[1]
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return {'roc_auc': roc_auc  }
    
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, sentences=None, error_file=None, label_list=None):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "semeval2014-atsc":
        return acc_and_f1macro(preds, labels)
    elif task_name == "biomarkers":
        return acc_and_f1macro(preds, labels)
    elif task_name == "medstemporality":
        return acc_and_f1macro(preds,labels)
    elif task_name == "smoking":
        return acc_and_f1macro(preds,labels)
    elif task_name == 'grade_irr':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'mets1':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'mets2':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'stage_irr':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'siteofmets':
        return label_wise_aucroc(preds, labels)
    elif task_name == 'surgery':
        return label_wise_aucroc(preds, labels)
    elif task_name == 'radiation':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'pcsubtype':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'tumor_irr':
        return acc_and_f1macro(preds, labels)
    elif task_name == 'alcohol':
        return acc_and_f1macro(preds,labels)
    elif task_name == 'menopause_cls':
        return acc_and_f1macro(preds, labels)
    elif task_name == "treatmentresponse_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "tumor_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "tumor_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "medication_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "radiation_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "grade_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "stage_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "performance_status_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "route_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "therapy_ongoing_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "comorbidities_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "comorbidities_irr":
        return acc_and_f1macro(preds,labels)
    elif task_name == "comorbidities_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "bio-ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "bio_name_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "site_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "site_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "metastasis_ner":
        return acc_and_f1macro(preds,labels)
    elif task_name == "metastasis_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "drug_nerc":
        return acc_and_f1macro(preds,labels)
    elif task_name == "radiation_modality_irr":
        return acc_and_f1macro(preds,labels)
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "semeval2014-atsc":SemEval2014AtscProcessor,
    "biomarkers":BiomarkersProcessor,
    "bio-ner": BiomarkerAndAttributesNerProcessor,
    "bio_name_ner": BiomarkersNerProcessor,
    "bio_name_irr": BiomarkersIrrProcessor,
    "medstemporality" : MedsTemporalityProcessor,
    "smoking" : SmokingProcessor,
    "grade_irr" : GradeProcessor,
    "mets1" : Mets1Processor,
    "mets2" : Mets2Processor,
    "siteofmets" : MetSiteProcessor,
    "stage_irr" : StageProcessor,
    "surgery": SurgeryProcessor,
    "surgery_irr": SurgeryIrrProcessor,
    "radiation" : RadiationProcessor,
    "medication": MedicationProcessor,
    "pcsubtype": PCSubtypeProcessor,
    "tumor_irr": TumorIrrProcessor,
    "alcohol": AlcoholProcessor,
    "menopause_cls": MenopauseProcessor,
    "grade_nerc": GradeNERCProcessor,
    "stage_nerc": StageNERCProcessor,
    "performance_status_nerc": PerformanceStatusNERCProcessor,
    "route_nerc": RouteNERCProcessor,
    "therapy_ongoing_nerc": TherapyNERCProcessor,
    "treatmentresponse_ner": TreatmentResponseNERProcessor,
    "tumor_ner": TumorNERProcessor,
    "tumor_nerc": TumorNERCProcessor,
    "medication_ner": MedicationNERProcessor,
    "radiation_ner": RadiationNERProcessor,
    "comorbidities_ner": ComorbiditiesNERProcessor,
    "comorbidities_irr": ComorbiditiesIrrProcessor,
    "comorbidities_nerc": ComorbiditiesNERCProcessor,
    "oncologyv2": BERTNEROncologyv2Processor,
    "site_ner": SiteNERProcessor,
    "site_nerc": SiteNERCProcessor,
    "metastasis_ner": MetastasisNERProcessor,
    "metastasis_nerc": MetastasisNERCProcessor,
    "race_ethnicity_ner": RaceEthnicityNERProcessor,
    "race_nerc": RaceNERCProcessor,
    "ethnicity_nerc": EthnicityNERCProcessor,
    "hist_irr": HistologyIrrProcessor,
    "drug_nerc": DrugNERCProcessor,
    "radiation_modality_irr": ModalityIrrProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "semeval2014-atsc":"classification",
    "biomarkers" : "classification",
    "medstemporality" : "classification",
    'bio-ner': 'tokenclassification',
    'bio_name_ner': 'tokenclassification',
    'bio_name_irr': 'classification',
    "smoking" : "classification",
    "grade_irr" : "classification",
    "mets1" : "classification",
    "mets2" : "classification",
    "siteofmets" : "multilabelclassification",
    "stage_irr" : "classification",
    "surgery_irr": "classification",
    "surgery":"classification",
    "radiation" : "classification",
    "medication": "classification",
    "pcsubtype": "classification",
    "tumor_irr": "classification",
    "alcohol":"classification",
    "menopause_cls": "classification",
    "grade_nerc": "classification",
    "stage_nerc": "classification",
    "performance_status_nerc": "classification",
    "route_nerc": "classification",
    "therapy_ongoing_nerc": "classification",
    "treatmentresponse_ner": "tokenclassification",
    "tumor_ner": "tokenclassification",
    "tumor_nerc": "classification",
    "medication_ner": "tokenclassification",
    "radiation_ner": "tokenclassification",
    "comorbidities_ner": "tokenclassification",
    "comorbidities_irr": "classification",
    "comorbidities_nerc": "classification",
    "oncologyv2": "tokenclassification",
    "site_ner": "tokenclassification",
    "site_nerc": "classification",
    "metastasis_ner": "tokenclassification",
    "metastasis_nerc": "classification",
    "race_ethnicity_ner": "tokenclassification",
    "race_nerc": "classification",
    "ethnicity_nerc": "classification",
    "hist_irr": "classification",
    "drug_nerc": "classification",
    "radiation_modality_irr": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "semeval2014-atsc":3,
    "biomarkers" : 6,
    "medstemporality" : 4,
    "smoking" : 3,
    "grade_irr" : 2,
    "mets1" : 3,
    "mets2" : 2,
    "siteofmets" : 8,
    'stage_irr' : 3,
    "surgery": 2,
    "surgery_irr": 2,
    "radiation_ner" : 5,
    "medication": 2,
    "pcsubtype": 4,
    "tumor_irr": 2,
    "alcohol":3,
    "menopause_cls": 4,
    "grade_nerc": 12,
    "stage_nerc": 151,
    "performance_status_nerc": 18,
    "route_nerc": 8,
    "therapy_ongoing_nerc": 3,
    "treatmentresponse_ner": 3,
    "tumor_ner": 3,
    "tumor_nerc": 85,
    "medication_ner": 13,
    "comorbidities_ner": 3,
    "comorbidities_irr": 2,
    "comorbidities_nerc": 30,
    "oncologyv2": 13,
    "race_ethnicity_ner":5,
    "race_nerc":7,
    "ethnicity_nerc":4,
    "bio-ner": 19,
    "bio_name_ner": 3,
    "bio_name_irr": 2,
    "site_ner": 3,
    "site_nerc": 4,
    "metastasis_ner": 3,
    "metastasis_nerc": 3,
    "hist_irr": 2,
    "drug_nerc": 3,
    "radiation_modality_irr": 2
}
