import logging
import re
from allennlp_models.pretrained import load_predictor
from nltk import word_tokenize

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

TAGS = ['$R-ARGM-ADV', '$R-ARGM-CAU', '$C-ARG2', '$ARGM-LVB', '$ARGM-COM', '$ARGA', '$ARGM-ADJ', '$ARGM-PNC',
        '$ARG0', '$ARGM-MNR', '$C-ARGM-LOC', '$C-ARGM-MNR', '$C-ARG4', '$ARGM-DIR', '$ARGM-EXT', '$R-ARG1',
        '$R-ARGM-EXT', '$ARGM-LOC', '$R-ARGM-MNR', '$ARGM-PRR', '$C-ARG1', '$ARGM-ADV', '$ARGM-MOD', '$ARGM-REC',
        '$R-ARG2', '$C-ARGM-EXT', '$ARGM-PRP', '$ARGM-DIS', '$ARG3', '$ARGM-TMP', '$R-ARG3', '$ARGM-GOL', '$R-ARG0',
        '$C-ARGM-ADV', '$ARG1', '$ARGM-CAU', '$C-ARG0', '$V', '$ARG4', '$R-ARGM-TMP', '$ARGM-PRD', '$ARG5', '$ARG2',
        '$C-ARGM-TMP', '$ARGM-NEG', '$R-ARGM-DIR', '$R-ARGM-LOC', '$R-ARGM-GOL']


class E2EQGPipeline:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, use_cuda: bool, baseline):
        self.srl = list()
        self.model = model
        self.tokenizer = tokenizer
        self.baseline = baseline

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

        self.default_generate_kwargs = {
            "max_length": 512,
            "num_beams": 10,
            "length_penalty": 1,
            # "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, sentence: str, **generate_kwargs):
        if self.baseline:
            inputs = self._prepare_inputs_for_baseline(sentence)

            if not generate_kwargs:
                generate_kwargs = self.default_generate_kwargs

            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                **generate_kwargs
            )

            prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)

            return prediction

        srl = list()
        sentence = self._handle(sentence)
        predictor = load_predictor("structured-prediction-srl-bert")
        preds = predictor.predict(sentence)
        return_str=""

        return_str="sentence: " + sentence+"\n"
        return_str=return_str+"------------------- \n"
        

        if len(preds["verbs"]) == 0:
            srl.append((sentence, sentence, dict()))
            return_str=return_str+"This sentence has no SRL representation. \n"
        else:
            return_str=return_str+"representations: \n "
            for verb in preds["verbs"]:
                labels = dict()
                rep = verb['description']
                return_str=return_str+rep+"\n-------------------\n"
                
                brackets = re.findall('\[(.*?)\]', rep)
                for bracket in brackets:
                    if not bracket.__contains__(': ') or bracket.__contains__(' : '):
                        continue

                    key, value = bracket.split(": ", 1)
                    labels[key] = value

                    if key in ['V', 'ARGM-MOD']:
                        rep = rep.replace('[{}]'.format(bracket), value)
                    else:
                        rep = rep.replace('[{}]'.format(bracket), "$" + key)

                srl.append((sentence, rep, labels))

        questions = set()
        if len(srl) <= 1:
            (sentence, rep, labels) = srl[0]
            inputs = self._prepare_inputs_for_one(sentence, rep)

            if not generate_kwargs:
                generate_kwargs = self.default_generate_kwargs

            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                **generate_kwargs
            )

            prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
            prediction = prediction.replace("?", " ?")
            for i in TAGS:
                if not prediction.__contains__(i):
                    continue
                prediction = prediction.replace(f"{i}", f" {i} ")

            for key, value in labels.items():
                if prediction.__contains__("$" + key):
                    prediction = prediction.replace("$" + key, value)
            prediction = re.sub(' +', ' ', prediction)
            questions.add(prediction)
        else:
            sentences = list()
            reps = list()
            for s in srl:
                (sentence, rep, labels) = s
                sentences.append(sentence)
                reps.append(rep)
            inputs = self._prepare_inputs_for_more(sentences, reps)

            if not generate_kwargs:
                generate_kwargs = self.default_generate_kwargs

            outs = self.model.generate(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                **generate_kwargs
            )

            #print("predictions: ")
            return_str=return_str+"\n predictions: \n"
            predictions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            for p, s in zip(predictions, srl):
                (sentence, rep, labels) = s
                gen = p.replace("?", " ?")
                for i in TAGS:
                    if not gen.__contains__(i):
                        continue
                    gen = gen.replace(f"{i}", f" {i} ")
                #print(gen)
                return_str=return_str+"\n"+gen+"\n -------------------\n"
                
                for key, value in labels.items():
                    if gen.__contains__("$" + key):
                        gen = gen.replace("$" + key, value)
                gen = re.sub(' +', ' ', gen)
                questions.add(gen)
        print(return_str)
        return questions,return_str

    def _prepare_inputs_for_more(self, context, rep):
        source_text = list()
        for c, r in zip(context, rep):
            source_text.append(f"question: {r} context: {c}")

        if self.model_type == "t5":
            source_text = [i + " </s>" for i in source_text]

        inputs = self._tokenize(source_text)
        return inputs

    def _prepare_inputs_for_one(self, context, rep):
        source_text = f"question: {rep} context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _prepare_inputs_for_baseline(self, context):
        source_text = f"question: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs

    def _handle(self, text):
        text = text.strip()
        text = re.sub(r'\n+', '.', text)
        text = re.sub(r'\[\d+\]', ' ', text)
        return ' '.join(word_tokenize(text)).strip()


def pipeline(model_name,alpha):
    model = "./"+model_name+"_"+alpha+"C/"
    tokenizer = "./"+model_name+"_"+alpha+"C/"
    use_cuda = True
    if(model_name=="base_bart_2"or model_name=="base_t5_7"):
        baseline=True
        model="./"+model_name+"/"
        tokenizer="./"+model_name+"/"
    else:
        baseline=False
    print(model,tokenizer)
        
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    return E2EQGPipeline(model=model, tokenizer=tokenizer, use_cuda=use_cuda, baseline=baseline)
