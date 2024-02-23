from __future__ import annotations
import locale


def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


# import spacy
# English pipelines include a rule-based lemmatizer
# nlp = spacy.load("en_core_web_sm")
import simplejson as JSON
import urllib.request
from pathlib import Path
from typing import Iterable
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask import Response, jsonify
from flask_cors import CORS, cross_origin
import requests
from abc import ABCMeta
from transformers_interpret import SequenceClassificationExplainer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,

)
from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm
import numpy as np
from time import strftime
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

app = Flask(__name__)
CORS(app, support_credentials=True)

import os, sys, shutil, locale, openai, re, random, pymongo, time, torch, json


class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            logging.info("Creating the singleton instance")
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class custom_tablellama_persona(metaclass=SingletonABCMeta):
    __metaclass__ = SingletonABCMeta

    def __init__(self, model_name) -> None:
        if model_name is None:
            pass
        else:
            # model_name = "NousResearch/Llama-2-7b-chat-hf"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=True,
                return_dict=True, torch_dtype=torch.float16)
            self.output_dir = "customer360_tablellama_copy"
            self.model = PeftModel.from_pretrained(self.model, self.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_eos_token = True
            self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=50)


def get_collection():
    connection_string = "mongodb://admin_personas:admin123@34.41.129.234:27017"
    client = pymongo.MongoClient(connection_string)
    db = client.db_personas
    collection = db.coll_payments_data

    docs = collection.find({'Store': 'Bank of America', "clean_data": {'$exists': True}})
    DATA = [item for item in docs]
    return DATA, collection


@app.route("/get_transaction_data", methods=["GET", "POST"])
@cross_origin(supports_credentials=True)
def get_transaction_data():
    if request.method == "GET":
        data = get_collection()
        for item in data:
            del item['_id']
        js = [{'data': data}]
    return Response(json.dumps(js), mimetype='application/json')


instruction = "This is a table classification task. The goal of this task is to identify the persona of the person based on the transaction history table of the person. "

question = """The transaction history of the person can be classified as one of the following 10 personas. Classify each person's financial statement as belonging to as one of the 10 personas list below:
'College Student',
 'Corporate Executive',
 'Freelancer',
 'High Net Worth Individual',
 'Immigrant Family',
 'Local Business Owner',
 'Military Personnel',
 'Non-profit Director',
 'Retired Couple',
 'Small Business Owner',
 'Working Mom'
"""


def generate_test_prompt(instruction, inputs, question):
    return f"""Below is a table attached, of a bank transaction data for a particular individual. Try to analyse it.
            ### Instruction:
            {instruction}

            ### Input:
            {inputs}

            ### Question:
            {question}

            ### Response = """.strip()


def process_table(doc):
    columns = ['Date', 'Description', 'Transaction Type', 'Amount(USD)', 'Balance']
    input_table = "[TLE] This table is a transaction data of a person. [TAB] col: Date | Description | Transaction Type | Amount(USD)| Balance"
    cnt = 0
    for txn in doc['clean_data']:
        if cnt < 150:
            temp_row = ' [SEP] row {}: '.format(cnt)
            for col in columns:
                temp_row += str(txn[col]) + '|'
            input_table += temp_row
            cnt += 1
        else:
            break
    return input_table


"""    
    cnt = 0
    columns = ['Date', 'Description', 'Transaction Type', 'Amount(USD)', 'Balance']
    input_table = "[TLE] This table is a transaction data of a person. [TAB] col: Date | Description | Transaction Type | Amount(USD)| Balance"
    for txn in doc['clean_data']:
        temp_row = '[SEP] row {}: '.format(cnt)
        for col in columns:
            temp_row += str(txn[col]) + '|'
        cnt += 1
        input_table += temp_row
    return input_table

"""


def get_record_id(idx):
    myquery = {"id": idx}
    connection_string = "mongodb://admin_personas:admin123@34.41.129.234:27017"
    client = pymongo.MongoClient(connection_string)
    db = client.db_personas
    collection = db.coll_payments_data
    doc = list(collection.find(myquery))
    return doc


def get_persona(payload):
    connection_string = "mongodb://admin_personas:admin123@34.41.129.234:27017"
    client = pymongo.MongoClient(connection_string)
    db = client.db_personas
    collection_persona = db.coll_personas
    myq = {'name': 'Bank of America'}
    d = list(collection_persona.find(myq))
    persona_description = {}
    for item in d[0]['personas']:
        persona_description[list(item.items())[0][0]] = list(item.items())[0][1]

    personas_upper = ['College Student', 'Corporate Executive', 'Freelancer', 'High Net Worth Individual',
                      'Immigrant Family',
                      'Military Personnel', 'Non-profit Director', 'Retired Couple', 'Small Business Owner',
                      'Working Mom']
    personas_dict = {}
    personas_lower = []
    for persona in personas_upper:
        personas_dict[persona.lower()] = persona
    for item in personas_upper:
        personas_lower.append(item.lower())
    tablellama_instance = custom_tablellama_persona()
    idx = payload['id']
    doc = get_record_id(idx)
    input_table = process_table(doc[0])
    llm_input = generate_test_prompt(instruction, input_table, question)
    res = tablellama_instance.pipe(llm_input)
    pred_persona = res[0]['generated_text'].split("###")[-1]
    print(pred_persona)
    found = False
    response = None
    desc = None
    for item in personas_upper:
        if item.lower() in pred_persona.lower():
            print(item)
            found = True
            response = item
            desc = persona_description[response]
        if found:
            break
    if not found:
        response = personas_upper[0]
        desc = persona_description[response]
    js = [{'persona': response, 'description': desc}]
    return js


@app.route("/persona_predictor", methods=["POST"])
@cross_origin(supports_credentials=True)
def persona_predictor():
    if request.method == "POST":
        payload = request.json
        print("Payload: ", payload)
        js = get_persona(payload)
        print(js)
    return Response(json.dumps(js), mimetype='application/json')


if __name__ == '__main__':
    custom_tablellama_persona(model_name='osunlp/TableLlama')
    app.run(host='0.0.0.0', port=8888, debug=False)
