from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import os
from newsplease import NewsPlease
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import threading

load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6, max_tokens=4096, api_key = "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")
embeddings = OpenAIEmbeddings()
basepath = os.path.abspath(os.curdir)
text_splitter = RecursiveCharacterTextSplitter()
output_parser = StrOutputParser()
load_dotenv()
basepath = os.path.abspath(os.curdir)
cred = credentials.Certificate(basepath + '\\serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
# Imports End here



# Chain creation
def read_prompt(chain_name,basepath=basepath):
    with open(basepath+'\\prompts\\'+chain_name+'-prompt.txt',encoding='utf-8') as fh:
        return fh.read()

def create_chain(chain_name,sample):
    if sample==False:
        prompt = ChatPromptTemplate.from_messages([("system", read_prompt(chain_name)),("user", "{input}")])
        final_chain =  prompt | llm | output_parser    
        return final_chain
    elif sample==True:
        prompt = ChatPromptTemplate.from_messages([("system", read_prompt(chain_name)),("user", "{input}")])
        if  chain_name[:4] == 'logi':
            loader = UnstructuredExcelLoader(basepath+'\\Training\\'+chain_name+"-train.xlsx")
        elif  chain_name[:4] == 'stat':
            loader = TextLoader(basepath+'\\Training\\'+chain_name+"-train.txt")
        elif  chain_name[:4] == 'fact':
            loader = TextLoader(basepath+'\\Training\\'+chain_name+"-train.txt")
        data = loader.load()
        documents = text_splitter.split_documents(data)
        vector = FAISS.from_documents(documents, embeddings)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        return create_retrieval_chain(retriever, document_chain)

# Core Codes
def get_articles(urls):
    '''Method to scrape articles using URLs'''
    articles = []
    for url in urls:
        raw_article = NewsPlease.from_url(url,timeout=5)
        if hasattr(raw_article, 'maintext'):
            if raw_article.maintext==None:
                if not requests.get(url).ok:
                    print("Cannot scrape URL "+str((urls.index(url))+1)+" due to copyright issues.")
                    continue
            if raw_article.maintext==None:
                print("Cannot scrape URL "+str((urls.index(url))+1)+" due to copyright issues.")
                continue
            if len(raw_article.maintext)>60000:
                print("Article in URL "+str((urls.index(url))+1)+" is too long.")
                continue
            articles.append(raw_article.maintext)
        else:
            print("Cannot scrape URL "+str((urls.index(url))+1)+" due to copyright issues.")
    print("\nScraped "+str(len(articles))+" out of "+str(len(urls))+" URLs.\n\nProcessing scraped articles...")
    return articles



def init_all_chains():
    chains = {}
    chain_names = [x[:8] for x in os.listdir(basepath+'\\'+'Prompts')]
    for chain_name in chain_names:
        if chain_name[:4] == "logi":
            continue
        if chain_name not in [x[:8] for x in os.listdir(basepath+'\\'+'Training')]:
            chains[chain_name] = create_chain(chain_name,sample=False)
            print(chain_name,' Success without context')
            continue
        chains[chain_name] = create_chain(chain_name,sample=True)
        print(chain_name,' Success with context')
    print('completed initializing all chains')
    return chains

#old
# def init_all_chains():
#     chains = {}
#     chain_names = [x[:8] for x in os.listdir(basepath+'\\'+'Prompts')]
#     for chain_name in chain_names:
#         if chain_name in ['logi-con','logi-par']:
#             chains[chain_name] = create_chain(chain_name,sample=False)
#             print(chain_name,' Success without context')
#             continue
#         chains[chain_name] = create_chain(chain_name,sample=True)
#         print(chain_name,' Success with context')
#     print('completed initializing all chains')
#     return chains





chains = init_all_chains()
def invoke(chain,input,chains=chains):
    response =  chains[chain].invoke({"input": input})
    print(chain," ran successfully")
    return response['answer']

def invoke_no_sample(chain,input,chains=chains):
    response =  chains[chain].invoke({"input": input})
    print(chain," ran successfully")
    return response

def preprocess_articles_logi(articles):
    valid_paragraphs = []
    rel = []
    par = []
    for article in articles:  
        relevant_text = invoke_no_sample('logi-con', article)
        rel.append(relevant_text)
        paraphrased_text = invoke_no_sample('logi-par',relevant_text)
        par.append(paraphrased_text)
        paragraph_list = [pl for pl in paraphrased_text.split("\n\n") if len(pl)>100]
        val_responses = []
        for paragraph in paragraph_list:
            val_response = invoke('logi-val',paragraph)
            val_responses.append(val_response)
            val_response_list = val_response.split("\n\n")
            validity = val_response_list[-1]
            if ("invalid" or "not valid") in validity.lower():
                decision = "Invalid"
            elif "valid" in validity.lower():
                decision = "Valid"
            else:
                decision = "Unclear"
            if decision == "Valid":
                valid_paragraphs.append(paragraph)
    return valid_paragraphs



# def logical_chain(articles, resultant_list):
#     valid_paragraphs = preprocess_articles_logi(articles)
#     ques_responses = []
#     opt_responses = []
#     pre_improve = []
#     improved_responses = []
#     final_mcqs = []
#     for p in valid_paragraphs:
#         #Question
#         ques_response = invoke('logi-que', p)
#         ques_responses.append(ques_response)
#         ques_response_split = ques_response.split("\n\n")
#         for qrs in ques_response_split:
#             if qrs[:6].lower()!="source" and qrs[:8].lower()!="question":
#                 ques_response_split.remove(qrs)
#         source_text = ques_response_split[0]
#         final_ques = "\n\n".join(ques_response_split)
#         #Options
#         opt_response = invoke('logi-opt',final_ques)
#         opt_responses.append(opt_response)
#         opt_response_split = opt_response.split("\n\n")
#         if opt_response_split[0][:6].lower()=="source":
#             final_opt = opt_response_split[0] + "\n\n" + opt_response_split[2]
#         else:
#             final_opt = source_text + "\n\n" + opt_response_split[1]
#         pre_improve.append(final_opt)
#         #Improvement
#         mcq_response = invoke('logi-imp', final_opt)
#         improved_responses.append(mcq_response)
#         mcq_response_list = mcq_response.split("\n\n")
#         for mrl in mcq_response_list:
#             if mrl[:12].lower()=="improved mcq":
#                 final_mcq = mrl[14:]
#                 final_mcqs.append(final_mcq)
#     resultant_list.extend(final_mcqs)
#     return final_mcqs

# def stat_chain(articles, resultant_list):
#     inter_mcqs = []
#     final_mcqs = []
#     for article in articles:
#         extracted_data = invoke('stat-ext', article)
#         mcqs = invoke('stat-gen', extracted_data)
#         mcqs=mcqs.split("\n\n")
#         for mcq in mcqs:
#             inter_mcqs.append(mcq)
#         for im in inter_mcqs:
#             final_mcq = invoke('stat-imp', im)
#             final_mcq_split = final_mcq.split("\n\n")
#             final_mcq = final_mcq_split[1][final_mcq_split[1].index("\n"):]
#             final_mcqs.append(final_mcq)
#     resultant_list.extend(final_mcqs)
#     return resultant_list

def stat_chain(articles, resultant_list):
    final_mcqs = []
    for article in articles:
        extracted_data = invoke('stat-ext', article)
        mcqs = invoke('stat-gen', extracted_data)
        mcqs=mcqs.split("\n\n")
        for m in mcqs:
            final_mcq = invoke_no_sample('stat-imp', m)
            final_mcq_split = final_mcq.split("\n\n")
            final_mcq = final_mcq_split[1][final_mcq_split[1].index("\n"):]
            final_mcqs.append(final_mcq)
    resultant_list.extend(final_mcqs)
    print("Stat success")
    return 1

def fact_chain(articles, resultant_list):
    mcqs = []
    for article in articles:
        extracted_list = []
        article_mcqs = []
        extracted_data = invoke_no_sample("fact-ext", article)
        extracted_data = extracted_data.split("\n\n")
        if extracted_data[0][:9].lower() == "extracted":
            extracted_list.append(extracted_data[0][15:])
        for e in extracted_list:
            generation_response = invoke_no_sample("fact-gen", e)
            article_mcqs.append(generation_response)
            mcq=generation_response.split("\n\n")
            mcq.remove(mcq[-1])
            mcqs.extend(mcq)
    resultant_list.extend(mcqs)
    print("Fact success")
    return 1

def add_desc(resultant_list):
    for r in range(len(resultant_list)):
        desc = invoke_no_sample('desc-gen', resultant_list[r])
        resultant_list[r] += "\n\nAdditional Information:\n" + desc
    return 1

def convert_to_dict(mcq_content):
    fields = mcq_content.split('\n')
    data = {}
    for field in fields:
        row = field.split(": ")
        data[row[0]] = row[1]
    return data

def run_threads(articles):
    resultant_list = []
    # logi_thread = threading.Thread(target=logical_chain, args = (articles, resultant_list))
    # stat_thread = threading.Thread(target=stat_chain, args = (articles, resultant_list))
    fact_thread = threading.Thread(target=fact_chain, args = (articles, resultant_list))
    # logi_thread.start()
    # stat_thread.start()
    fact_thread.start()
    # logi_thread.join()
    # stat_thread.join()
    fact_thread.join()
    return resultant_list

# Flask code starts here
app = Flask(__name__)
CORS(app)



@app.route('/quiz-generator', methods=['POST'])
def process_urls():
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'Invalid input. Provide an array of URLs in JSON format.'}), 400
    urls = data['urls']
    print(urls)
    articles = get_articles(urls)
    resultant_list = run_threads(articles)
    add_desc(resultant_list)


    results = {'url': 'https://www.example.com', 'mcqs': resultant_list}
    return jsonify(results), 200


@app.route('/submit', methods=['POST'])
def submit_mcq():
    data = request.json
    mcq_content = data.get('mcq', '')
    print(mcq_content)
    if mcq_content:
        # Create a new document in Firestore with the MCQ content
        mcq_ref = db.collection('ClimaQuiz').document()
        mcq_ref.set(convert_to_dict(mcq_content))

        return jsonify({'success': True, 'message': 'MCQ submitted successfully'}), 200
    else:
        return jsonify({'success': False, 'message': 'MCQ content is empty'}), 400



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

