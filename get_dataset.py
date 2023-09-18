import pickle
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def get_chat(filepath):
    chatlog=pickle.load(open(filepath,'rb'))
    chat=chatlog['convo_dict']
    return chat

def get_positive(chat):
    query_list = []
    question_list = []
    for conversation_idx, conversation in enumerate(chat.values()):
        if conversation_idx == 0:
            continue
        for turn_idx, turn in enumerate(conversation):
            if 'I am happy to help you with:' in turn:
                qry=conversation[turn_idx-1]
                qn=turn.split('\n')[1:6]
                try:
                    response=int(conversation[turn_idx+1])
                    if response not in range(1,6):
                        response=0
                except IndexError:
                    response=0
                except ValueError:
                    response=0
                if response!=0:
    
                    # pair=pd.DataFrame({'query':[qry.lower()], 'question':[(qn[response][4:]).lower()]}) 
                    if len(qn) > response:
                        query_list.append(qry.lower())
                        question_list.append((qn[response][4:]).lower())
                    # df_list.append(pair)
                    # dataframe=pd.concat([dataframe,pair],ignore_index=True,axis=0)

    # dataframe=pd.concat(df_list,ignore_index=True,axis=0)
    dataframe=pd.DataFrame({'query':query_list, 'question':question_list})
    dataframe=dataframe.drop_duplicates()
    dataframe['count'] = dataframe.groupby(['query','question'])['question'].transform('count')
    return dataframe

def get_negative(chat):
    n_query_list=[]
    n_question_list=[]
    for conversation_idx, conversation in enumerate(chat.values()):
        if conversation_idx == 0:
            continue
        for turn_idx, turn in enumerate(conversation):
            if 'I am happy to help you with:' in turn:
                qry=conversation[turn_idx-1]
                qns=turn.split('\n')[1:6]
                try:
                    response=int(conversation[turn_idx+1])
                except IndexError:
                    response=0
                except ValueError:
                    response=0
                if response==0:
                    for qn in qns:
                        if qn!='' and 'me again??' not in qn:
                            n_query_list.append(qry.lower())
                            n_question_list.append((qn[4:]).lower())
                        #pair=pd.DataFrame({'query':[qry.lower()], 'question':[(qn[4:]).lower()],'count':0}) 
                        #negative_list.append(pair)
    #negative=pd.concat(negative_list,ignore_index=True,axis=0)
    negative=pd.DataFrame({'query':n_query_list, 'question':n_question_list, 'count':0})
    #negative['question'].replace('', np.nan, inplace=True)
    #negative.dropna(subset=['question'], inplace=True)
    #negative=negative[negative["question"].str.contains("me again??")==False]
    negative=negative.drop_duplicates()
    multiple=negative[negative["query"].str.contains(' ')==True]
    return multiple

def get_dataset(filepath):
    chat=get_chat(filepath)
    positives=get_positive(chat)
    negatives=get_negative(chat)
    dfm=pd.concat([positives,negatives])
    dfm = dfm.drop_duplicates(subset=['query','question'], keep="first")
    return dfm

if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('savepath')
    args = parser.parse_args()

    print(args.filepath)
    print(get_dataset(args.filepath).head())

    get_dataset(args.filepath).to_csv(Path(args.savepath),index=False)
    print('Dataframe saved as csv.')



