import json,pylcs
from rouge import Rouge
import numpy as np
from logging import Logger

logger = Logger(__name__)

"""
计算pCLUE任务总分，及子分数
"""

def f1_sim(text_a, text_b):
    """F1相似度
    说明：算出两个文本的最长公共子序列长度，然后乘2并处以两者
    长度之和。推荐用pylcs算，速度较快。
    """
    if not text_a and not text_b:
        return 0.
    else:
        lcs = pylcs.lcs(text_a, text_b)
        return 2. * lcs / (len(text_a) + len(text_b))

def rouge_l_zh(target, pred):
    """计算Rouge-l得分，Rouge-l指标常用于评估自动文本摘要及翻译任务
    target: 真实标签
    pred: 预测标签"""
    if not(isinstance(target, str) or isinstance(pred, str)):
        logger.info("target或pred为非字符串！请检查!")
        return
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(list(pred)), " ".join(list(target)))
        score = scores[0]["rouge-l"]
        return score["f"]

def normalize(text):
    """简单的文本标准化
    """
    return ' '.join(text.lower().split())

def evaluate_pclue_fn(predict_file,target_file):
    """
    计算pclue的成绩
    :param predict_file: 预测文件
    :param target_file:  正确的文件
    :return: 一个dict，包括总分score，以及各个部分的分数（mrc, generate, classify, nli）
    """
    predict_lines=open(predict_file,'r').readlines()
    target_lines=open(target_file,'r').readlines()

    # 1.记录
    classify_list=[]
    mrc_list=[]
    generate_list=[]
    nli_list=[]
    for i, target_line in enumerate(target_lines):
        # e.g. target_line = {"target": "不同"}
        predict_line=predict_lines[i]
        target_answer=json.loads(target_line.replace("，",","))["target"] # 正确的标签
        if isinstance(target_answer, list):  # 将列表转换为字符串，如关键词生成
            target_answer = "，".join(target_answer)
        target_answer=normalize(target_answer)
        predict_answer=json.loads(predict_line)["target"] # 预测的标签
        predict_answer=normalize(predict_answer)
        # print(i,"target_answer:",target_answer,";predict_answer:",predict_answer)

        type=json.loads(target_line.replace("，",","))["type"] # 替换可能存在问题的数据，如有，以便能加载为json
        if type=='classify' or type=='anaphora_resolution': # 分类
            label_temp=True if target_answer==predict_answer else False
            classify_list.append(label_temp)
        elif type=='mrc': # 阅读理解
            em=1 if target_answer==predict_answer else 0
            f1=f1_sim(predict_answer,target_answer)
            mrc_list.append((em, f1))
        elif type=='generate': # 生成
            rouge_l=rouge_l_zh(target_answer, predict_answer)
            generate_list.append(rouge_l)
        elif type=='nli': # 推理
            label_temp = True if target_answer == predict_answer else False
            nli_list.append(label_temp)
        else:
            print("error...predict_line:",predict_line,";target_line:",target_line)
            break # 中断运行
        # if predict_answer==target_answer: count_right=count_right+1
        if i<10: print(i, 'target_answer:',target_answer,";predict_answer:",predict_answer) # 显示部分内容


    # 2.计算最后的得分
    classify_score=np.average(classify_list)
    nli_score=np.average(nli_list)
    generate_score=np.average(generate_list)
    mrc_em_score=np.average([x[0] for x in mrc_list])
    mrc_f1_score=np.average([x[1] for x in mrc_list])
    mrc_score=np.average([mrc_em_score,mrc_f1_score])
    # 计算总分
    score=np.average([classify_score,nli_score,generate_score,mrc_score])
    # 保存分数
    result_dict={"score":score,"classify_score":classify_score,"nli_score":nli_score,"generate_score":generate_score,
                 "mrc_em_score":mrc_em_score,"mrc_f1_score":mrc_f1_score}
    return result_dict

# # 预测的文件，以及正确的文件
# target_file='datasets/pCLUE_test_public_1.json'
# predict_file='datasets/pCLUE_test_public_1.json'
# result=evaluate_pclue_fn(predict_file,target_file)
# print("result:",result)