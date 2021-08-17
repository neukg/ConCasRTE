from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from model import ConCasRTE
from util import *
from tqdm import tqdm
import random
import os
import torch.nn as nn
import torch
from transformers.modeling_bert import BertConfig
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #注意修改

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def judge(ex):
    '''判断样本是否正确'''
    for s,p,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True

class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self,args,train_data, tokenizer,predicate2id,id2predicate):
        super(data_generator,self).__init__(train_data,args.batch_size)
        self.max_len=args.max_len
        self.tokenizer=tokenizer
        self.predicate2id=predicate2id
        self.id2predicate=id2predicate

    def __iter__(self, is_random=True): 
        batch_token_ids, batch_mask = [], []
        batch_subject_labels,batch_s2o_loc, batch_object_labels,batch_subject_ids,batch_object_ids,batch_r ,\
        batch_so_mask = [], [], [],[],[],[],[]
        for is_end, d in self.sample(is_random):
            if judge(d)==False: 
                continue
            token_ids, _ ,mask = self.tokenizer.encode(
                d['text'], max_length=self.max_len
            )
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['triple_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject_labels标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1

                # batch_s2o_loc 和 object_labels :o_pred阶段使用
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                s2o_loc = (start, end) 
                object_labels = np.zeros((len(token_ids), 2))
                if s2o_loc in spoes: 
                    for item in spoes[s2o_loc]:
                        o1,o2,_=item
                        object_labels[o1,0]=1
                        object_labels[o2,1]=1

                #subject_id \ object_id \ r ：p_pred阶段使用
                subject_id=random.choice(list(spoes.keys()))
                os,oe,_=random.choice(spoes[subject_id])
                object_id=(os,oe)
                r=np.zeros(len(self.id2predicate))
                if subject_id in spoes:
                    for o1,o2,the_r in spoes[subject_id]:
                        if o1==object_id[0] and o2==object_id[1]:
                            r[the_r]=1

                # so_mask
                so_mask=np.zeros(len(token_ids))
                so_mask[subject_id[0]:subject_id[1]+1]=1
                so_mask[object_id[0]:object_id[1]+1]=1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_mask.append(mask)

                batch_subject_labels.append(subject_labels)

                batch_s2o_loc.append(s2o_loc) #[B,2]
                batch_object_labels.append(object_labels)

                batch_subject_ids.append(subject_id)
                batch_object_ids.append(object_id)
                batch_r.append(r)
                batch_so_mask.append(so_mask)

                if len(batch_token_ids) == self.batch_size or is_end:   #输出batch
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_mask = sequence_padding(batch_mask)
                    #s_pred使用
                    batch_subject_labels = sequence_padding(batch_subject_labels,dim=0)
                    #o_pred使用
                    batch_s2o_loc = np.array(batch_s2o_loc)
                    batch_object_labels = sequence_padding(batch_object_labels,dim=0)

                    #p_pred使用
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_ids = np.array(batch_object_ids)
                    batch_so_mask=sequence_padding(batch_so_mask).astype(np.int)
                    batch_r = np.array(batch_r)

                    yield [
                        batch_token_ids, batch_mask,
                        batch_subject_labels,batch_s2o_loc, batch_object_labels,
                        batch_subject_ids,batch_object_ids,
                        batch_r,batch_so_mask
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_subject_labels,batch_s2o_loc, batch_object_labels, batch_subject_ids, batch_object_ids, \
                    batch_r,batch_so_mask=[],[],[], [], [], [], []


class CE():
    def __call__(self,args,targets, pred, from_logist=False):
        '''
        计算二分类交叉熵
        :param targets: [batch,seq,2]
        :param pred: [batch,seq,2]
        :param from_logist:是否没有经过softmax/sigmoid
        :return: loss.shape==targets.shape==pred.shape
        '''
        c = torch.tensor(args.conf_value).to("cuda")  # 固定conf_value
        if not from_logist:
            '''返回到没有经过softmax/sigmoid得张量'''
            # 截取pred，防止趋近于0或1,保持在[min_num,1-min_num]
            pred = torch.where(pred < 1 - args.min_num, pred, torch.ones(pred.shape).to("cuda") * 1 - args.min_num).to("cuda")
            pred = torch.where(pred > args.min_num, pred, torch.ones(pred.shape).to("cuda") * args.min_num).to("cuda")
            pred = torch.log(pred / (1 - pred))
        # 利用公式loss=max(x, 0) - x * z + log(1 + exp(-abs(x))),参考tf.nn.sigmoid_cross_entropy_with_logist源码
        relu = nn.ReLU()
        # 计算传统的交叉熵loss
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        if args.better_CE:
            # 重新获得概率
            sigmoid = nn.Sigmoid()
            pred = sigmoid(pred)
            # 是否预测正确
            pred_res_1 = (((pred - 0.5) * (targets - 0.5)) > 0)
            # 是否预测结果自信
            pred_res_2 = torch.abs(pred - 0.5).to("cuda") > c
            # 设置loss权重矩阵
            # 预测错误的权重为1
            shape = loss.shape
            weight = torch.ones(shape).to("cuda")
            # 预测正确且自信，权重为0
            weight = torch.where((pred_res_1 == True) & (pred_res_2 == True), torch.zeros(shape).to("cuda"), weight).to("cuda")
            loss = loss * weight
        return loss

def train(args):
    output_path=os.path.join(args.base_path,args.dataset,"output")
    train_path=os.path.join(args.base_path,args.dataset,"train.json")
    dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_pred_path=os.path.join(output_path,"test_pred.json")
    dev_pred_path=os.path.join(output_path,"dev_pred.json")
    log_path=os.path.join(output_path,"log.txt")

    # 加载数据集
    train_data = json.load(open(train_path))
    valid_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))

    tokenizer = Tokenizer(args.bert_vocab_path)  # 注意修改
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)
    train_model = ConCasRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataloader = data_generator(args,train_data, tokenizer,predicate2id,id2predicate)

    t_total = len(dataloader) * args.num_train_epochs

    """ 优化器准备 """
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1 = -1.0  # 全局的best_f1
    step = 0
    binary_crossentropy=CE()
    no_change=0
    for epoch in range(args.num_train_epochs):
        if no_change>=args.stop_epoch:#如果stop_epoch轮，分数没有上涨的话，就停止训练
            break
        train_model.train()
        epoch_loss = 0
        with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(dataloader):
                batch = [torch.tensor(d).to("cuda") for d in batch]
                batch_token_ids, batch_mask,batch_subject_labels,batch_s2o_loc, batch_object_labels,batch_subject_ids,\
                batch_object_ids,batch_r,batch_so_mask= batch

                s_pred,o_pred,p_pred = train_model(batch_token_ids, batch_mask,batch_s2o_loc,
                                                   [batch_subject_ids,batch_object_ids],batch_so_mask)

                #计算损失
                s_loss = binary_crossentropy(args,targets=batch_subject_labels, pred=s_pred) #BL2
                s_loss=torch.mean(s_loss,dim=2).to("cuda") #BL
                s_loss=torch.sum(s_loss*batch_mask).to("cuda")/torch.sum(batch_mask).to("cuda")

                o_loss = binary_crossentropy(args,targets=batch_object_labels, pred=o_pred)
                o_loss=torch.mean(o_loss,dim=2).to("cuda")
                o_loss=torch.sum(o_loss*batch_mask).to("cuda")/torch.sum(batch_mask).to("cuda")

                r_loss=binary_crossentropy(args,targets=batch_r,pred=p_pred)
                r_loss=r_loss.mean()

                loss=s_loss+o_loss+r_loss

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)
        f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model,valid_data,dev_pred_path)

        if f1 > best_f1:
            no_change=0
            # Save model checkpoint
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, "dev_%s"%(WEIGHTS_NAME)))  # 保存最优模型权重
        else:
            no_change+=1

        epoch_loss = epoch_loss / dataloader.__len__()
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f\tno_change:%d" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1,no_change), file=f)

    #对test集合进行预测
    #加载训练好的权重
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model, test_data, test_pred_path)
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)

def extract_spoes(args,tokenizer,id2predicate,model,text,entity_start=0.5,entity_end=0.5,p_num=0.5):
    """抽取输入text所包含的三元组
    """
    #sigmoid=nn.Sigmoid()
    if isinstance(model,torch.nn.DataParallel):
        model=model.module
    model.to("cuda")
    tokens = tokenizer.tokenize(text, max_length=args.max_len)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, _ ,mask = tokenizer.encode(text, max_length=args.max_len)
    #获取BERT表示
    model.eval()
    with torch.no_grad():
        head,tail,rel,cls = model.get_embed(torch.tensor([token_ids]).to("cuda"), torch.tensor([mask]).to("cuda"))
        head = head.cpu().detach().numpy() #[1,L,H]
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()

    def get_entity(entity_pred):
        start = np.where(entity_pred[0, :, 0] > entity_start)[0]
        end = np.where(entity_pred[0, :, 1] > entity_end)[0]
        entity = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity

    #抽取subject
    model.eval()
    with torch.no_grad():
        subject_preds = model.s_pred(torch.tensor(head).to("cuda"),torch.tensor(cls).to("cuda"))
        subject_preds = subject_preds.cpu().detach().numpy() #[1,L,2]

        subject_preds[:,0,:],subject_preds[:,-1,:]=0.0,0.0

    subjects=get_entity(subject_preds)

    #获得s_loc,o_loc
    s_loc,o_loc =[], []
    # 生成batch_so_mask,
    batch_so_mask = []
    for s in subjects:
        #s:(start,end)
        s_mask=np.zeros(len(token_ids))
        s_mask[s[0]:s[1] + 1] = 1.0
        model.eval()
        with torch.no_grad():
            o_pred=model.o_pred(torch.tensor(head).to("cuda"),torch.tensor(tail).to("cuda"),torch.tensor([s]).to("cuda"),cls=torch.tensor(cls).to("cuda"))
            o_pred = o_pred.cpu().detach().numpy()  # [1,L,2]
            o_pred[:, 0, :], o_pred[:, -1, :] = 0.0, 0.0
        objects = get_entity(o_pred)
        if objects:
            for o in objects:
                s_loc.append(s)
                o_loc.append(o)
                batch_so_mask.append(np.zeros(len(token_ids)))
                batch_so_mask[-1][s[0]:s[1] + 1] = 1.0
                batch_so_mask[-1][o[0]:o[1] + 1] = 1.0

    if s_loc and o_loc:
        spoes = []
        head=np.repeat(head,len(s_loc),0)
        tail=np.repeat(tail,len(s_loc),0)
        rel=np.repeat(rel,len(s_loc),0)
        cls=np.repeat(cls,len(s_loc),0)

        batch_so_mask=np.array(batch_so_mask).astype(np.int)

        # 传入subject，抽取object和predicate
        model.eval()
        with torch.no_grad():
            p_pred = model.p_pred(head=torch.tensor(head).to("cuda"),
                                  tail=torch.tensor(tail).to("cuda"),
                                  rel=torch.tensor(rel).to("cuda"),
                                  entity_loc=[torch.tensor(s_loc).to("cuda").long(),
                                              torch.tensor(o_loc).to("cuda").long()],
                                  batch_so_mask=torch.tensor(batch_so_mask).to("cuda"),
                                  cls=torch.tensor(cls).to("cuda")
                                  )
            p_pred = p_pred.cpu().detach().numpy() #BR

        index,p_index=np.where(p_pred>p_num)
        for i,p in zip(index,p_index):
            subject=s_loc[i]
            object=o_loc[i]
            spoes.append(
                (
                 (mapping[subject[0]][0],mapping[subject[1]][-1]),
                  p,
                 (mapping[object[0]][0], mapping[object[1]][-1])
                )
            )

        return [(text[s[0]:s[1] + 1], id2predicate[str(p)], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []

def evaluate(args,tokenizer,id2predicate,model,evl_data,evl_path):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    for d in evl_data:
        R = set(extract_spoes(args,tokenizer,id2predicate,model,d['text']))
        T = set([(i[0],i[1],i[2]) for i in d['triple_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'triple_list': list(T),
            'triple_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },ensure_ascii=False,indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall

def test(args):
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    output_path=os.path.join(args.base_path,args.dataset,"output")
    test_pred_path = os.path.join(output_path, "test_pred.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    config = BertConfig.from_pretrained(args.bert_config_path)
    tokenizer = Tokenizer(args.bert_vocab_path)
    config.num_p=len(id2predicate)
    train_model = ConCasRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model, test_data, test_pred_path)
    print("f1:%f, precision:%f, recall:%f"%(f1, precision, recall))