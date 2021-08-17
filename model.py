from transformers.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn

class ConCasRTE(BertPreTrainedModel):
    def __init__(self, config):
        super(ConCasRTE, self).__init__(config)
        self.bert=BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.w1=nn.Linear(config.hidden_size,config.hidden_size)
        self.w2=nn.Linear(config.hidden_size,config.hidden_size)
        self.w3=nn.Linear(config.hidden_size,config.hidden_size)
        #s
        self.s_classier=nn.Linear(config.hidden_size,2)
        #o
        self.o_classier=nn.Linear(config.hidden_size,2)
        #p
        self.p_classier=nn.Linear(config.hidden_size,config.num_p)
        self.sigmoid=nn.Sigmoid()
        self.init_weights()

    def forward(self, token_ids, mask_token_ids,s2o_loc, entity_loc,batch_so_mask):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        #获取表示
        head,tail,rel,cls=self.get_embed(token_ids, mask_token_ids)
        # 预测s
        s_pred=self.s_pred(head,cls=cls)
        #预测o
        o_pred=self.o_pred(head,tail,s2o_loc=s2o_loc,cls=cls)
        #预测r
        p_pred=self.p_pred(head,tail,rel,entity_loc=entity_loc,batch_so_mask=batch_so_mask,cls=cls)
        return s_pred,o_pred,p_pred

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        head=self.w1(embed)
        tail=self.w2(embed)
        rel=self.w3(embed)
        cls=bert_out[1]
        head=head+tail[:,0,:].unsqueeze(dim=1)
        head, tail,rel,cls=self.dropout(head),self.dropout(tail),self.dropout(rel),self.dropout(cls)
        return head, tail,rel,cls

    def extract_entity(self, input, index=None):
        '''
        取首尾平均
        :param input:[batch,seq,dim]
        :param index: [batch,2]
        :param mask:BL
        :return: [batch,dim]
        '''
        _,_,dim=input.shape
        entity=input.gather(dim=1, index=index.unsqueeze(dim=-1).repeat([1, 1, dim])) #[batch,2,dim]
        entity=entity.mean(dim=1)
        return entity

    def s_pred(self,head,cls):
        s_logist=self.s_classier(head+cls.unsqueeze(dim=1)) #BL,2
        s_pred=self.sigmoid(s_logist)
        return s_pred

    def o_pred(self,head,tail,s2o_loc,cls):
        s_entity=self.extract_entity(head,index=s2o_loc)
        s2o_embed=tail*s_entity.unsqueeze(dim=1) #BLH
        o_logist=self.o_classier(s2o_embed+cls.unsqueeze(dim=1)) #BL2
        o_pred=self.sigmoid(o_logist)
        return o_pred #BL2

    def p_pred(self, head, tail,rel, entity_loc,batch_so_mask,cls):
        [s_loc, o_loc] = entity_loc
        s_entity=self.extract_entity(head,index=s_loc).unsqueeze(dim=1) #B1H
        o_entity=self.extract_entity(tail,index=o_loc).unsqueeze(dim=1) #B1H
        embed=rel*s_entity+o_entity #BLH
        logist=self.p_classier(embed+cls.unsqueeze(dim=1)) #BLR
        batch_so_mask=batch_so_mask.unsqueeze(dim=-1) #BL1
        logist=(logist*batch_so_mask).sum(dim=1)/batch_so_mask.sum(dim=1) #BR/B1b 只对实体取平均
        r_pred=self.sigmoid(logist)
        return r_pred #BR