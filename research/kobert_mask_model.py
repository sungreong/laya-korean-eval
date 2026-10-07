"""KoBERT encoder with LAYA-like [MASK] marker readout and shared scorer."""
from pathlib import Path
import json, random
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class KoBertMaskDecision(nn.Module):
    def __init__(self, model_dir, head_layers=2, dropout=.1):
        super().__init__()
        self.model_dir=str(model_dir)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir,use_fast=False,local_files_only=True)
        self.encoder=AutoModel.from_pretrained(self.model_dir,local_files_only=True)
        width=self.encoder.config.hidden_size
        if head_layers:
            layer=nn.TransformerEncoderLayer(width,self.encoder.config.num_attention_heads,4*width,dropout,
                                             batch_first=True,norm_first=True,activation='gelu')
            self.head=nn.TransformerEncoder(layer,head_layers)
        else:
            self.head=nn.Identity()
        self.scorer=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.head_layers=head_layers

    def forward(self,input_ids,attention_mask,token_type_ids,marker_positions,marker_mask,freeze_encoder=True):
        if freeze_encoder:
            with torch.no_grad():
                hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                                    token_type_ids=token_type_ids).last_hidden_state
            hidden=hidden.detach()
        else:
            hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                                token_type_ids=token_type_ids).last_hidden_state
        if self.head_layers:
            hidden=self.head(hidden,src_key_padding_mask=~attention_mask.bool())
        index=marker_positions.clamp(min=0)[:,:,None].expand(-1,-1,hidden.size(-1))
        marker_hidden=torch.gather(hidden,1,index)
        logits=self.scorer(marker_hidden).squeeze(-1).float()
        return logits.masked_fill(~marker_mask,-1e4)

    def trainable_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('encoder.')}


def pack_one(tokenizer,state,instruction,criteria,max_length=512,order=None,state_reserve=128):
    keys=list(criteria)
    if order is not None: keys=[keys[i] for i in order]
    ids=[tokenizer.cls_token_id]+tokenizer.encode(instruction,add_special_tokens=False)+[tokenizer.sep_token_id]
    types=[0]*len(ids); markers=[]
    state_ids=tokenizer.encode(state,add_special_tokens=False)
    # KoBERT has a hard 512-token limit. Allocate every option the same budget
    # while reserving room for the complaint text. This prevents early options
    # from crowding later candidates out of a 27-way question.
    reserved_state=min(state_reserve,len(state_ids))
    available=max_length-len(ids)-1-reserved_state-2*len(keys)
    option_budget=min(32,available//len(keys))
    if option_budget<1:
        raise ValueError(f'{len(keys)} option markers leave no usable option/state budget within {max_length}')
    option_tokens_dropped=0
    for key in keys:
        encoded=tokenizer.encode(criteria[key],add_special_tokens=False)
        option_tokens_dropped+=max(0,len(encoded)-option_budget)
        markers.append(len(ids)); option=[tokenizer.mask_token_id]+encoded[:option_budget]+[tokenizer.sep_token_id]
        ids+=option; types += [1]*len(option)
    room=max_length-len(ids)-1
    if room<1: raise ValueError(f'Question/options use {len(ids)} tokens and leave no state room within {max_length}')
    dropped=max(0,len(state_ids)-room); state_ids=state_ids[:room]
    ids+=state_ids+[tokenizer.sep_token_id];types += [0]*(len(state_ids)+1)
    assert all(ids[p]==tokenizer.mask_token_id for p in markers)
    return {'ids':ids,'types':types,'markers':markers,'keys':keys,'state_tokens_dropped':dropped,
            'option_tokens_dropped':option_tokens_dropped,'option_token_budget':option_budget}


def collate(packed,pad_id,targets=None):
    batch=len(packed); length=max(len(x['ids']) for x in packed); options=max(len(x['markers']) for x in packed)
    ids=torch.full((batch,length),pad_id,dtype=torch.long); attention=torch.zeros((batch,length),dtype=torch.long)
    types=torch.zeros((batch,length),dtype=torch.long); positions=torch.full((batch,options),-1,dtype=torch.long)
    marker_mask=torch.zeros((batch,options),dtype=torch.bool)
    for i,item in enumerate(packed):
        n=len(item['ids']); k=len(item['markers'])
        ids[i,:n]=torch.tensor(item['ids']); attention[i,:n]=1;types[i,:n]=torch.tensor(item['types'])
        positions[i,:k]=torch.tensor(item['markers']);marker_mask[i,:k]=True
    result={'input_ids':ids,'attention_mask':attention,'token_type_ids':types,
            'marker_positions':positions,'marker_mask':marker_mask}
    if targets is not None: result['targets']=torch.tensor(targets,dtype=torch.long)
    return result


def save_head(model,path,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    torch.save(model.trainable_state(),path/'head.pt')
    (path/'config.json').write_text(json.dumps(dict(metadata,head_layers=model.head_layers),ensure_ascii=False,indent=2),encoding='utf-8')


def load_trained(model_dir,checkpoint):
    config=json.loads((Path(checkpoint)/'config.json').read_text('utf-8'))
    model=KoBertMaskDecision(model_dir,head_layers=config['head_layers'])
    missing,unexpected=model.load_state_dict(torch.load(Path(checkpoint)/'head.pt',map_location='cpu',weights_only=True),strict=False)
    if unexpected or any(not k.startswith('encoder.') for k in missing):
        raise RuntimeError(f'Invalid head checkpoint; missing={missing}, unexpected={unexpected}')
    return model,config
