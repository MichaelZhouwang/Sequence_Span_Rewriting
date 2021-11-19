import transformers
import torch
import random
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from mosestokenizer  import MosesTokenizer,MosesDetokenizer
import argparse
import numpy as np
from tqdm import tqdm
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def mask_span(sentence, tokenizer, ratio=0.075):
    
    def verify(mask_span_start_id,span_length,masked_positions,seq_len):
    
        flag = True

        if mask_span_start_id-1 in masked_positions:
            return False

        for i in range(span_length+1):

            if mask_span_start_id+i in masked_positions or mask_span_start_id+i >= seq_len:
                flag = False
        return flag
    
    #tokenized_input = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    #input_tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
    with MosesTokenizer('en') as tokenize:
        input_tokens = tokenize(sentence)
    seq_len = len(input_tokens)
    
    sample_prob = torch.ones(seq_len)
    sample_prob /= torch.sum(sample_prob)
    
    num_spans = max(1, random.randint(round(seq_len*ratio)-1,round(seq_len*ratio)+1))
    masked_positions = []
    masked_start_positions = []
    masked_span_lengths = []
    total_span_length = 0
    for i in range(num_spans):
        span_length = max(1,np.random.poisson(3))
        mask_span_start_id = sample_prob.multinomial(1)
        trials = 0
        while not verify(mask_span_start_id,span_length,masked_positions,seq_len) and trials <= 10:
            mask_span_start_id = sample_prob.multinomial(1)
            trials += 1
        if trials >= 10:
            break
        for i in range(span_length):
            masked_positions.append(mask_span_start_id+i)
        masked_start_positions.append(mask_span_start_id)
        masked_span_lengths.append(span_length)
        total_span_length += span_length
        
    new_tokens = []
    masked_tokens = []
    #labels = []
    span_idx = 0
    for idx in range(seq_len):
        if idx in masked_start_positions:
            new_tokens.append('<extra_id_'+str(span_idx)+'>')
            masked_tokens.append(input_tokens[idx])
            if idx+1 not in masked_positions:
                masked_tokens.append('<extra_id_'+str(span_idx+1)+'>')
            span_idx += 1
        elif idx in masked_positions:
            masked_tokens.append(input_tokens[idx])
            if idx+1 not in masked_positions:
                masked_tokens.append('<extra_id_'+str(span_idx)+'>')
        else:
            new_tokens.append(input_tokens[idx])
    with MosesDetokenizer('en') as detokenize:
        new_tokens_str = detokenize(new_tokens)
    #print(masked_tokens)
    #print(input_tokens)
    #print(new_tokens_str)
    return new_tokens_str, total_span_length, masked_tokens, new_tokens


def build_filled_inputs(original_tokens, filled_tokens):
    
    valid = True
    target_tokens = []
    input_tokens = []
    
    filling_index = 0
    
    for token in original_tokens:
        
        if token.startswith('<extra_id'):
            input_tokens.append(token)
            if filling_index >= len(filled_tokens):
                valid = False
                break
            
            if filled_tokens[filling_index].startswith('<extra_id'):
                valid = False
            while not filled_tokens[filling_index].startswith('<extra_id'):
                target_tokens.append(filled_tokens[filling_index])
                input_tokens.append(filled_tokens[filling_index])
                filling_index += 1
                
                if filling_index >= len(filled_tokens):
                    valid = False
                    break
            if filling_index >= len(filled_tokens):
                valid = False
                break    
            target_tokens.append(filled_tokens[filling_index])
            input_tokens.append(filled_tokens[filling_index])
            filling_index += 1
        else:
            input_tokens.append(token)
    return target_tokens, input_tokens, valid

def generate_one_batch(sentences,tokenizer,t5_model):
    
    batch_size = len(sentences)
    
    masked_texts = []
    masked_texts_list = []
    target_texts = []
    max_length = 0
    for text in sentences:
        #print(text)
        masked_text, length, masked_tokens, masked_text_list = mask_span(text.strip(),tokenizer)
        max_length = max(max_length,length)
        masked_texts.append(masked_text)
        target_texts.append(masked_tokens)
        masked_texts_list.append(masked_text_list)
        
    #print('masked')
    
    input_ids = tokenizer.batch_encode_plus(masked_texts, add_special_tokens=True, return_tensors='pt',padding=True)
    #print('encoded')
    
    
    outputs = t5_model.generate(input_ids=input_ids['input_ids'].to(DEVICE), attention_mask=input_ids['attention_mask'].to(DEVICE), do_sample=True, top_p=0.9,
                          num_return_sequences=10,
                          max_length=round(max_length*2))
    
    #print('generated')
    
    outputs = outputs.reshape(batch_size,10,-1).data # 10 is num_return_sequences
    
    input_text = []
    #masked_input = []
    output_text = []
    #valids = []
    
    for i in range(batch_size):
        
        #outputs_i = tokenizer.convert_ids_to_tokens(outputs[i,0,2:])
        #original_inputs = tokenizer.convert_ids_to_tokens(input_ids['input_ids'][i])
        outputs_i = tokenizer.batch_decode(outputs[i,0,2:], skip_special_tokens=False, clean_up_tokenization_spaces=True)
        original_inputs = tokenizer.batch_decode(input_ids['input_ids'][i], skip_special_tokens=False, clean_up_tokenization_spaces=True)
        target_tokens, input_tokens, valid = build_filled_inputs(masked_texts_list[i],outputs_i)

        if valid:
            input_text.append(' '.join(input_tokens))
            
            #original_text.append(' '.join(tokenizer.tokenize(sentences[i])))
            #masked_input.append(' '.join(original_inputs))
            #output_text.append(' '.join([ tok[1:] if tok.startswith('_') else tok for tok in target_tokens]))
            output_text.append(' '.join(target_texts[i]))
        #valids.append(valid)
        

    return input_text, output_text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T5_PATH = 't5-large' # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU

t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
t5_config = T5Config.from_pretrained(T5_PATH)
t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

parser = argparse.ArgumentParser(description='input path')
parser.add_argument("--input_file", type=str, default=None, required=True)
parser.add_argument("--id", type=str, default='0', required=False)
parser.add_argument("--batch_size", type=int, default=8, required=False)
args = parser.parse_args()
if os.path.exists('track'+args.id):
    track_file = open('track'+args.id,'r',encoding='utf8')
    lines = track_file.readlines()
    if len(lines) <= 1:
        finished = 0
    else:
        finished = int(lines[-1].strip())
    track_file.close()
else:
    finished = 0
print(finished)
input_lines = open(args.input_file+str(args.id),'r',encoding='utf8').readlines()[finished:]
with open('track'+args.id,'a+',encoding='utf8') as tracker:
    with open('train_'+str(args.id)+'.src','a+',encoding='utf8') as writer_src:
        with open('train_'+str(args.id)+'.tgt','a+',encoding='utf8') as writer_tgt:
            with MosesDetokenizer('en') as detokenize:
                for examples_chunk in tqdm(list(chunks(input_lines, args.batch_size))):
                    finished += args.batch_size
                    tracker.write(str(finished)+'\n')
                    try:
                        input_text, output_text = generate_one_batch(examples_chunk,t5_tokenizer,t5_mlm) 
                        for input_line, output_line in zip(input_text,output_text):
                            writer_src.write(detokenize(input_line.strip().split(' '))+'\n')
                            writer_tgt.write(detokenize(output_line.strip().split(' '))+'\n')
                    except Exception as e:
                        pass
                    continue
    
