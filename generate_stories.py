import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)


def gen_opt_stories(
        model,
        tokenizer,
        device,
        model_name,
        real_stories,
        num_return_sequence=1,
        top_k=16,
):
    generated_stories = []

    for each in real_stories:
        gen_id = 0
        print(each['id'])
        tokenized_text = tokenizer.encode(each['text'], return_tensors='pt')
        # story_size = tokenized_text.shape[1]

        # First sentence as context
        first_line = f"{each['text'].split('.')[0]}."
        tokenized_first_line = tokenizer.encode(
            first_line,
            return_tensors='pt'
        )
        tokenized_first_line = tokenized_first_line.to(device)
        gen = model.generate(
            tokenized_first_line,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=num_return_sequence,
            max_length=1024
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for each_s in outputs:
            generated_stories.append({
                'id': each['id'],
                'gen_id': gen_id,
                'p_length': 'first_line',
                'model_name': model_name,
                'gen_text': each_s,
                'prompt': first_line
            })
            gen_id += 1

        # First 256 tokens as context
        prompt = tokenizer.batch_decode(tokenized_text[:, :256])
        tokenized_input = tokenized_text[:, :256]
        tokenized_input = tokenized_input.to(device)
        gen = model.generate(
            tokenized_input,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=num_return_sequence,
            max_length=1024
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for each_s in outputs:
            generated_stories.append({
                'id': each['id'],
                'gen_id': gen_id,
                'p_length': '256_tokens',
                'model_name': model_name,
                'gen_text': each_s,
                'prompt': prompt[0]
            })
            gen_id += 1

        # First 512 tokens as context
        prompt = tokenizer.batch_decode(tokenized_text[:, :512])
        tokenized_input = tokenized_text[:, :512]
        tokenized_input = tokenized_input.to(device)
        gen = model.generate(
            tokenized_input,
            do_sample=True,
            top_k=top_k,
            num_return_sequences=num_return_sequence,
            max_length=1024
        )
        outputs = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        for each_s in outputs:
            generated_stories.append({
                'id': each['id'],
                'gen_id': gen_id,
                'p_length': '512_tokens',
                'model_name': model_name,
                'gen_text': each_s,
                'prompt': prompt[0]
            })
            gen_id += 1

        # 10% as context
        # prompt = tokenizer.batch_decode(tokenized_text[:, :story_size//10])
        # tokenized_input = tokenized_text[:, :story_size//10]
        # tokenized_input = tokenized_input.to(device)
        # gen = model.generate(
        #     tokenized_input,
        #     do_sample=True,
        #     top_k=top_k,
        #     num_return_sequences=num_return_sequence,
        #     max_length=story_size/story_size_divider
        # )
        # outputs = tokenizer.batch_decode(
        #     gen,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False
        # )
        # for each_s in outputs:
        #     generated_stories.append({
        #         'id': each['id'],
        #         'gen_id': gen_id,
        #         'p_length': '10_percentage',
        #         'model_name': model_name,
        #         'gen_text': each_s,
        #         'prompt': prompt[0]
        #     })
        #     gen_id += 1

    return generated_stories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate stories with context'
    )
    parser.add_argument(
        '--model',
        dest='model_name',
        choices=[
            'opt',
            'llama'
        ],
        required=True,
    )
    parser.add_argument(
        '--stories_loc',
        dest='stories_loc',
        default='books/real_processed.json'
    )
    parser.add_argument(
        '--out_loc',
        dest='out_loc',
        default='books/'
    )
    parser.add_argument(
        '--local',
        dest='local',
        type=bool,
        default=False
    )
    args = parser.parse_args()
    print(args)
    if args.local:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                        "MPS not available because the current PyTorch install"
                        " was not built with MPS enabled."
                    )
            else:
                print(
                        "MPS not available because the current MacOS version "
                        "is not 12.3+ and/or you do not have an MPS-enabled "
                        "device on this machine."
                    )

        else:
            device = torch.device("mps")
        num_return_sequences = 1
        top_k = 16
    else:
        device = torch.device("cuda:0")
        num_return_sequences = 10
        top_k = 100
    real_stories = json.load(open(args.stories_loc))

    if args.model_name == 'opt':
        if args.local:
            model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        else:
            model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
        model.to(device)
        generated_stories = gen_opt_stories(
            model,
            tokenizer,
            device,
            args.model_name,
            real_stories,
            num_return_sequences,
            top_k,
        )
        json.dump(
            generated_stories,
            open(f'{args.out_loc}gen_stories_opt.json', 'w+')
        )
    elif args.model_name == 'llama':
        if args.local:
            model = LlamaForCausalLM.from_pretrained(
                '/scratch/pbhanda2/projects/llama/hf_weights'
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                '/scratch/pbhanda2/projects/llama/hf_weights'
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                '/scratch/pbhanda2/projects/llama/hf_weights'
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                '/scratch/pbhanda2/projects/llama/hf_weights'
            )
        model.to(device)
        generated_stories = gen_opt_stories(
            model,
            tokenizer,
            device,
            args.model_name,
            real_stories,
            num_return_sequences,
            top_k,
        )
        json.dump(
            generated_stories,
            open(f'{args.out_loc}gen_stories_llama.json', 'w+')
        )
