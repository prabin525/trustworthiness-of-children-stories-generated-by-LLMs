import argparse
import json
import pandas as pd

import stanza
# import conllu
from transformers import pipeline


detoxify_pipeline = pipeline(
    'text-classification',
    model='unitary/toxic-bert',
    tokenizer='bert-base-uncased',
    function_to_apply='sigmoid',
    return_all_scores=True
)


def get_toxicity_measures(stories):
    toxicty_measure = []
    for s in stories:
        try:
            a = detoxify_pipeline(s.split('.'))
            toxicty_measure.extend(a)
        except:
            print('Hi')
            pass
    return toxicty_measure


nlp = stanza.Pipeline('en')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate stories with context'
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
        books_loc = 'books/from_server'
    else:
        books_loc = 'books'

    # Create a dataframe with all stories and their generations

    all_stories = json.load(open('books/real_processed.json'))
    modern_stories = json.load(open('books/modern_processed.json'))
    modern_stories = pd.DataFrame(modern_stories)
    all_stories_opt = pd.DataFrame(all_stories)
    all_stories_llama = pd.DataFrame(all_stories)
    all_stories_alpaca = pd.DataFrame(all_stories)

    # Load OPT stories
    gen_stories = json.load(open(f'{books_loc}/gen_stories_opt.json'))
    gen_stories = pd.DataFrame(gen_stories)
    all_stories_opt = all_stories_opt.merge(gen_stories, on=['id'], how='left')

    # Load LLaMA stories
    gen_stories = json.load(open(f'{books_loc}/gen_stories_llama.json'))
    gen_stories = pd.DataFrame(gen_stories)
    all_stories_llama = all_stories_llama.merge(
        gen_stories,
        on=['id'],
        how='left'
    )

    # Load Alpaca stories
    gen_stories = json.load(open(f'{books_loc}/gen_stories_alpaca.json'))
    gen_stories = pd.DataFrame(gen_stories)
    all_stories_alpaca = all_stories_alpaca.merge(
        gen_stories,
        on=['id'],
        how='left'
    )
    free_stories_alpaca = gen_stories.loc[gen_stories.gen_id.isnull()].copy()

    original_stories = list(set(all_stories_opt.text.to_list()))
    modern_stories = list(set(modern_stories.text.to_list()))

    # Get Sentence Length and hash of original stories
    toxic_measures = get_toxicity_measures(original_stories)

    json.dump(
        toxic_measures,
        open('toxic_measures/original.json', 'w+')
    )

    # Get Sentence Length and hash of modern stories
    toxic_measures = get_toxicity_measures(modern_stories)

    json.dump(
        toxic_measures,
        open('toxic_measures/modern.json', 'w+')
    )

    # OPT
    p_lengths = list(set(all_stories_opt.p_length.to_list()))
    for each in p_lengths:
        stories = all_stories_opt.loc[
            all_stories_opt.p_length == each
        ].gen_text.to_list()

        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            toxic_measures,
            open(f'toxic_measures/opt_{each}.json', 'w+')
        )

    # LLaMA
    p_lengths = list(set(all_stories_llama.p_length.to_list()))
    for each in p_lengths:
        stories = all_stories_llama.loc[
            all_stories_llama.p_length == each
        ].gen_text.to_list()

        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            toxic_measures,
            open(f'toxic_measures/llama_{each}.json', 'w+')
        )

    # Alpaca title fixed
    t_types = set(all_stories_alpaca.t_type.to_list())

    for each in t_types:
        stories = all_stories_alpaca.loc[
            all_stories_alpaca.t_type == each
        ].gen_text.to_list()

        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            toxic_measures,
            open(f'toxic_measures/alpaca_{each}.json', 'w+')
        )

    # Alpaca free
    t_types = set(free_stories_alpaca.t_type.to_list())

    for each in t_types:
        stories = free_stories_alpaca.loc[
            free_stories_alpaca.t_type == each
        ].gen_text.to_list()

        toxic_measures = get_toxicity_measures(stories)
        json.dump(
            toxic_measures,
            open(f'toxic_measures/alpaca_free_{each}.json', 'w+')
        )
