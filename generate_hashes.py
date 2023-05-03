import argparse
import json
import pandas as pd

import stanza
# import conllu
import networkx as nx
from stanza.utils.conll import CoNLL
from conllu import parse_tree_incr
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
        a = detoxify_pipeline(s.split('.'))
        toxicty_measure.extend(a)
    return toxicty_measure


nlp = stanza.Pipeline('en')


def ud_2_graph(tree, parent=1, graph=None):
    # Convert UD to networkx graph
    if graph is None:
        graph = nx.Graph()
        graph.add_node(graph.number_of_nodes(), name='root', upos='ROOT')
        graph.add_node(
            graph.number_of_nodes(),
            name=tree.token['form'],
            upos=tree.token['upos']
        )
        graph.add_edge(0, parent, deprel='<root>')
    for child in tree.children:
        child_num = graph.number_of_nodes()
        graph.add_node(
            child_num,
            name=child.token['form'],
            upos=child.token['upos']
        )
        graph.add_edge(parent, child_num, deprel=child.token['deprel'])
        graph = ud_2_graph(child, child_num, graph)
    return graph


def get_hashes(stories):
    sen_lengths = []
    hashes = []
    for each in stories:
        doc = nlp(each)
        CoNLL.write_doc2conll(doc, "output.conllu")
        for sen in doc.sentences:
            sen_lengths.append(len(sen.words))

        with open('output.conllu') as f:
            sentences = parse_tree_incr(f)
            sentences = list(sentences)

        for each in sentences:
            hashes.append(
                nx.weisfeiler_lehman_graph_hash(ud_2_graph(each))
            )
    return sen_lengths, hashes


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

    # Get Sentence Length and hash of original stories
    sen_lengths, hashes = get_hashes(original_stories)
    toxic_measures = get_toxicity_measures(original_stories)

    json.dump(
        sen_lengths,
        open('sen_lengths/original.json', 'w+')
    )

    json.dump(
        hashes,
        open('hashes/original.json', 'w+')
    )

    json.dump(
        toxic_measures,
        open('toxic_measures/original.json', 'w+')
    )

    # OPT
    p_lengths = list(set(all_stories_opt.p_length.to_list()))
    for each in p_lengths:
        stories = all_stories_opt.loc[
            all_stories_opt.p_length == each
        ].gen_text.to_list()

        sen_lengths, hashes = get_hashes(stories)
        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths/opt_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes/opt_{each}.json', 'w+')
        )

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

        sen_lengths, hashes = get_hashes(stories)
        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths/llama_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes/llama_{each}.json', 'w+')
        )

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

        sen_lengths, hashes = get_hashes(stories)
        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths/alpaca_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes/alpaca_{each}.json', 'w+')
        )

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

        sen_lengths, hashes = get_hashes(stories)
        toxic_measures = get_toxicity_measures(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths/alpaca_free_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes/alpaca_free_{each}.json', 'w+')
        )

        json.dump(
            sen_lengths,
            open(f'toxic_measures/alpaca_free_{each}.json', 'w+')
        )
