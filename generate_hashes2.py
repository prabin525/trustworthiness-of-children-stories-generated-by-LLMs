import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

import stanza
# import conllu
import networkx as nx
from stanza.utils.conll import CoNLL
from conllu import parse_tree_incr


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


def draw_dep_tree(UDtree):
    g = ud_2_graph(UDtree)
    pos = nx.spring_layout(g)
    nx.draw(g, pos)
    nx.draw_networkx_labels(
        g,
        pos,
        labels={node: g.nodes[node]['name'] for node in g.nodes}
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels={edge: g.edges[edge]['deprel'] for edge in g.edges}
    )
    plt.show()


def pred_succ_subgraph(graph, node, succ, pred):
    ps_list = []
    ps_list.append(node)
    try:
        s = next(succ)
        ps_list.append(s)
        done_looping_s = False
        while not done_looping_s:
            try:
                ps_list.append(next(succ))
            except StopIteration:
                done_looping_s = True
    except StopIteration:
        pass
    try:
        p = next(pred)
        ps_list.append(p)
    except StopIteration:
        pass
    return graph.subgraph(ps_list)


def get_hashes(stories):
    sen_lengths = []
    hashes = []
    sub_tree_hashes = []
    for each in stories:
        doc = nlp(each)
        CoNLL.write_doc2conll(doc, "output.conllu")
        for sen in doc.sentences:
            sen_lengths.append(len(sen.words))

        with open('output.conllu') as f:
            sentences = parse_tree_incr(f)
            sentences = list(sentences)

        for each in sentences:
            directed_tree = ud_2_graph(each)
            hashes.append(
                nx.weisfeiler_lehman_graph_hash(directed_tree)
            )
            for node in directed_tree.nodes(data=True):
                try:
                    succ = directed_tree.successors(node[0])
                    pred = directed_tree.predecessors(node[0])
                    subgraph = pred_succ_subgraph(
                        directed_tree,
                        node[0],
                        succ,
                        pred
                    )
                    if not nx.is_empty(subgraph):
                        sub_tree_hashes.append(nx.weisfeiler_lehman_graph_hash(
                            subgraph,
                            edge_attr='deprel',
                            node_attr='upos'
                        ))
                except AttributeError:
                    pass
    return sen_lengths, hashes, sub_tree_hashes


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
    sen_lengths, hashes, sub_tree_hashes = get_hashes(original_stories)

    json.dump(
        sen_lengths,
        open('sen_lengths2/original.json', 'w+')
    )

    json.dump(
        hashes,
        open('hashes2/original.json', 'w+')
    )

    json.dump(
        sub_tree_hashes,
        open('sub_tree_hashes2/original.json', 'w+')
    )

    # Get Sentence Length and hash of modern stories
    sen_lengths, hashes, sub_tree_hashes = get_hashes(modern_stories)

    json.dump(
        sen_lengths,
        open('sen_lengths2/modern.json', 'w+')
    )

    json.dump(
        hashes,
        open('hashes2/modern.json', 'w+')
    )

    json.dump(
        sub_tree_hashes,
        open('sub_tree_hashes2/modern.json', 'w+')
    )

    # OPT
    p_lengths = list(set(all_stories_opt.p_length.to_list()))
    for each in p_lengths:
        stories = all_stories_opt.loc[
            all_stories_opt.p_length == each
        ].gen_text.to_list()

        sen_lengths, hashes, sub_tree_hashes = get_hashes(stories)

        json.dump(
            sen_lengths,
            open(f'sen_length2/opt_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes2/opt_{each}.json', 'w+')
        )

        json.dump(
            sub_tree_hashes,
            open(f'sub_tree_hashes2/opt_{each}.json', 'w+')
        )

    # LLaMA
    p_lengths = list(set(all_stories_llama.p_length.to_list()))
    for each in p_lengths:
        stories = all_stories_llama.loc[
            all_stories_llama.p_length == each
        ].gen_text.to_list()

        sen_lengths, hashes, sub_tree_hashes = get_hashes(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths2/llama_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes2/llama_{each}.json', 'w+')
        )

        json.dump(
            sub_tree_hashes,
            open(f'sub_tree_hashes2/llama_{each}.json', 'w+')
        )

    # Alpaca title fixed
    t_types = set(all_stories_alpaca.t_type.to_list())

    for each in t_types:
        stories = all_stories_alpaca.loc[
            all_stories_alpaca.t_type == each
        ].gen_text.to_list()

        sen_lengths, hashes, sub_tree_hashes = get_hashes(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths2/alpaca_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashe2/alpaca_{each}.json', 'w+')
        )

        json.dump(
            sub_tree_hashes,
            open(f'sub_tree_hashes2/alpaca_{each}.json', 'w+')
        )

    # Alpaca free

    t_types = set(free_stories_alpaca.t_type.to_list())

    for each in t_types:
        stories = free_stories_alpaca.loc[
            free_stories_alpaca.t_type == each
        ].gen_text.to_list()

        sen_lengths, hashes, sub_tree_hashes = get_hashes(stories)

        json.dump(
            sen_lengths,
            open(f'sen_lengths2/alpaca_free_{each}.json', 'w+')
        )

        json.dump(
            hashes,
            open(f'hashes2/alpaca_free_{each}.json', 'w+')
        )

        json.dump(
            sub_tree_hashes,
            open(f'sub_tree_hashes2/alpaca_free_{each}.json', 'w+')
        )
