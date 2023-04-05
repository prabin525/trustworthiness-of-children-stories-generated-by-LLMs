import os
import json


def process_real_stories(story_dirs):
    all_stories = []
    id = 0
    for story_dir in story_dirs:
        for file_name in os.listdir(story_dir):
            story_data = {
                'id': id,
                'text': ''
            }
            id += 1
            file = open(story_dir+'/'+file_name)
            for i, line in enumerate(file):
                if i == 0:
                    story_data['title'] = line
                elif i == 1:
                    continue
                else:
                    story_data['text'] += line
            all_stories.append(story_data)
    return all_stories


if __name__ == '__main__':
    story_dirs = [
        'books/Project Gutenburg/'
        'Favorite Fairy Tales by Logan Marshall',
    ]

    all_stories = process_real_stories(story_dirs)
    out_file = 'books/real_processed.json'
    with open(out_file, 'w+') as of:
        json.dump(all_stories, of)
