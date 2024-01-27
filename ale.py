main_characters = ['Rachel', 'Monica', 'Phoebe', 'Joey', 'Chandler', 'Ross']

def concatenate_and_tokenize(phrases, speakers):
    combined_phrases = []
    
    for i in range(0, len(phrases), 4):
        if i + 2 < len(phrases):
            speaker_token = f'[{speakers[i]}]' if speakers[i] in main_characters else None
            combined_phrases.append(['[CLS]', speaker_token, '[says]', phrases[i], phrases[i + 2], '[SEP]'])
        else:
            speaker_token = f'[{speakers[i]}]' if speakers[i] in main_characters else None
            combined_phrases.append(['[CLS]', speaker_token, '[says]', phrases[i], '[SEP]'])
        if i + 1 < len(phrases) and i + 3 < len(phrases):
            speaker_token = f'[{speakers[i + 3]}]' if speakers[i + 3] in main_characters else None
            combined_phrases.append(['[CLS]', speaker_token, '[says]', phrases[i + 3], '[SEP]'])
        elif i + 1 < len(phrases):
            speaker_token = f'[{speakers[i + 1]}]' if speakers[i + 1] in main_characters else None
            combined_phrases.append(['[CLS]', speaker_token, '[says]', phrases[i + 1], '[SEP]'])
    
    return combined_phrases

def process_dataframe(df):
    df['utterances'] = df.apply(lambda row: concatenate_and_tokenize(row['utterances'], row['speakers']), axis=1)
    return df
