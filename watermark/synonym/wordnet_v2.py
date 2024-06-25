from nltk.corpus import wordnet

def get_most_similar_synonym(word, pos=None):
    """
    Get the most similar synonym for the input word with the given part of speech (if specified).
    Return the synonym and its part of speech.
    If the word has multiple parts of speech, return None.
    """
    synsets = wordnet.synsets(word)
    if len(synsets) == 0:
        return None

    if pos is not None:
        # Filter synsets by the given part of speech
        synsets = [synset for synset in synsets if synset.pos() == pos]
    
    # Return None if the word has multiple parts of speech
    if len(set([synset.pos() for synset in synsets])) > 1:
        return None
    
    # Get the most similar synonym and its part of speech
    synonym = None
    max_similarity = -1
    word_synset = synsets[0]
    for synset in synsets:
        similarity = word_synset.path_similarity(synset)
        if similarity is not None and similarity > max_similarity and word != synset.name().split('.')[0]:
            max_similarity = similarity
            synonym = synset.name().split('.')[0]
    
    if synonym is None:
        return None
    else:
        return f"{word} ({word_synset.pos()}) -> {synonym} ({word_synset.pos()})"


print (get_most_similar_synonym('car'))
print (get_most_similar_synonym('hair'))
print (get_most_similar_synonym('like'))