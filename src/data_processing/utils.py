
def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def flatten(input_list):
    return [item for sublist in input_list for item in sublist]


