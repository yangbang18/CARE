PAD = 0
UNK = 1
BOS = 2
EOS = 3
MASK = 4
VIS = 5

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
MASK_WORD = '<mask>'
VIS_WORD = '<vis>'

attribute_start = 6
attribute_end = 3006

base_checkpoint_path = './exps'	# base path to save checkpoints
base_data_path = '/data/video_datasets' # base path to load corpora and features

opt_filename = 'opt_info.json' # the filename to save training/model options (settings)

# the maximum number of uniformly sampled frames to represent a video
# this is used in feature extraction and dataloader
n_total_frames = 60

# mapping of nltk pos tags
pos_tag_mapping = {}
content = [
    [["``", "''", ",", "-LRB-", "-RRB-", ".", ":", "HYPH", "NFP"], "PUNCT"],
    [["$", "SYM"], "SYM"],
    [["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"], "VERB"],
    [["WDT", "WP$", "PRP$", "DT", "PDT"], "DET"],
    [["NN", "NNP", "NNPS", "NNS"], "NOUN"],
    [["WP", "EX", "PRP"], "PRON"],
    [["JJ", "JJR", "JJS", "AFX"], "ADJ"],
    [["ADD", "FW", "GW", "LS", "NIL", "XX"], "X"],
    [["SP", "_SP"], "SPACE"], 
    [["RB", "RBR", "RBS","WRB"], "ADV"], 
    [["IN", "RP"], "ADP"], 
    [["CC"], "CCONJ"],
    [["CD"], "NUM"],
    [["POS", "TO"], "PART"],
    [["UH"], "INTJ"]
]
for item in content:
    ks, v = item
    for k in ks:
        pos_tag_mapping[k] = v


index2category = {
    0: 'music', 
    1: 'people', 
    2: 'gaming', 
    3: 'sports/actions', 
    4: 'news/events/politics', 
    5: 'education', 
    6: 'tv-shows', 
    7: 'movie/comedy',
    8: 'animation',
    9: 'vehicles/autos',
    10: 'how-to',
    11: 'travel',
    12: 'science/technology',
    13: 'animals/pets',
    14: 'kids/family',
    15: 'documentary',
    16: 'food/drink',
    17: 'cooking',
    18: 'beauty/fashion',
    19: 'advertisement'
}

flag2modality = {
    'I': 'i',
    'IT': 'ir',
    'V': 'mi',
    'VA': 'ami',
    'VAT': 'amir',
    'VT': 'mir',
    'A': 'a',
    'T': 'r',
}
